import math
import random

from flash_attn import flash_attn_func
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, AutoTokenizer, TrainerCallback, TrainerState, TrainerControl
from transformers.modeling_outputs import CausalLMOutputWithPast


# rms归一化
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# 旋转位置编码
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)

    #
# @torch.no_grad()
# def llama_rotary_embedding(length, config):
#     inv_freq = torch.arange(0, config.head_dim, 2) / config.head_dim
#     inv_freq = 1 / (config.rope_theta ** inv_freq)
#     inv_freq = inv_freq.reshape(config.head_dim // 2, 1)
#
#     position_ids = torch.arange(length).reshape(1, length).float()
#     freq = inv_freq.matmul(position_ids).transpose(0, 1)
#     emb = torch.cat((freq, freq), -1)
#     return emb.cos(), emb.sin()
#
#
# def apply_rotary_pos_emb(x, cos, sin, config):
#     def rotate_half(x):
#         left = x[..., :config.head_dim // 2]
#         right = -x[..., config.head_dim // 2:]
#         return torch.cat((right, left), -1)
#
#     return x * cos + rotate_half(x) * sin
#
#
# class LlamaRMSNorm(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.rms_norm_eps = config.rms_norm_eps
#         self.weight = torch.nn.Parameter(torch.ones(config.hidden_size))
#
#     def forward(self, x):
#         var = x.pow(2).mean(-1, keepdim=True)
#         x = x * (var + self.rms_norm_eps).rsqrt()
#         return self.weight * x

def get_causal_mask(attention_mask):
    """
    attention_mask = torch.tensor([
        [1, 1, 0, 0, 0],  # 第一个序列
        [1, 1, 1, 1, 1] ,  # 第二个序列
    ])
    tensor([[[[ 0.0000e+00, -1.0000e+15, -1.0000e+15, -1.0000e+15, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00, -1.0000e+15, -1.0000e+15, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00, -1.0000e+15, -1.0000e+15, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00, -1.0000e+15, -1.0000e+15, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00, -1.0000e+15, -1.0000e+15, -1.0000e+15]]],


            [[[ 0.0000e+00, -1.0000e+15, -1.0000e+15, -1.0000e+15, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00, -1.0000e+15, -1.0000e+15, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+15, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0000e+15],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]]])
    :param attention_mask:
    :return:
    """
    B, L = attention_mask.shape
    min_value = -1e15
    causal_mask = torch.full((L, L), min_value).triu(diagonal=1)
    causal_mask = causal_mask.reshape(1, 1, L, L).repeat(B, 1, 1, 1)
    causal_mask = causal_mask.to(attention_mask.device)

    mask = attention_mask.reshape(B, 1, 1, L) == 0
    causal_mask = causal_mask.masked_fill(mask, min_value)
    return causal_mask


class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.num_attention_heads = config.num_attention_heads  # 总头数
        self.q_lora_rank = config.q_lora_rank  # q低秩压缩到的维度
        self.kv_lora_rank = config.kv_lora_rank  # kv低秩压缩到的维度
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # qk的总维度，不带旋转位置编码的维度加上带旋转位置编码的维度
        self.v_head_dim = config.v_head_dim  # value的维度，等于不带旋转位置编码的k维度
        self.max_seq_len = config.max_seq_len

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank)  # q 的降维矩阵
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.qk_head_dim)  # q的升维矩阵

        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)  # kv的降维矩阵
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank,
                               self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim))  # kv的升维矩阵

        self.wo = nn.Linear(self.num_attention_heads * self.v_head_dim, self.hidden_size)

        self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim, config.max_seq_len)  # 旋转旋转位置编码

    def forward(self, x):
        bs, seq_len, _ = x.shape
        q = self.wq_a(x)  # [bs, seq_len, q_lora_rank]
        q = self.q_norm(q)  # [bs, seq_len, q_lora_rank]
        q = self.wq_b(q)  # [bs, seq_len, n_heads * qk_head_dim]
        q = q.view(bs, seq_len, self.num_attention_heads, self.qk_head_dim)  # [bs, seq_len, n_heads, qk_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim],
                                   dim=-1)  # q_nope shape:[bs, seq_len, n_heads, qk_nope_head_dim] q_pe shape:[bs, seq_len, n_heads, qk_rope_head_dim]

        kv = self.wkv_a(x)  # [bs, seq_len, kv_lora_rank + qk_rope_head_dim]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],
                               dim=-1)  # kv shape:[bs, seq_len, kv_lora_rank] k_pe shape:[bs, seq_len, qk_rope_head_dim]

        k_pe = k_pe.unsqueeze(2)
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)  # [bs, seq_len, n_heads, qk_rope_head_dim]  [bs, seq_len, 1, qk_rope_head_dim]
        k_pe = k_pe.squeeze(2)  # [bs, seq_len, qk_rope_head_dim]

        wkv_b = self.wkv_b.weight  # [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        wkv_b = wkv_b.view(self.num_attention_heads, -1,  self.kv_lora_rank)  # [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

        kv = self.kv_norm(kv)

        scores_nope = torch.einsum("bshc,btc->bsht", q_nope, kv)  # [bs, seq_len, n_heads, kv_lora_rank] *[bs, seq_len, kv_lora_rank]= [bs, seq_len, n_heads, seq_len]
        scores_pe = torch.einsum("bshr,btr->bsht", q_pe, k_pe)  # [bs, seq_len, n_heads, qk_rope_head_dim] *  [bs, seq_len, qk_rope_head_dim] =  [bs, seq_len, n_heads, seq_len]
        scores = (scores_nope + scores_pe) / math.sqrt( self.qk_nope_head_dim + self.qk_rope_head_dim)  # [bs, seq_len, n_heads, seq_len]

        mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = mask[:, :, :seq_len, :seq_len].transpose(1, 2).to(scores.device)
        scores += mask

        scores = scores.softmax(dim=-1)
        x = torch.einsum("bsht,btc->bshc", scores, kv)  # [bs, seq_len, n_heads, seq_len]x[bs, seq_len, kv_lora_rank]=[bs, seq_len, n_heads, kv_lora_rank]
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:,
                                              -self.v_head_dim:])  # [bs, seq_len, n_heads, kv_lora_rank] x  [n_heads, v_head_dim, kv_lora_rank] = [bs, seq_len, n_heads, v_head_dim]
        x = x.contiguous().view(bs, seq_len, -1)  # [bs, seq_len, n_heads, v_head_dim] [bs, seq_len, n_heads*v_head_dim]
        x = self.wo(x)
        return x



def load_balancing_loss_func(
        gate_logits,
        num_experts,
        top_k):
    concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits],
                                         dim=0)  # 各个层的gate_logit进行合并[layers X batch_size X sequence_length, num_experts]
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num)

    def forward(self, x):
        # x dim: b, s, hidden_size
        logits = self.gate(x)  # gate: b, s, expert_num
        logits_topk, indices = logits.topk(self.topk, dim=-1)  # 选择概率最大的两个专家，返回两个专家对每个token的概率
        zeros = torch.full_like(logits, float("-inf"))  # 创建一个全为负无穷的矩阵，用于屏蔽其他专家的概率并重新归一化概率最大的两个专家
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # 将选择的两个专家的概率按指定索引填充
        sparse_logits = F.softmax(sparse_logits, dim=-1)  # 得到一个稀疏矩阵，选择的两个专家对每个token的概率和为1
        gate_logit = logits.view(-1, self.expert_num)

        return sparse_logits, indices, gate_logit


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.expert_num = config.expert_num
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size // self.expert_num, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size // self.expert_num, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size // self.expert_num, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)
        if self.config.shared_expert is not None:
            self.shared_experts = Expert(config)

    def forward(self, x):
        sparse_logits, indices, gate_logit = self.gating(x)
        final_outputs = torch.zeros_like(x)
        x_flat = x.view(-1, x.shape[-1])  # (batch_size * seq_len, dim)
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])  # (batch_size * seq_len, export_num))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)  # (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1)  # (batch_size * seq_len)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # (seq_true, dim)
                export_output = expert(expert_input)  # (seq_true, dim)
                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true) --> (seq_true, 1)
                weighted_output = export_output * gate_scores  # (seq_true, dim)
                final_outputs[expert_mask] += weighted_output
        if self.config.shared_expert:
            final_outputs = final_outputs + self.shared_experts(x)
        return final_outputs, gate_logit


class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MLA(config)
        self.moe = MoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, hidden_state):
        res = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state) + res
        res = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_states, gate_logit = self.moe(hidden_state)
        hidden_states += res
        return hidden_states, gate_logit


class LlamaForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk
        self.loss = None
        self.aux_loss = None

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None):

        all_router_logits = () if self.config.output_router_logits else None

        hidden_states = self.embed_tokens(input_ids)
        for idx, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states, gate_logit = checkpoint(layer, hidden_states)
            else:
                hidden_states, gate_logit = layer(hidden_states)
            if gate_logit is not None and all_router_logits is not None:
                all_router_logits += (gate_logit,)

        hidden_states = self.norm(hidden_states)

        if labels is not None:
            logits = self.lm_head(hidden_states)
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none').view(
                labels.size())
            self.loss = (self.loss * attention_mask).sum() / attention_mask.sum()
        else:
            logits = self.lm_head(hidden_states[:, [-1], :])
            self.loss = None

        if self.config.output_router_logits and labels is not None:
            self.aux_loss = load_balancing_loss_func(all_router_logits, self.expert_num, self.topk)
            self.loss += self.config.aux_loss_coef * self.aux_loss.to(self.loss.device)

        return CausalLMOutputWithPast(self.loss, logits)

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        self.gradient_checkpointing = False

    @torch.inference_mode
    def generate(self, input_ids, eos_token_id, max_new_length=10, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1., ):
        old = input_ids.shape[1]
        while input_ids.shape[1] < min(old + max_new_length, self.config.max_seq_len) - 1:
            inference_res = self(input_ids)
            logits = inference_res.logits  # [b, s, d]
            logits = logits[:, -1, :]  # [b, d]

            for token in set(input_ids.tolist()[0]):
                logits[:, token] /= repetition_penalty
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)  # [b, topk]
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)
            if idx_next == eos_token_id:
                break
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        return input_ids


class GenerateTextCallback(TrainerCallback):
    def __init__(self, tokenizer, generate_every=500, max_new_length=50):
        self.tokenizer = tokenizer
        self.generate_every = generate_every
        self.max_new_length = max_new_length

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step % self.generate_every == 0:
            model.eval()
            prefix = ["请帮助我", "写一首", "给我一些", "你知道"]
            input_ids = self.tokenizer(self.tokenizer.bos_token + random.choice(prefix), return_tensors="pt")["input_ids"].to(
                model.device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_new_length=self.max_new_length,
                                         eos_token_id=self.tokenizer.eos_token_id, )
            decode = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Step {state.global_step}: Generated text: {decode}")
            model.train()


class ChatCallback(TrainerCallback):
    def __init__(self, tokenizer,  generate_every=500, max_new_length=50):
        self.tokenizer = tokenizer
        self.generate_every = generate_every
        self.max_new_length = max_new_length

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        prompt = ["请帮我写一个关于小狗的故事", "给我健身建议", "介绍杭州的热门景区"]
        messages = [{"role": 'user', "content": random.choice(prompt)}]
        new_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = torch.tensor(self.tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)
        if state.global_step % self.generate_every == 0:
            model.eval()
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_new_length=self.max_new_length,
                                         eos_token_id=self.tokenizer.eos_token_id, )
            decode = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Step {state.global_step}: Generated text: {decode}")
            model.train()

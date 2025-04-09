import math
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, TrainerCallback, \
    TrainerState, TrainerControl, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from llama_moe_mla.model.llama_moe_mla_config import LMConfig


@torch.no_grad()
def llama_rotary_embedding(length, config):
    inv_freq = torch.arange(0, config.head_dim, 2) / config.head_dim
    inv_freq = 1 / (config.rope_theta ** inv_freq)
    inv_freq = inv_freq.reshape(config.head_dim // 2, 1)

    position_ids = torch.arange(length).reshape(1, length).float()
    freq = inv_freq.matmul(position_ids).transpose(0, 1)
    emb = torch.cat((freq, freq), -1)
    return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cos, sin, config):
    def rotate_half(x):
        left = x[..., :config.head_dim // 2]
        right = -x[..., config.head_dim // 2:]
        return torch.cat((right, left), -1)

    return x * cos + rotate_half(x) * sin


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.weight = torch.nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * (var + self.rms_norm_eps).rsqrt()
        return self.weight * x


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        left = self.act_fn(self.gate_proj(x))
        right = self.up_proj(x)
        return self.down_proj(left * right)


def get_causal_mask(attention_mask):
    B, L = attention_mask.shape
    min_value = -1e15
    causal_mask = torch.full((L, L), min_value).triu(diagonal=1)
    causal_mask = causal_mask.reshape(1, 1, L, L).repeat(B, 1, 1, 1)
    causal_mask = causal_mask.to(attention_mask.device)

    mask = attention_mask.reshape(B, 1, 1, L) == 0
    causal_mask = causal_mask.masked_fill(mask, min_value)
    return causal_mask


class LlamaAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_proj = torch.nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_state):
        B, L, _ = hidden_state.shape
        cos, sin = llama_rotary_embedding(L, self.config)
        cos, sin = cos.to(hidden_state.device), sin.to(hidden_state.device)

        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)

        q = q.view(B, L, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(B, L, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(B, L, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin, self.config)
        k = apply_rotary_pos_emb(k, cos, sin, self.config)

        if self.config.flas_attention:
            q = q.transpose(1, 2).type_as(v)
            k = k.transpose(1, 2).type_as(v)
            v = v.transpose(1, 2).type_as(v)
            attn = flash_attn_func(q, k, v, causal=True)
            attn = attn.view(B, L, -1)
        else:
            k = k.unsqueeze(2).repeat(1, 1, self.config.num_attention_heads // self.config.num_key_value_heads, 1, 1).reshape(B, -1, L, self.config.head_dim)
            v = v.unsqueeze(2).repeat(1, 1, self.config.num_attention_heads // self.config.num_key_value_heads, 1, 1).reshape(B, -1, L, self.config.head_dim)

            attn = q @ k.transpose(-2, -1) / math.sqrt(self.config.head_dim)
            mask = torch.triu(torch.full((L, L), float("-inf"), device=attn.device), diagonal=1)
            attn = attn + mask
            attn = F.softmax(attn.float(), dim=-1).type_as(q)
            attn = attn @ v
            attn = attn.transpose(1, 2).reshape(B, L, -1)

        attn = self.o_proj(attn)
        return attn


class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def forward(self, hidden_state):
        res = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state) + res
        res = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state) + res
        return hidden_state


class LlamaForCausalLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_state = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = checkpoint(layer, hidden_state)
            else:
                hidden_state = layer(hidden_state)
        hidden_state = self.norm(hidden_state)
        logits = self.lm_head(hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits.reshape(-1, self.config.vocab_size)
            shift_labels = labels.reshape(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction='none').view(labels.size())
            loss = (loss * attention_mask).sum() / attention_mask.sum()
        return CausalLMOutputWithPast(loss, logits)

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        self.gradient_checkpointing = False

    @torch.inference_mode
    def generate(self, input_ids, eos_token_id, max_new_length=10, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1., ):
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
    def __init__(self, tokenizer, prefix="我是", generate_every=500, max_new_length=50):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.generate_every = generate_every
        self.max_new_length = max_new_length

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step % self.generate_every == 0:
            model.eval()
            input_ids = self.tokenizer(self.tokenizer.bos_token + self.prefix, return_tensors="pt")["input_ids"].to(
                model.device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_length=self.max_new_length,
                                         eos_token_id=self.tokenizer.eos_token_id, )
            decode = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Step {state.global_step}: Generated text: {decode}")
            model.train()


class ChatCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="请帮我写一个关于小狗的故事", generate_every=500, max_new_length=50):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.generate_every = generate_every
        self.max_new_length = max_new_length

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        messages = [{"role": 'user', "content": self.prompt}]
        new_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = torch.tensor(self.tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)
        if state.global_step % self.generate_every == 0:
            model.eval()
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, max_length=self.max_new_length,
                                         eos_token_id=self.tokenizer.eos_token_id, )
            decode = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Step {state.global_step}: Generated text: {decode}")
            model.train()

if __name__ == '__main__':
    # 测试标准注意力计算方式和 flash_attention 的结果是否一样
    device = torch.device("cuda")
    hidden_state = torch.randn(1, 100, 512).to(device)

    config = LMConfig()
    tokenizer = AutoTokenizer.from_pretrained(r"D:\minimind\model\minimind_tokenizer")
    config.vocab_size = len(tokenizer)
    model = LlamaAttention(config).to(device)

    attn_standard = model(hidden_state, flash_attn=False)

    with autocast():
        attn_flash = model(hidden_state, flash_attn=True)

    diff = torch.abs(attn_flash - attn_standard).mean()
    print("Mean Absolute Difference:", diff.item())
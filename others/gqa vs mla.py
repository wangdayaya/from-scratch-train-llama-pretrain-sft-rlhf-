import math
import time

import torch
from torch import nn

from llama.model.llama_config import LMConfig


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.rms_norm_eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * (var + self.rms_norm_eps).rsqrt()
        return self.weight * x


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


def get_causal_mask(attention_mask):
    B, L = attention_mask.shape
    min_value = -1e15
    causal_mask = torch.full((L, L), min_value).triu(diagonal=1)
    causal_mask = causal_mask.reshape(1, 1, L, L).repeat(B, 1, 1, 1)
    causal_mask = causal_mask.to(attention_mask.device)

    mask = attention_mask.reshape(B, 1, 1, L) == 0
    causal_mask = causal_mask.masked_fill(mask, min_value)
    return causal_mask


class MLA(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.num_attention_heads = config.num_attention_heads  # 总头数
        self.q_lora_rank = config.q_lora_rank  # q低秩压缩到的维度
        self.kv_lora_rank = config.kv_lora_rank  # kv低秩压缩到的维度
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # qk的总维度，不带旋转位置编码的维度加上带旋转位置编码的维度
        self.v_head_dim = config.v_head_dim  # value的维度，等于不带旋转位置编码的k维度
        self.max_seq_len = config.max_seq_len
        self.max_batch_size = config.max_batch_size

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank)  # q 的降维矩阵
        self.q_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.qk_head_dim)  # q的升维矩阵

        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)  # kv的降维矩阵
        self.kv_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim))   # kv的升维矩阵

        self.wo = nn.Linear(self.num_attention_heads * self.v_head_dim, self.hidden_size)

        self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim, config.max_seq_len)  # 旋转旋转位置编码
        self.register_buffer('kv_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_lora_rank),  persistent=False)
        self.register_buffer('pe_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim),  persistent=False)

    def forward(self, x, mask=None):
        bs, seq_len, _ = x.shape
        q = self.wq_a(x)  # [bs, seq_len, q_lora_rank]
        q = self.q_norm(q)  # [bs, seq_len, q_lora_rank]
        q = self.wq_b(q)  # [bs, seq_len, n_heads * qk_head_dim]
        q = q.view(bs, seq_len, self.num_attention_heads, self.qk_head_dim)  # [bs, seq_len, n_heads, qk_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # q_nope shape:[bs, seq_len, n_heads, qk_nope_head_dim] q_pe shape:[bs, seq_len, n_heads, qk_rope_head_dim]

        kv = self.wkv_a(x)  # [bs, seq_len, kv_lora_rank + qk_rope_head_dim]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim],   dim=-1)  # kv shape:[bs, seq_len, kv_lora_rank] k_pe shape:[bs, seq_len, qk_rope_head_dim]
        kv = self.kv_norm(kv)

        k_pe = k_pe.unsqueeze(2)
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)  # [bs, seq_len, n_heads, qk_rope_head_dim]  [bs, seq_len, 1, qk_rope_head_dim]
        k_pe = k_pe.squeeze(2)  # [bs, seq_len, qk_rope_head_dim]
        self.kv_cache[:bs, :seq_len, :].copy_(kv.detach())  # kv shape:[bs, seq_len, kv_lora_rank]
        self.pe_cache[:bs, :seq_len, :].copy_(k_pe.detach())  # k_pe shape:[bs, seq_len, qk_rope_head_dim]

        wkv_b = self.wkv_b.weight  # [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        wkv_b = wkv_b.view(self.num_attention_heads, -1,  self.kv_lora_rank)  # [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope,  wkv_b[:, :self.qk_nope_head_dim])  # [bs, seq_len, n_heads, qk_nope_head_dim] * [n_heads, qk_nope_head_dim, kv_lora_rank]=[bs, seq_len, n_heads, kv_lora_rank]
        scores_nope = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bs, :seq_len, :])  # [bs, seq_len, n_heads, kv_lora_rank] *[bs, seq_len, kv_lora_rank]= [bs, seq_len, n_heads, seq_len]
        scores_pe = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bs, :seq_len, :])  #   [bs, seq_len, n_heads, qk_rope_head_dim] *  [bs, seq_len, qk_rope_head_dim] =  [bs, seq_len, n_heads, seq_len]
        scores = (scores_nope + scores_pe) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)  # [bs, seq_len, n_heads, seq_len]
        if mask is not None:
            mask = get_causal_mask(mask).squeeze(1)  # [bs, 1, seq_len, seq_len] -》[bs, seq_len, seq_len]
            scores += mask.unsqueeze(2)

        scores = scores.softmax(dim=-1)
        x = torch.einsum("bsht,btc->bshc", scores,  self.kv_cache[:bs, :seq_len])  # [bs, seq_len, n_heads, seq_len]x[bs, seq_len, kv_lora_rank]=[bs, seq_len, n_heads, kv_lora_rank]
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])  # [bs, seq_len, n_heads, kv_lora_rank] x  [n_heads, v_head_dim, kv_lora_rank] = [bs, seq_len, n_heads, v_head_dim]
        x = x.contiguous().view(bs, seq_len, -1)  # [bs, seq_len, n_heads, v_head_dim] [bs, seq_len, n_heads*v_head_dim]
        x = self.wo(x)
        return x


class MHA(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_seq_len)  # 旋转旋转位置编码
        # self.max_seq_len = config.max_seq_len
        # self.max_batch_size = config.max_batch_size
        # self.kv_lora_rank = config.kv_lora_rank
        # self.qk_rope_head_dim = config.qk_rope_head_dim
        # self.register_buffer('kv_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_lora_rank),
        #                      persistent=False)
        # self.register_buffer('pe_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim),
        #                      persistent=False)

    def forward(self, hidden_state, mask):
        B, L, _ = hidden_state.shape
        q = self.q_proj(hidden_state).reshape(B, L, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_state).reshape(B, L, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_state).reshape(B, L, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)

        q, k = self.rotary_emb(q, k,)
        # cos, sin = llama_rotary_embedding(L)
        # cos, sin = cos.to(hidden_state.device), sin.to(hidden_state.device)
        # q = apply_rotary_pos_emb(q, cos, sin)
        # k = apply_rotary_pos_emb(k, cos, sin)
        # k = k.unsqueeze(2).repeat(1, 1, 4, 1, 1).reshape(B, -1, L, 32)
        # v = v.unsqueeze(2).repeat(1, 1, 4, 1, 1).reshape(B, -1, L, 32)

        attn = q.matmul(k.transpose(2, 3)) / math.sqrt(config.hidden_size//config.num_attention_heads)
        mask = get_causal_mask(mask)
        attn = (attn + mask).softmax(-1)
        attn = attn.matmul(v)

        attn = attn.transpose(1, 2).reshape(B, L, -1)
        attn = self.o_proj(attn)
        return attn

# 测试代码
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(" ---------MLA--------")
    config = LMConfig(v_head_dim=16, hidden_size=8192, head_dim=256)
    print(config)
    mla = MLA(config)
    for name, p in mla.named_parameters():
        print(name, p.numel())
    num_params = sum(p.numel() for p in mla.parameters())
    print("mla 模型参数量:", num_params)  # 10955696
    # 创建测试输入
    test_input = torch.randn(config.max_batch_size, config.max_seq_len, config.hidden_size)
    test_mask = torch.ones((config.max_batch_size, config.max_seq_len), dtype=torch.int32)
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
    mla = mla.to(device)
    print(f"加载 mla 模型显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")
    test_input = test_input.to(device)
    test_mask = test_mask.to(device)

    start = time.time()
    outputs = mla(test_input, test_mask)
    print("耗时：", time.time() - start)
    print(f"加载 mla 模型和数据进行计算显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")

    print("---------MHA--------")
    # config = LMConfig(hidden_size=8192, head_dim=256)
    # mha = MHA(config)
    # for name, p in mha.named_parameters():
    #     print(name, p.numel())
    # num_params = sum(p.numel() for p in mha.parameters())
    # print("mha 模型参数量:", num_params)  # 10955696
    # # 创建测试输入
    # test_input = torch.randn(config.max_batch_size, config.max_seq_len, config.hidden_size)
    # test_mask = torch.ones((config.max_batch_size, config.max_seq_len), dtype=torch.int32)
    # initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
    # mha = mha.to(device)
    # print(f"加载 mha 模型显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")
    # test_input = test_input.to(device)
    # test_mask = test_mask.to(device)
    #
    # start = time.time()
    # outputs = mha(test_input, test_mask)
    # print("耗时：", time.time() - start)
    # print(f"加载 mha 模型和数据进行计算显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")


# mla 模型参数量: 12217264
# 加载 mla 模型显存占用: 0.0502 GB
# 耗时： 0.22209906578063965
# 加载 mla 模型和数据进行计算显存占用: 2.4410 GB



# mla 模型参数量: 71403440
# 加载 mla 模型显存占用: 0.2706 GB
# 耗时： 0.2320079803466797
# 加载 mla 模型和数据进行计算显存占用: 2.9349 GB


# mha 模型参数量: 268435456
# 加载 mha 模型显存占用: 1.0062 GB
# 耗时： 0.29510498046875
# 加载 mha 模型和数据进行计算显存占用: 4.1392 GB
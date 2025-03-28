import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        """
        输入:
            x: [seq_true, hidden_size] 需要处理的输入
        输出:
            down_proj: [seq_true, hidden_size] 专家网络的输出
        """
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.top_k = config.top_k
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num, bias=config.mlp_bias)

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, hidden_size]
        输出:
            sparse_logits: [batch_size, seq_len, expert_num] 稀疏化的路由权重
            indices: [batch_size, seq_len, top_k] 选择的专家索引
            gate_logit: [batch_size * seq_len, expert_num] 原始门控网络输出
        """
        logits = self.gate(x)  # [batch_size, seq_len, expert_num]
        logits_topk, indices = logits.topk(self.top_k, dim=-1)  # [batch_size, seq_len, top_k]
        zeros = torch.full_like(logits, float("-inf"))  # [batch_size, seq_len, expert_num] 创建一个全为负无穷的矩阵，用于屏蔽其他专家的概率
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # [batch_size, seq_len, expert_num]
        sparse_probs = F.softmax(sparse_logits, dim=-1)  # [batch_size, seq_len, expert_num] 稀疏化的路由权重
        gate_logit = logits.view(-1, self.expert_num)  # [batch_size * seq_len, expert_num]  展平门控网络输出
        return sparse_probs, indices, gate_logit

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)

    def forward(self, x):
        """
        输入:
            x: [batch_size, seq_len, hidden_size]
        输出:
            final_outputs: [batch_size, seq_len, hidden_size] 最终输出
            gate_logit: [batch_size * seq_len, expert_num] 原始门控网络输出
        """
        sparse_probs, indices, gate_logit = self.gating(x)  # 门控网络输出
        final_outputs = torch.zeros_like(x)  # [batch_size, seq_len, hidden_size] 初始化最终输出
        x_flat = x.view(-1, x.shape[-1])  # [batch_size * seq_len, hidden_size] 展平输入
        sparse_logits_flat = sparse_probs.view(-1, sparse_probs.shape[-1])  # [batch_size * seq_len, expert_num] 展平稀疏化路由权重
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)  # [batch_size, seq_len]  创建专家掩码，标记哪些位置的token需要被当前专家处理
            expert_mask_flat = expert_mask.view(-1)  # [batch_size * seq_len]展平掩码
            if expert_mask_flat.any():  # 如果有需要处理的输入
                expert_input = x_flat[expert_mask_flat]  # [seq_true, hidden_size] 提取需要处理的 token
                expert_output = expert(expert_input)  # [seq_true, hidden_size] 专家网络的输出
                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # [seq_true, 1]   # 提取当前专家的路由权重
                weighted_output = expert_output * gate_scores  # 加权输出: [seq_true, hidden_size]
                final_outputs[expert_mask] += weighted_output # 累加到最终输出
        return final_outputs, gate_logit

def load_balancing_loss_func(gate_logits, num_experts, top_k):
    concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0)  # [l*b*s， e] 各个层的 gate_logit 进行合并 [layers X batch_size X sequence_length, num_experts]
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1) # [l*b*s， e]
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [l*b*s， topk]
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)  # [l*b*s， topk， e]
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)    # [topk， e] 计算每个专家在所有 token 上被选中的频率
    router_prob_per_expert = torch.mean(routing_weights, dim=0)   # [e]   计算每个专家的路由概率
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class MultiMLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlps = nn.ModuleList([MLP(config) for _ in range(config.expert_num)])  # 将6个MLP封装到ModuleList中

    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)  # 依次通过每个MLP
        return x



# 测试代码
if __name__ == "__main__":
    # --------Config----------
    class DummyConfig:
        hidden_size = 1024
        expert_num = 6
        intermediate_size = 4096
        top_k = 2
        mlp_bias = False


    config = DummyConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # --------MoE----------
    # 初始化MoE模型
    moe_layer = MoE(config)
    num_params = sum(p.numel() for p in moe_layer.parameters())
    print("MoE模型参数量:", num_params)
    # 创建测试输入
    test_input = torch.randn(8, 100, config.hidden_size)
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
    moe_layer = moe_layer.to(device)
    print( f"加载 MoE 模型显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")
    test_input = test_input.to(device)
    # 前向传播测试
    start = time.time()
    outputs, gate_logit = moe_layer(test_input)
    print("耗时：", time.time() - start)
    print(f"加载 MoE 模型和数据进行计算显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")
    aux = load_balancing_loss_func([gate_logit], config.expert_num, config.top_k)
    print(f"辅助损失：{aux}")



    # ---------MLP----------
    test_input = torch.randn(8, 100, config.hidden_size)
    mlps = MultiMLPModel(config)
    num_params = sum(p.numel() for p in mlps.parameters())
    initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
    mlps = mlps.to(device)
    print(f"加载 MLP 模型显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")
    print("MLP 模型参数量:", num_params)
    test_input = test_input.to(device)

    start = time.time()
    output = mlps(test_input)
    print("耗时：", time.time() - start)
    print(f"加载 MLP 模型和数据进行计算显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 3) - initial_memory:.4f} GB")





from transformers import PretrainedConfig


class LMConfig(PretrainedConfig):
    model_type = "my_llama"

    def __init__(
            self,

            flash_attention=True,
            vocab_size=6400,
            max_seq_len=512,
            hidden_size=512,
            num_hidden_layers=32,
            num_attention_heads=16,
            head_dim=32,
            num_key_value_heads=4,
            rope_theta=1000000.0,
            rms_norm_eps=1e-06,
            # moe
            max_batch_size=10,
            dropout=0.0,
            expert_num=4,
            topk=2,
            output_router_logits=False,
            aux_loss_coef=0.01,
            # mla
            q_lora_rank=128,
            kv_lora_rank=128,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            **kwargs
    ):
        self.flas_attention = flash_attention
        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = hidden_size * 4
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        super().__init__(**kwargs)
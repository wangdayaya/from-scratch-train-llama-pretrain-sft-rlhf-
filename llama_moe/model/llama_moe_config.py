from transformers import PretrainedConfig


class LMConfig(PretrainedConfig):
    model_type = "my_llama"

    def __init__(
            self,
            flash_attention=False,
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
            expert_num=4,
            topk=2,
            output_router_logits=True,
            aux_loss_coef=0.01,
            shared_expert=True,

            # mla
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            **kwargs
    ):
        self.flash_attention = flash_attention
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        self.shared_expert = shared_expert

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

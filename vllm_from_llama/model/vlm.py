from torch.utils.checkpoint import checkpoint
import torch
from PIL import Image
from torch import nn
from transformers import CLIPProcessor, CLIPModel, TrainerCallback, TrainerState, TrainerControl
from transformers.modeling_outputs import CausalLMOutputWithPast

from vllm_from_llama.model.vlm_config import VLMConfig
from vllm_from_llama.model.llama import LlamaForCausalLM


class VisionProj(nn.Module):
    def __init__(self, ve_dim=768, lm_dim=512):
        super().__init__()
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_dim, self.lm_dim)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


class VLM(LlamaForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None):
        super().__init__(params)
        if not params:
            params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_model()
        self.vision_proj = VisionProj(lm_dim=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path="clip-vit-base-patch16"):
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):  # image_tensors:[bs,3,224,224]
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()  # 可能第一个位置是特殊 token
        return img_embedding  # [bs, 196, 768]

    def count_vision_proj(self, tokens, h, vision_tensors=None,
                          seqlen=512):  # [bs, ml-1], [bs,ml-1, 512]  [bs,num,196,768]
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)  # [bs, 444, 196]
            matches = (tokens_view == image_ids_tensor).all(dim=2)  # [bs, 444,]
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        # tokens:[16,639], self.params.image_ids:[196]  image_indices: bs 个字典，每个value 中存放图像的开始、结束位置的列表
        image_indices = find_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)  # [bs,n,196,512]
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)  # 【bs, 639, 512】
        return h

    def forward(self, input_ids=None, attention_mask=None, labels=None, pixel_tensors=None, **args):
        hidden_state = self.embed_tokens(input_ids)  # [bs, max_length-1, dim]

        if pixel_tensors is not None:
            if len(pixel_tensors.shape) == 6:
                pixel_tensors = pixel_tensors.squeeze(2)  # [bs, 1, 3, 224,224]
            bs, num, c, im_h, im_w = pixel_tensors.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack(
                [VLM.get_image_embeddings(pixel_tensors[:, i, :, :, :], self.vision_encoder) for i in range(num)],
                dim=stack_dim)
            hidden_state = self.count_vision_proj(tokens=input_ids, h=hidden_state, vision_tensors=vision_tensors,
                                                  seqlen=input_ids.shape[
                                                      1])  # input_ids：【bs, max_length-1】, vision_tensors:[bs,num,196,768] , [bs,max_length-1,512]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_state = checkpoint(layer, hidden_state, attention_mask)
            else:
                hidden_state = layer(hidden_state, attention_mask)

        logits = self.lm_head(self.norm(hidden_state))

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


class VLMCallback(TrainerCallback):
    def __init__(self, tokenizer, preprocess, config, generate_every=500, max_new_length=50):
        self.tokenizer = tokenizer
        self.generate_every = generate_every
        self.max_new_length = max_new_length
        self.preprocess = preprocess
        self.config = config

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step % self.generate_every == 0:
            image_dir = r'D:\PycharmProjects\from scratch train llama\vllm_from_llama\eval_image\彩虹瀑布-Rainbow-Falls .jpg'
            prompt = f"{self.config.image_special_token}\n描述一下这个图像的内容。"

            image = Image.open(image_dir).convert('RGB')
            pixel_tensors = model.image2tensor(image, self.preprocess).unsqueeze(0).to(model.device)
            messages = [{"role": "user", "content": prompt}]

            new_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)[-self.config.max_seq_len + 1:]

            print(f'[Image]: {image_dir}')
            with torch.no_grad():
                e = self.tokenizer(new_prompt)
                input_ids = torch.tensor(e['input_ids'], device=model.device).unsqueeze(0)
                attention_mask = torch.tensor(e['attention_mask'], device=model.device).unsqueeze(0)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_length=self.max_new_length,
                    pixel_tensors=pixel_tensors
                )
                print('🤖️: ', end='')
                print(self.tokenizer.decode(outputs.squeeze()[input_ids.shape[1]:].tolist(), skip_special_tokens=True),
                      end='')
                print('\n')

        # messages = [{"role": 'user', "content": self.prompt}]
        # new_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # input_ids = torch.tensor(self.tokenizer(new_prompt)['input_ids'], device=model.device).unsqueeze(0)
        # if state.global_step % self.generate_every == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         outputs = model.generate(input_ids=input_ids, max_length=self.max_new_length,
        #                                  eos_token_id=self.tokenizer.eos_token_id, )
        #     decode = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #     print(f"Step {state.global_step}: Generated text: {decode}")
        #     model.train()

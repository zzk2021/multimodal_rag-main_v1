import json
from typing import Optional, Any

import torch
from PIL import Image
from llama_index.core.base.llms.types import LLMMetadata, CompletionResponse, ChatResponse, CompletionResponseGen
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import TextStreamer, AutoModel, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

from LLM.mipha.constants import IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
from LLM.mipha.conversation import conv_templates
from LLM.mipha.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria
from LLM.mipha.model.builder import load_pretrained_model
from LLM.mipha.serve.cli import load_image

with open('config/config.json') as user_file:
    config = user_file.read()
config = json.loads(config)
# {
#   "name": "John",
#   "age": 50,
#   "is_married": false,
#   "profession": null,
#   "hobbies": ["travelling", "photography"]
# }
DEFAULT_LLM_MODEL = config["model_LLM_path"]

class MobileVLM():
    def __init__(self, model_path=DEFAULT_LLM_MODEL):
        super().__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto',
        )

        from scripts.inference import inference_once
        # model_path = "mtgv/MobileVLM-1.7B" # MobileVLM
        model_path = "mtgv/MobileVLM_V2-1.7B"  # MobileVLM V2
        image_file = "assets/samples/demo.jpg"
        prompt_str = "Who is the author of this book?\nAnswer the question using a single word or phrase."
        # (or) What is the title of this book?
        # (or) Is this book related to Education & Teaching?

        args = type('Args', (), {
            "model_path": model_path,
            "image_file": image_file,
            "prompt": prompt_str,
            "conv_mode": "v1",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512,
            "load_8bit": False,
            "load_4bit": False,
        })()

        inference_once(args)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, image: str, **kwargs: Any) -> CompletionResponse:
        pass

    #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        prompt, image = prompt[0], prompt[1]
        msgs = [{"content": prompt, "role": "user"}]
        if image is not None:
            image = Image.open(image).convert("RGB")
        answer = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return answer

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, image: Optional[torch.FloatTensor] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        pass
class MiniCPM(CustomLLM):
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 128  # 输出的token数量
    model_name: str = "Mipha"  # 模型名称
    tokenizer: object = None  # 分词器
    model: object = None  # 模型
    image_processor: object = None # image_processor
    def __init__(self, pretrained_model_name_or_path=DEFAULT_LLM_MODEL):
        super().__init__()
        # GPU方式加载模型
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,device_map="cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, image: str, **kwargs: Any) -> CompletionResponse:
        pass

  #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        prompt, image = prompt[0], prompt[1]
        msgs = [{"content":prompt,"role":"user"}]
        if image is not None:
            image = Image.open(image).convert("RGB")
        answer = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return answer

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str,image: Optional[torch.FloatTensor] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        pass

class OurLLM(CustomLLM):
    context_window: int = 8192  # 上下文窗口大小
    num_output: int = 128  # 输出的token数量
    model_name: str = "Mipha"  # 模型名称
    tokenizer: object = None  # 分词器
    model: object = None  # 模型
    image_processor: object = None # image_processor
    def __init__(self, pretrained_model_name_or_path=DEFAULT_LLM_MODEL):
        super().__init__()
        # GPU方式加载模型
        tokenizer, model, image_processor, context_len = load_pretrained_model(pretrained_model_name_or_path, model_base=None, model_name="Mipha-phi2")
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, image: str, **kwargs: Any) -> CompletionResponse:
        pass

  #  @llm_chat_callback()  # 回调函数
    def chat(self, prompt, **kwargs: Any) -> ChatResponse:
        # 完成函数
        prompt, image = prompt[0], prompt[1]
        conv = conv_templates["phi"].copy()
        roles = conv.roles
        inp = f"{roles[0]}: {prompt}"

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        if image is not None:
            # first message
            image = load_image(image)
            # image = image.resize((224, 224))
            image_tensor = process_images([image], self.image_processor, self.model.config)
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
            image_tensor = None

        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.eos_token_id,  # Pad token
                stopping_criteria=[stopping_criteria]
            )
        outputs = self.tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()
        return outputs

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str,image: Optional[torch.FloatTensor] = None, **kwargs: Any
    ) -> CompletionResponseGen:
        pass

import pdb
import numpy as np
import os
import pdb
import torch
import time
import habitat_sim
import cv2
from collections import Counter
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer
import sentencepiece as spm

from PIL import Image
from src.utils import *
from src.vlmAgent import VLMAgent

class LLaVaAgent(VLMAgent):

    def __init__(self, *pargs, **kwargs):
        self.setup_kwargs = kwargs
        self.setup_pargs = pargs
        self.is_setup = False
        self.name = f"llava-v1.6-{kwargs['llm']}-{kwargs['size']}-hf"
        if kwargs['size'] == '34b':
            self.name = "llava-v1.6-34b-hf"


    def setup(self, llm, size, quantize=True, torch_dtype=torch.float16, low_cpu_mem_usage=True):

        assert llm in ['vicuna', 'mistral', 'hermes']
        assert size in ['7b', '13b', '34b']
        model_kwargs = {}
        self.llm = llm
        
        model_kwargs['pretrained_model_name_or_path'] = f'llava-hf/{self.name}'
        processor_path = f'llava-hf/{self.name}'

        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs['quantization_config'] = quantization_config
        model_kwargs['torch_dtype'] = torch_dtype
        model_kwargs['low_cpu_mem_usage'] = low_cpu_mem_usage

        self.processor = LlavaNextProcessor.from_pretrained(processor_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(**model_kwargs).to('cuda:0')
    
    
    def call(self, visual_prompt: np.array, text_prompt: str, num_samples=0):
        
        if not self.is_setup:
            self.setup(*self.setup_pargs, **self.setup_kwargs)
            self.is_setup=True

        if self.llm == 'vicuna':
            text_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{text_prompt} ASSISTANT:"
        elif self.llm == 'mistral':
            text_prompt = f"[INST] <image>\n{text_prompt} [/INST]"
        else:
            text_prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{text_prompt}<|im_end|><|im_start|>assistant\n"
        if visual_prompt.shape[-1] == 4:
            visual_prompt = visual_prompt[:, :, 0:3]
        image = Image.fromarray(visual_prompt, mode='RGB') 

        inputs = self.processor(text_prompt, image, return_tensors="pt").to(self.model.device)
        t = time.time()
        input_tokens = inputs['input_ids'].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=600)
        duration = time.time() - t
        tokens_generated = output.shape[1]-input_tokens
        print(f'{self.name} finished inference, took {duration} seconds, speed of {tokens_generated/duration} t/s')

        output_text = self.processor.decode(output[0][input_tokens:], skip_special_tokens=True)
        
        return output_text, {'tokens_generated': tokens_generated, 'duration': duration, 'input_tokens': input_tokens}

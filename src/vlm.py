import base64
from collections import Counter
from math import e
import random
import socket
from sqlite3 import DatabaseError

from cv2 import log
import httplib2
import numpy as np
from sympy import im
from src.utils import *
import pdb
import numpy as np
import pdb
import torch
import time
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer, AutoProcessor, VipLlavaForConditionalGeneration, LlavaForConditionalGeneration
# import sentencepiece as spm
# import torch.nn as nn
from PIL import Image
from src.utils import *
import io
import os
import google.generativeai as genai
from googleapiclient.errors import HttpError
from openai import OpenAI
import requests

class VLM:
    """
    Trivial agent for testing
    """
    def __init__(self, **kwargs):
        self.name = "not implemented"

    def call(self, visual_prompt: np.array, text_prompt: str):
        raise NotImplementedError
    
    def call_chat(self, history, visual_prompt, text_prompt, add_timesteps_prompt=True, step=None):
        return ""

    def reset(self):
        raise NotImplementedError

    def rewind(self):
        raise NotImplementedError


class LlavaModel(VLM):

    def __init__(self, **kwargs):
        
        self.setup_kwargs = kwargs
        self.folder = 'llava-hf'
        self.model_cls = LlavaNextForConditionalGeneration
        self.processor_cls = LlavaNextProcessor
        self.is_setup = False
        name = kwargs['name']
        if name == '34b':
            self.name = "llava-v1.6-34b-hf"
        elif name == '13b':
            self.name = "llava-v1.6-vicuna-13b-hf"
        elif name == 'mistral7b':
            self.name="llava-v1.6-mistral-7b-hf"
        elif name == 'vicuna7b':
            self.name="llava-v1.6-vicuna-7b-hf"
        elif name == 'qwen7b':
            self.name='llava-interleave-qwen-7b-hf'
            self.processor_cls = AutoProcessor
            self.model_cls = LlavaForConditionalGeneration
        elif name == 'dpo':
            self.name='llava-interleave-qwen-7b-dpo-hf'
            self.processor_cls = AutoProcessor
            self.model_cls = LlavaForConditionalGeneration         
        elif name == '8b':
            self.name='llama3-llava-next-8b-hf'
        elif name == '72b':
            self.name='llava-next-72b-hf'
        elif name == 'vip7b':
            self.name = 'vip-llava-7b-hf'
            self.processor_cls = AutoProcessor
            self.model_cls = VipLlavaForConditionalGeneration

        elif name == 'vip13b':
            self.folder = 'llava-hf'
            self.name = 'vip-llava-13b-hf'
            self.processor_cls = AutoProcessor
            self.model_cls = VipLlavaForConditionalGeneration
        else:
            raise(f"Name {name} is not a valid llava name")
    def setup(self, name = None, quantize=False, torch_dtype=torch.float16,
               low_cpu_mem_usage=True, device_map=None, use_flash_attention_2=False,
               do_sample=False, num_beams=1
               ):

        model_kwargs = {}
        model_kwargs['do_sample'] = do_sample
        model_kwargs['num_beams'] = num_beams
        model_kwargs['pretrained_model_name_or_path'] = f'{self.folder}/{self.name}'

        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs['quantization_config'] = quantization_config
        if device_map:
            model_kwargs['device_map'] = 'auto'
        model_kwargs['torch_dtype'] = torch_dtype
        model_kwargs['low_cpu_mem_usage'] = low_cpu_mem_usage
        if use_flash_attention_2:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
        # pdb.set_trace()
        self.model = self.model_cls.from_pretrained(**model_kwargs)
        self.processor = self.processor_cls.from_pretrained(f'{self.folder}/{self.name}')
        if not device_map:
            self.model = self.model.to('cuda')
    
    def call_multi_image(self, conversation, images):
        if not self.is_setup:
            self.setup(**self.setup_kwargs)
            self.is_setup=True
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        ims = []
        for image in images:
            ims.append(Image.fromarray(image[:, :, 0:3]))
        print(prompt)
        inputs = self.processor(prompt, images=ims, return_tensors="pt").to(self.model.device, self.model.dtype)
        t = time.time()
        input_tokens = inputs['input_ids'].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=600)
        duration = time.time() - t
        tokens_generated = output.shape[1]-input_tokens
        print(f'{self.name} finished inference, took {duration} seconds, speed of {tokens_generated/duration} t/s')

        output_text = self.processor.decode(output[0][input_tokens:], skip_special_tokens=True)
        
        return output_text, {'tokens_generated': tokens_generated, 'duration': duration, 'input_tokens': input_tokens}


    def call(self, visual_prompt: np.array, text_prompt: str, num_samples=0):
        
        if not self.is_setup:
            self.setup(**self.setup_kwargs)
            self.is_setup=True

        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)


        if visual_prompt.shape[-1] == 4:
            visual_prompt = visual_prompt[:, :, 0:3]
        image = Image.fromarray(visual_prompt, mode='RGB') 

        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device, self.model.dtype)
        t = time.time()
        input_tokens = inputs['input_ids'].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=600)
        duration = time.time() - t
        tokens_generated = output.shape[1]-input_tokens
        print(f'{self.name} finished inference, took {duration} seconds, speed of {tokens_generated/duration} t/s')

        output_text = self.processor.decode(output[0][input_tokens:], skip_special_tokens=True)
        
        return output_text, {'tokens_generated': tokens_generated, 'duration': duration, 'input_tokens': input_tokens}

class GeminiModel(VLM):
    
    def __init__(self, model="gemini-1.5-flash", sys_instruction=None):
        self.name = model
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        # Create the model
        self.generation_config = {
        "temperature": 1,
        "top_p": 0.8,
        "top_k": 64,
        "max_output_tokens": 500,
        "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
        model_name = model,
        generation_config=self.generation_config,
        system_instruction=sys_instruction
        )
        self.session = self.model.start_chat(history=[])

    def call_chat(self, history, images, text_prompt, add_timesteps_prompt=True, step=None):
        
  
        try:
            uploaded = []
            for image in images:
                im = Image.fromarray(image[:, :, 0:3], mode='RGB')
                # im.save('logs/temp.png', overwrite=True)
                retries = 3  # Number of retry attempts
                for attempt in range(retries):
                    try:
                        # Attempt to upload the file
                        uploaded.append(im)
                        #uploaded.append(genai.upload_file('logs/temp.png'))
                        break  # Break the retry loop if successful
                    except (socket.timeout, httplib2.ServerNotFoundError, HttpError, BrokenPipeError, OSError) as e:
                        print(f"Error uploading file (Attempt {attempt + 1}/{retries}): {e}")
                        if attempt < retries - 1:
                            time.sleep(2 ** attempt) 
                        else:
                            raise
            rng_state = random.getstate()
            t = time.time()
            response = self.session.send_message([text_prompt] + uploaded)
            finish = time.time() - t
            rng_state = random.setstate(rng_state)

            if history == 0:
                self.session = self.model.start_chat(history=[])
            else:
                if len(self.session.history) > 2*history:
                    self.session.history = self.session.history[-2*history:]

                if add_timesteps_prompt:
                    self.session.history[-2].parts[0].text = f"[PREVIOUS OBSERVATION] Timestep {step}:"
                else:
                    self.session.history[-2].parts = self.session.history[-2].parts[1:]

            resp = response.text
            perf = {'tokens_generated': response.usage_metadata.candidates_token_count, 'duration': finish, 'input_tokens': response.usage_metadata.prompt_token_count}
            print(f'\n{self.name} finished inference, took {np.round(finish, 2)} seconds, speed of {np.round(perf["tokens_generated"]/finish, 2)} t/s')
        
        except Exception as e:  
            resp = f"ERROR: {e}"
            print(resp)
            perf = {'tokens_generated': 0, 'duration': 1, 'input_tokens': 0}


        return resp, perf
    
    def rewind(self):
        if len(self.session.history) > 1:
            self.model.rewind()

    def reset(self):

        del self.session
        self.session = self.model.start_chat(history=[])

    def call(self, images, text_prompt: str, logprobs=None):
        ims = [Image.fromarray(image[:, :, 0:3], mode='RGB') for image in images]

        try:
            t = time.time()
            response = self.model.generate_content([text_prompt] + ims)
            finish = time.time() - t
            perf = {'tokens_generated': response.usage_metadata.candidates_token_count, 'duration': finish, 'input_tokens': response.usage_metadata.prompt_token_count}
            print(f'{self.name} finished inference, took {finish} seconds, speed of {perf["tokens_generated"]/finish} t/s')
            resp = response.text
        except Exception as e:  
            resp = f"GEMINI API ERROR: {e}"
            print(resp)
            perf = {'tokens_generated': 0, 'duration': 1, 'input_tokens': 0}
        return resp, perf
    
import clip


# Load the CLIP model and the preprocessing method



class CLIPModel(VLM):

    def __init__(self, model="ViT-B/32"):
        self.name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model, device=self.device)
    
    def call(self, images, text_prompt: str, logprobs=None):
        images = [self.preprocess(Image.fromarray((img_array * 255).astype(np.uint8))) for img_array in images]
        stacked = torch.stack(images).to(self.device)

        text = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(stacked)
            text_features = self.model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze()
        print("Image features:", image_features)
        print("Text features:", text_features)
        print("Similarities:", similarities)

        return [s for s in similarities.cpu().numpy()]



class GPTModel(VLM) :

    def __init__(self, model="gpt-4o-mini", api_key=None, sys_instruction="You are a helpful assistant"):
        self.name = model
        self.client = OpenAI()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Set up the configuration
        self.spend = 0
        if self.name == 'gpt-4o-mini':
            self.inp_cost = 0.15/1000000
            self.out_cost = 0.6/1000000
        else:
            self.inp_cost = 5/1000000
            self.out_cost = 15/1000000
        self.session_history = []
        self.base_message = [{'role': 'system', 'content': sys_instruction}]

    def call_chat(self, history, images, text_prompt, add_timesteps_prompt=True, step=None, logprobs=0):
        def encode_image_from_rgb(rgb_array):
            # Convert the RGB array to a format that can be encoded, such as PNG
            _, buffer = cv2.imencode('.png', cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode('utf-8')
        def encode_image(rgb_array):
            im = Image.fromarray(rgb_array[:, :, 0:3], mode='RGB')
            im.save('logs/temp.png', overwrite=True)
            with open('logs/temp.png', "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        message = {
            "role": "user",
            "content": [{'type': 'text', 'text': text_prompt}]
        }

        for image in images:
            encoded = encode_image(image)
            message['content'].append(        {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}",
                "detail": 'low'
                }}
            )

        try:
            rng_state = random.getstate()
            t = time.time()
            payload = {
                "model": self.name,
                "messages": self.base_message + [message],
                "max_tokens": 500,
                "temperature": 0.2,
            }
            if logprobs > 0:
                payload['logprobs'] = True
                payload['top_logprobs'] = logprobs

            headers = {
                'Content-type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            # r = self.client.chat.completions.create(model=self.name, messages=self.base_message + [message], max_tokens=500)
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
            # pdb.set_trace()
            self.spend += r['usage']['prompt_tokens']*self.inp_cost + r['usage']['completion_tokens']*self.out_cost
            self.get_spend()
            print(r['usage'])
            # pdb.set_trace()
            response = r['choices'][0]['message']
            resp = response['content']
            finish = time.time() - t
            random.setstate(rng_state)
            self.session_history.append(message)
            self.session_history.append(response)
            if history == 0:
                self.session_history = []
            else:
                if len(self.session_history) > 2*history:
                    self.session_history = self.session_history[-2*history:]
            print(f"{self.name} finished inference, took {finish} seconds, speed of {r['usage']['completion_tokens']/finish} t/s")
            if logprobs > 0:
                return resp, {"tokens_generated": r['usage']['completion_tokens'], "duration": finish, "input_tokens": r['usage']['prompt_tokens'], "logprobs": r['choices'][0]['logprobs']['content'][0]['top_logprobs']}
            return resp, {"tokens_generated": r['usage']['completion_tokens'], "duration": finish, "input_tokens": r['usage']['prompt_tokens']}
                    
        except Exception as e:
            print("OPENAI API ERROR")
            print(e)
            return "ERROR", {"tokens_generated": 0, "duration": 1, "input_tokens": 0}
    
    def call(self, images, text_prompt: str, logprobs):
        return self.call_chat(0, images, text_prompt, logprobs=logprobs)

    def rewind(self):
        if len(self.session_history) > 1:
            self.session_history = self.session_history[:-2]
#   

    def get_spend(self):
        print(f"Total spend for {self.name}: {np.round(self.spend, 2)}\n")
        return self.spend

    def reset(self):
        self.session_history = []


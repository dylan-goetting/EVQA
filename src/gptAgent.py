import openai
import numpy as np
from PIL import Image
import io
from src.utils import *
from src.vlmAgent import VLMAgent
import os

class GPTAgent(VLMAgent):
    
    def __init__(self, model):
        self.name = model
        self.openai = openai
        self.openai.api_key = os.environ['openai_key']

    def call(self, visual_prompt: np.array, text_prompt: str, num_samples):

        image = Image.fromarray(visual_prompt)

        image_io = io.BytesIO()
        image.save(image_io, format='PNG')
        image_io.seek(0)
        image_data = image_io.read()

        response = self.openai.Completion.create(
            model=self.name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text_prompt, "images": [{"data": image_data}]}
            ]
        )

        return response['choices'][0]['message']['content']
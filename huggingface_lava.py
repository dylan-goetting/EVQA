# Use a pipeline as a high-level helper
import pdb
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import time

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                          quantization_config=quantization_config,
                                                          device_map="auto"
                                                          )

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# prepare image and text prompt, using the appropriate prompt template
image = Image.open("../Pictures/exp1.png")
pt = """
What number arrow should the agent take to solve the maze?
Rules:
You are looking at a maze that a reinforcement learning agent is trying to solve. The agent is represented by a blue X inscribed in a circle and is partially through the maze
The start is labeled at the top and the end is labeled at the bottom
The numbered arrows in red represent the possible actions the agent can take, one in each of the four cardinal directions
Note that these labels are superimposed on top of the maze and not a part of the actual maze
The agent will move one step immediately in the direction of the arrow. If there is a wall in between the label and the agent, the agent WILL NOT MOVE. It is important that you consider this

Instructions:
1. Reason through the task
2. Explain each action choice in the context of the maze, its direction and whether or not the action is possible due to walls
3. Return the number of the best action choice.
"""
prompt = f"[INST] <image>\n{pt} [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to(model.device)

print('starting output')
t = time.time()
output = model.generate(**inputs, max_new_tokens=1000)
print(f'finished output, took {time.time() - t} seconds')
print(processor.decode(output[0], skip_special_tokens=True))

pdb.set_trace()

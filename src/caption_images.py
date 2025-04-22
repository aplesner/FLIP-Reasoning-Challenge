import json
from pathlib import Path
import os
import argparse

import traceback
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

import gc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import logging

# Suppress warnings at the logging level
logging.getLogger("transformers").setLevel(logging.ERROR)


BATCH_SIZE_default = 1
MAX_WORKERS_NUM=4

HUGGING_FACE_TOKEN =None
GOOGLE_API_KEY=None
OPENAI_API_KEY=None
TEST_MODE=0

ref_storage = {}
# !!! can be memory intense !!!
image_captions = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_exp_params(models_list = ['ViPLlava'], params_file = None):
    global ref_storage

    def get_model_load_params(model_params):
        model_load_params = model_params
        if "torch_dtype" in model_load_params:
            if "torch.bfloat16"==model_load_params["torch_dtype"]:
                model_load_params["torch_dtype"] = torch.bfloat16
            elif "torch.float16"==model_load_params["torch_dtype"]:
                model_load_params["torch_dtype"] = torch.float16
            elif "torch.float32"==model_load_params["torch_dtype"]:
                model_load_params["torch_dtype"] = torch.float32
        if "BitsAndBytesConfig" in model_load_params:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(**model_load_params['BitsAndBytesConfig'])
            del model_load_params["BitsAndBytesConfig"]
            model_load_params["quantization_config"]=quantization_config
        return model_load_params
    
    if params_file is not None:
        params=read_json(params_file)

    else:
        raise Exception("task with no parameters is prevented")
        
    for model_name in models_list:
        # ViPLlava_7B = 13770MiB at bfloat16 precision on 1 GPU
        if "ViPLlava" in model_name:
            from transformers import VipLlavaForConditionalGeneration, AutoProcessor

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {}
            
            ref_storage[model_name] = {
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "model":VipLlavaForConditionalGeneration.from_pretrained(
                    params[model_name]["model_id"], 
                    **model_load_params
                    # device_map="auto", 
                    # torch_dtype=torch.bfloat16
                    ),
                "processor":AutoProcessor.from_pretrained(params[model_name]["model_id"]),
                "generation_kwargs": params[model_name]["gen_kw"],

                "method":caption_VipLlava_batched
                }

            try:
                # ref_storage[model_name]["model"].generation_config.pad_token_id = ref_storage[model_name]['model'].generation_config.eos_token_id
                ref_storage[model_name]["processor"].patch_size = ref_storage[model_name]["model"].config.vision_config.patch_size
                ref_storage[model_name]["processor"].vision_feature_select_strategy = ref_storage[model_name]["model"].config.vision_feature_select_strategy
            except:
                print(f"one of the settings [(pad_token_id to eos_token_id), patch_size, vision_feature_select_strategy] failed for {model_name}")
            
            print(f"{model_name} LOADED IN")
            print(os.system("nvidia-smi"))

        # LlavaNeXT_mistral_7B = 14798MiB at bfloat16 precision on 1 GPU
        if "LlavaNeXT" in model_name:
            from transformers import LlavaNextForConditionalGeneration, AutoProcessor

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {}
            
            ref_storage[model_name] = {
            "prompt":params[model_name]["prompt"],
            "question":params[model_name]["question"],
            "model":LlavaNextForConditionalGeneration.from_pretrained(
                    params[model_name]["model_id"], 
                    **model_load_params
                    # device_map="auto", 
                    # torch_dtype=torch.bfloat16
                    ),
            "processor":AutoProcessor.from_pretrained(params[model_name]["model_id"]),
            "generation_kwargs": params[model_name]["gen_kw"],

            "method":caption_LlavaNext_batched,
            "batch_inference":True
            }
            try:
                ref_storage[model_name]["model"].generation_config.pad_token_id = ref_storage[model_name]['model'].generation_config.eos_token_id
                ref_storage[model_name]["processor"].patch_size = ref_storage[model_name]["model"].config.vision_config.patch_size
                ref_storage[model_name]["processor"].vision_feature_select_strategy = ref_storage[model_name]["model"].config.vision_feature_select_strategy
                ref_storage[model_name]["processor"].tokenizer.padding_side = "left"            
            except:
                print(f"one of the settings [(pad_token_id to eos_token_id), patch_size, vision_feature_select_strategy] failed for {model_name}")
            print(f"{model_name} LOADED IN")
            print(os.system("nvidia-smi"))

        # DOESNOT WORK ON MULTI1-GPU settings
        # blip2-flan-t5-xxl ~ 36.426GB in 8bit
        if "BLIP2" in model_name:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {}
            
            ref_storage[model_name] = {
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "model":Blip2ForConditionalGeneration.from_pretrained(
                    params[model_name]["model_id"], 
                    **model_load_params
                    # device_map="auto", 
                    # quantization_config=BitsAndBytesConfig(load_in_8bit=True)
                    ),
                "processor":Blip2Processor.from_pretrained(params[model_name]["model_id"]),
                "generation_kwargs": params[model_name]["gen_kw"],
                "method":caption_blip2_prompted_batched
                }
            print(f"{model_name} LOADED IN")
            print(os.system("nvidia-smi"))
        
        # Cannot access without hf login
        #11B with bfloat16 ~ 21_166MiB
        if "Llama_3_2" in model_name:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            from huggingface_hub import login
            login(token=HUGGING_FACE_TOKEN)

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {}

            ref_storage[model_name] = {
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "model":MllamaForConditionalGeneration.from_pretrained(
                    params[model_name]["model_id"], 
                    **model_load_params
                    # device_map="auto", 
                    # torch_dtype=torch.bfloat16
                    ),
                "processor":AutoProcessor.from_pretrained(params[model_name]["model_id"]),
                "generation_kwargs": params[model_name]["gen_kw"],

                "method":caption_Llama_3_2_prompt
                }
            print(f"{model_name} LOADED IN")
            print(os.system("nvidia-smi"))

        # ~10GB at full precision
        if "Phi_3_5" in model_name:
            from transformers import AutoModelForCausalLM, AutoProcessor

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {}
            
            ref_storage[model_name] = {
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "model":AutoModelForCausalLM.from_pretrained(
                    params[model_name]["model_id"], 
                    **model_load_params),
                    # device_map="auto", 
                    # _attn_implementation='eager', 
                    # trust_remote_code=True, 
                    # torch_dtype=torch.bfloat16
                    # ),
                "processor":AutoProcessor.from_pretrained(
                    params[model_name]["model_id"], 
                    num_crops=16, 
                    trust_remote_code=True),
                "generation_kwargs": params[model_name]["gen_kw"],

                "method":caption_Phi_3_5_prompt_batched
                }
            print(f"{model_name} LOADED IN")
            print(os.system("nvidia-smi"))
    return 

#----------------------------------------------------------------------------------------------------------------------------------------------------------



def caption_VipLlava_batched(model_name: str, images: List[Image.Image]) -> Tuple[List[str], List[str]]:
    global ref_storage
    ref = ref_storage[model_name]
    
    
    # for img in images:
    if isinstance(ref['question'], list):
        # VipLlava_prompt = ref['prompt'].format(' '.join(ref['question']))
        per_question_batches = []
        generated_texts = []
        model_names = []
        images_rgb = [img.convert("RGB") for img in images]
        # VipLlava_prompts = []
        for VipLlava_Q in ref['question']:
            VipLlava_prompt = ref['prompt'].format(VipLlava_Q)
            # VipLlava_prompts.append(VipLlava_prompt)
            # VipLlava_prompts=VipLlava_prompts*len(images)
            inputs = ref["processor"](images=images_rgb, text=[VipLlava_prompt]*len(images), return_tensors="pt", padding=True).to(ref['model'].device)
            
            # Check if multi-temperature setting is enabled
            if "temperature" in ref["generation_kwargs"] and isinstance(ref["generation_kwargs"]["temperature"], str) and "multi" == ref["generation_kwargs"]["temperature"][:5]:
                multi_temp_param = ref["generation_kwargs"]["temperature"]

                # Define linspace range based on multi_temp_param
                if multi_temp_param == "multi1":
                    linspace_range = [0.1, 3]
                elif multi_temp_param == "multi2":
                    linspace_range = [1, 21]
                else:
                    raise ValueError(f"Unknown multi temperature param: {multi_temp_param}")

                # Loop through each temperature value
                for temp in np.linspace(linspace_range[0], linspace_range[1], num=5):
                    temp = round(temp, 3)
                    ref["generation_kwargs"]["temperature"] = temp

                    # Generate captions in batch
                    outputs = ref["model"].generate(
                        **inputs,
                        **ref["generation_kwargs"]
                    )
                    
                    # Decode and clean up all outputs in batch
                    decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
                    for idx, out in enumerate(decoded_outputs):
                        while "ASSISTANT:" in out:
                            out = out[out.index("ASSISTANT:") + 10:]
                        while "Assistant:" in out:
                            out = out[out.index("Assistant:") + 10:]
                        if len(generated_texts)<=idx:
                            generated_texts.append(f"Question: {VipLlava_Q} Answer: {out}.")
                        else:
                            generated_texts[idx]+=f"Question: {VipLlava_Q} Answer: {out}."
                        if len(model_names)<=idx:
                            model_names.append(f"{model_name}_{temp}t".replace('.', '_'))
                # Reset temperature value back to the original multi setting for future calls
                ref["generation_kwargs"]["temperature"] = multi_temp_param
            else:
                # Generate captions without multiple temperatures
                outputs = ref["model"].generate(
                    **inputs,
                    **ref["generation_kwargs"]
                )
                
                # Decode and clean up all outputs in batch
                decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
                for idx, out in enumerate(decoded_outputs):
                    while "<image>" in out:
                        out = out[out.index("<image>") + 7:]
                    while "ASSISTANT:" in out:
                        out = out[out.index("ASSISTANT:") + 10:]
                    while "Assistant:" in out:
                        out = out[out.index("Assistant:") + 10:]
                    if len(generated_texts)<=idx:
                        generated_texts.append(f"Question: {VipLlava_Q} Answer: {out}.")
                    else:
                        generated_texts[idx]+=f"Question: {VipLlava_Q} Answer: {out}."
                    if len(model_names)<=idx:
                        model_names.append(model_name)

    elif isinstance(ref['question'], str):
        # Prepare prompts for batch processing
        # VipLlava_prompts = []
        VipLlava_prompt = ref['prompt'].format(ref['question'])

        # VipLlava_prompts.append(VipLlava_prompt)
        # VipLlava_prompts=VipLlava_prompts*len(images)
        images_rgb = [img.convert("RGB") for img in images]
        
        # Process all images and prompts in batch
        inputs = ref["processor"](images=images_rgb, text=[VipLlava_prompt]*len(images), return_tensors="pt", padding=True).to(ref['model'].device)

        generated_texts = []
        model_names = []

        # Check if multi-temperature setting is enabled
        if "temperature" in ref["generation_kwargs"] and isinstance(ref["generation_kwargs"]["temperature"], str) and "multi" == ref["generation_kwargs"]["temperature"][:5]:
            multi_temp_param = ref["generation_kwargs"]["temperature"]

            # Define linspace range based on multi_temp_param
            if multi_temp_param == "multi1":
                linspace_range = [0.1, 3]
            elif multi_temp_param == "multi2":
                linspace_range = [1.5, 3]
            else:
                raise ValueError(f"Unknown multi temperature param: {multi_temp_param}")

            # Loop through each temperature value
            for temp in np.linspace(linspace_range[0], linspace_range[1], num=5):
                temp = round(temp, 3)
                ref["generation_kwargs"]["temperature"] = temp

                # Generate captions in batch
                outputs = ref["model"].generate(
                    **inputs,
                    **ref["generation_kwargs"]
                )
                
                # Decode and clean up all outputs in batch
                decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
                for out in decoded_outputs:
                    while "ASSISTANT:" in out:
                        out = out[out.index("ASSISTANT:") + 10:]
                    while "Assistant:" in out:
                        out = out[out.index("Assistant:") + 10:]
                    generated_texts.append(out)
                    model_names.append(f"{model_name}_{temp}t".replace('.', '_'))  # Append for each temperature

            # Reset temperature value back to the original multi setting for future calls
            ref["generation_kwargs"]["temperature"] = multi_temp_param
        else:
            # Generate captions without multiple temperatures
            outputs = ref["model"].generate(
                **inputs,
                **ref["generation_kwargs"]
            )
            
            # Decode and clean up all outputs in batch
            decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
            for out in decoded_outputs:
                while "ASSISTANT:" in out:
                    out = out[out.index("ASSISTANT:") + 10:]
                while "Assistant:" in out:
                    out = out[out.index("Assistant:") + 10:]
                generated_texts.append(out)
                model_names.append(model_name)
    else:
        raise Exception(f"for {model_name} the question instance type is not supported")
        
    

    return generated_texts, model_names

def caption_LlavaNext_batched(model_name: str, images: List[Image.Image]) -> Tuple[List[str], List[str]]:
    global ref_storage
    ref = ref_storage[model_name]
    
    # Prepare prompts for batch processing
    LlavaNeXT_prompts = []
    if isinstance(ref['question'], list):
        LlavaNeXT_prompt = ref['prompt'].format(' '.join(ref['question']))
    elif isinstance(ref['question'], str):
        LlavaNeXT_prompt = ref['prompt'].format(ref['question'])
    else:
        raise Exception(f"for {model_name} the question instance type is not supported")
    
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": LlavaNeXT_prompt}]}
    ]
    LlavaNeXT_prompts.append(ref["processor"].apply_chat_template(conversation, add_generation_prompt=True))
    #all prompts are the same for every image - otherwise loop
    LlavaNeXT_prompts=LlavaNeXT_prompts*len(images)
    # Process all images and prompts in batch
    images_rgb = [img.convert("RGB") for img in images]
    inputs = ref["processor"](images=images_rgb, text=LlavaNeXT_prompts, return_tensors="pt", padding=True).to(ref['model'].device)

    generated_texts = []  # Initialize at the right scope to accumulate results across all temperature runs
    model_names = []

    # Check if multi-temperature setting is enabled
    if "temperature" in ref["generation_kwargs"] and isinstance(ref["generation_kwargs"]["temperature"], str) and "multi" == ref["generation_kwargs"]["temperature"][:5]:
        multi_temp_param = ref["generation_kwargs"]["temperature"]

        # Define linspace range based on multi_temp_param
        if multi_temp_param == "multi1":
            linspace_range = [0.1, 3]
        elif multi_temp_param == "multi2":
            linspace_range = [1.5, 3]
        else:
            raise ValueError(f"Unknown multi temperature param: {multi_temp_param}")

        # Loop through each temperature value
        for temp in np.linspace(linspace_range[0], linspace_range[1], num=10):
            temp = round(temp, 3)
            ref["generation_kwargs"]["temperature"] = temp

            # Generate captions in batch
            outputs = ref["model"].generate(
                **inputs,
                **ref["generation_kwargs"]
            )
            
            # Decode and clean up all outputs in batch
            decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
            for out in decoded_outputs:
                if "/INST]" in out:
                    out = out[out.index("/INST]") + 6:]
                if "ASSISTANT:" in out:
                    out = out[out.index("ASSISTANT:") + 10:]
                generated_texts.append(out)
                model_names.append(f"{model_name}_{temp}t".replace('.', '_'))  # Append for each temperature

        # Reset temperature value back to the original multi setting for future calls
        ref["generation_kwargs"]["temperature"] = multi_temp_param
    else:
        # Generate captions without multiple temperatures
        outputs = ref["model"].generate(
            **inputs,
            **ref["generation_kwargs"]
        )
        
        # Decode and clean up all outputs in batch
        decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
        for out in decoded_outputs:
            if "/INST]" in out:
                out = out[out.index("/INST]") + 6:]
            if "ASSISTANT:" in out:
                out = out[out.index("ASSISTANT:") + 10:]
            generated_texts.append(out)
            model_names.append(model_name)

    return generated_texts, model_names


def caption_blip2_prompted_batched(model_name: str, images: List[Image.Image]) -> Tuple[List[str], List[str]]:
    """Generate captions for a batch of images using BLIP2 model with prompts, including multi-temperature support."""

    global ref_storage
    ref = ref_storage[model_name]
    
    # Prepare the prompt
    if isinstance(ref['question'], list):
        per_question_batches = []
        generated_texts = []
        model_names = []
        images_rgb = [img.convert("RGB") for img in images]
        for BLIP2_q in ref['question']:    
            BLIP2_prompt = ref['prompt'].format(BLIP2_q)
            # Prepare inputs for batch processing
            inputs = ref["processor"](
                images=images_rgb,
                text=[BLIP2_prompt] * len(images),  # Same prompt applied to each image in the batch
                return_tensors="pt",
                padding=True
            ).to(ref["model"].device)

            # Check if multi-temperature setting is enabled
            if "temperature" in ref["generation_kwargs"] and isinstance(ref["generation_kwargs"]["temperature"], str) and "multi" == ref["generation_kwargs"]["temperature"][:5]:
                multi_temp_param = ref["generation_kwargs"]["temperature"]

                # Define linspace range based on multi_temp_param
                if multi_temp_param == "multi1":
                    linspace_range = [0.1, 3]
                elif multi_temp_param == "multi2":
                    linspace_range = [1, 21]
                else:
                    raise ValueError(f"Unknown multi temperature param: {multi_temp_param}")

                # Loop through each temperature value
                for temp in np.linspace(linspace_range[0], linspace_range[1], num=5):
                    temp = round(temp, 3)
                    ref["generation_kwargs"]["temperature"] = temp

                    # Generate captions in batch
                    outputs = ref["model"].generate(
                        **inputs,
                        **ref["generation_kwargs"]
                    )
                    
                    # Decode and clean up all outputs in batch
                    decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
                    for idx, out in enumerate(decoded_outputs):
                        if len(generated_texts)<=idx:
                            generated_texts.append(BLIP2_prompt + " " + out + ". ")
                        else:
                            generated_texts[idx]+=BLIP2_prompt + " " + out + ". "
                        if len(model_names)<=idx:
                            model_names.append(f"{model_name}_{temp}t".replace('.', '_'))

                # Reset temperature value back to the original multi setting for future calls
                ref["generation_kwargs"]["temperature"] = multi_temp_param
            else:
                # Generate captions without multiple temperatures
                outputs = ref["model"].generate(
                    **inputs,
                    **ref["generation_kwargs"]
                )
                
                # Decode and clean up all outputs in batch
                decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
                for idx, out in enumerate(decoded_outputs):
                    if len(generated_texts)<=idx:
                        generated_texts.append(BLIP2_prompt + " " + out + ". ")
                    else:
                        generated_texts[idx]+=BLIP2_prompt + " " + out + ". "
                    if len(model_names)<=idx:
                        model_names.append(model_name)
            
    elif isinstance(ref['question'], str):
        BLIP2_prompt = ref['prompt'].format(ref['question'])
        generated_texts = []
        model_names = []
        # Convert each image to RGB format for consistency
        images_rgb = [img.convert("RGB") for img in images]
        
        # Prepare inputs for batch processing
        inputs = ref["processor"](
            images=images_rgb,
            text=[BLIP2_prompt] * len(images),  # Same prompt applied to each image in the batch
            return_tensors="pt",
            padding=True
        ).to(ref["model"].device)

        # Check if multi-temperature setting is enabled
        if "temperature" in ref["generation_kwargs"] and isinstance(ref["generation_kwargs"]["temperature"], str) and "multi" == ref["generation_kwargs"]["temperature"][:5]:
            multi_temp_param = ref["generation_kwargs"]["temperature"]

            # Define linspace range based on multi_temp_param
            if multi_temp_param == "multi1":
                linspace_range = [0.1, 3]
            elif multi_temp_param == "multi2":
                linspace_range = [1.5, 3]
            else:
                raise ValueError(f"Unknown multi temperature param: {multi_temp_param}")

            # Loop through each temperature value
            for temp in np.linspace(linspace_range[0], linspace_range[1], num=5):
                temp = round(temp, 3)
                ref["generation_kwargs"]["temperature"] = temp

                # Generate captions in batch
                outputs = ref["model"].generate(
                    **inputs,
                    **ref["generation_kwargs"]
                )
                
                # Decode and clean up all outputs in batch
                decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
                for out in decoded_outputs:
                    generated_texts.append(out)
                    model_names.append(f"{model_name}_{temp}t".replace('.', '_'))  # Append for each temperature

            # Reset temperature value back to the original multi setting for future calls
            ref["generation_kwargs"]["temperature"] = multi_temp_param
        else:
            # Generate captions without multiple temperatures
            outputs = ref["model"].generate(
                **inputs,
                **ref["generation_kwargs"]
            )
            
            # Decode and clean up all outputs in batch
            decoded_outputs = ref["processor"].batch_decode(outputs, skip_special_tokens=True)
            for out in decoded_outputs:
                generated_texts.append(out)
                model_names.append(model_name)
    else:
        raise Exception(f"For {model_name} the question instance type is not supported")


    return generated_texts, model_names

def caption_Llama_3_2_prompt(model_name: str, images: List[Image.Image]) -> Tuple[List[str], List[str]]:
    """Generate captions for a single image using Llama 3.2 model, with multi-temperature support."""
    
    global ref_storage
    ref = ref_storage[model_name]
    
    # Prepare the prompt based on question type
    if isinstance(ref['question'], list):
        prompt = ref['prompt'].format(' '.join(ref['question']))
    elif isinstance(ref['question'], str):
        prompt = ref['prompt'].format(ref['question'])
    else:
        raise Exception(f"For {model_name} the question instance type is not supported")
    Llama_3_2_prompts = []
    # Apply chat template to format the prompt
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    Llama_3_2_prompts.append(ref["processor"].apply_chat_template(messages, add_generation_prompt=True))
    Llama_3_2_prompts=Llama_3_2_prompts*len(images)
    images_rgb = [img.convert("RGB") for img in images]
    # Process the input image and text prompt
    inputs = ref["processor"](
        images=images_rgb,
        text=Llama_3_2_prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True
    ).to(ref["model"].device)

    generated_texts = []
    model_names = []

    # Check for multi-temperature setting
    if "temperature" in ref["generation_kwargs"] and isinstance(ref["generation_kwargs"]["temperature"], str) and "multi" == ref["generation_kwargs"]["temperature"][:5]:
        multi_temp_param = ref["generation_kwargs"]["temperature"]

        # Define linspace range based on multi_temp_param
        if multi_temp_param == "multi1":
            linspace_range = [0.1, 1.5]
        elif multi_temp_param == "multi2":
            linspace_range = [1.5, 3]
        else:
            raise ValueError(f"Unknown multi temperature param: {multi_temp_param}")

        # Loop through each temperature value
        for temp in np.linspace(linspace_range[0], linspace_range[1], num=5):
            temp = round(temp, 3)
            ref["generation_kwargs"]["temperature"] = temp

            # Generate caption
            output = ref["model"].generate(
                **inputs,
                **ref["generation_kwargs"]
            )

            # Decode and clean the generated output
            decoded_outputs = ref["processor"].batch_decode(output, skip_special_tokens=True)
            for generated_text in decoded_outputs:
                # # cleanup from question
                # if isinstance(ref['question'], list):
                #     if ref['question'][-1] in generated_text:
                #         generated_text=generated_text[generated_text.index(ref['question'][-1])+len(ref['question'][-1])]
                # # cleanup from question
                # elif isinstance(ref['question'], str):
                #     if ref['question'] in generated_text:
                #         generated_text=generated_text[generated_text.index(ref['question'])+len(ref['question'])]
                
                generated_texts.append(generated_text)
                model_names.append(f"{model_name}_{temp}t".replace('.', '_'))

        # Reset temperature to its original multi setting
        ref["generation_kwargs"]["temperature"] = multi_temp_param
    else:
        # Generate a single caption without multi-temperature handling
        output = ref["model"].generate(
            **inputs,
            **ref["generation_kwargs"]
        )
        
        # Decode and clean the generated output
        decoded_outputs = ref["processor"].batch_decode(output, skip_special_tokens=True)
        for generated_text in decoded_outputs:
            # cleanup from question
            if isinstance(ref['question'], list):
                if ref['question'][-1] in generated_text:
                    generated_text=generated_text[generated_text.index(ref['question'][-1])+len(ref['question'][-1])]
            # cleanup from question
            elif isinstance(ref['question'], str):
                if ref['question'] in generated_text:
                    generated_text=generated_text[generated_text.index(ref['question'])+len(ref['question'])]
            
            generated_texts.append(generated_text)
            model_names.append(model_name)

    return generated_texts, model_names

# Phi_3_5 only works with batch_size==1 # https://github.com/microsoft/Phi-3CookBook/issues/158
# I attempted to use this: https://github.com/microsoft/Phi-3CookBook/issues/201 but it started taking too much time
def caption_Phi_3_5_prompt_batched(model_name: str, images: List[Image.Image]) -> Tuple[List[str], List[str]]:
    """Generate captions for a batch of images using the Phi 3.5 model with prompts and temperature variations."""

    # # Batching function
    # def create_batched_input(input_list, model, processor):
    #     """
    #     Create a batched input dictionary from a list of processed inputs.
    #     """
    #     def pad_sequence(sequences, padding_side='right', padding_value=0):
    #         """
    #         Pad a list of sequences to the same length.
    #         sequences: list of tensors in [seq_len, *] shape
    #         """
    #         assert padding_side in ['right', 'left']
    #         max_size = sequences[0].size()
    #         trailing_dims = max_size[1:]
    #         max_len = max(len(seq) for seq in sequences)
    #         batch_size = len(sequences)
    #         output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    #         for i, seq in enumerate(sequences):
    #             length = seq.size(0)
    #             if padding_side == 'right':
    #                 output.data[i, :length] = seq
    #             else:
    #                 output.data[i, -length:] = seq
    #         return output


    #     batched_input_id = []
    #     batched_pixel_values = []
    #     batched_image_sizes = []
        
    #     for inp in input_list:
    #         batched_input_id.append(inp["input_ids"].squeeze(0))
    #         batched_pixel_values.append(inp["pixel_values"])
    #         batched_image_sizes.append(inp["image_sizes"])

    #     input_ids = pad_sequence(batched_input_id, padding_side='right', padding_value=model.pad_token_id)
    #     attention_mask = input_ids != model.pad_token_id
    #     pixel_values = torch.cat(batched_pixel_values, dim=0)
    #     image_sizes = torch.cat(batched_image_sizes, dim=0)

    #     batched_input = {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "pixel_values": pixel_values,
    #         "image_sizes": image_sizes
    #     }
        
    #     return batched_input
    
    ref = ref_storage[model_name]

    # Ensure the prompt is a single string
    if isinstance(ref['question'], list):
        Phi_3_5_prompt = ref['prompt'].format(' '.join(ref['question']))
    elif isinstance(ref['question'], str):
        Phi_3_5_prompt = ref['prompt'].format(ref['question'])
    else:
        raise Exception(f"For {model_name} the question instance type is not supported")

    # Prepare each image individually

    messages = [
        {"role": "user", "content": "<|image_1|>\n" + Phi_3_5_prompt},
    ]
    prompt = ref["processor"].tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    processed_inputs = [
        ref["processor"](
            images=[img.convert("RGB")],  # Convert to RGB for consistency
            text=prompt,
            return_tensors="pt"
        ).to(ref['model'].device)
        for img in images
    ]

    # Create batched input
    # batched_input = create_batched_input(processed_inputs, ref["model"], ref["processor"])

    # Check if multi-temperature generation is requested
    generated_texts = []
    model_names = []
    if "temperature" in ref["generation_kwargs"] and isinstance(ref["generation_kwargs"]["temperature"], str) and "multi" == ref["generation_kwargs"]["temperature"][:5]:
        multi_temp_param = ref["generation_kwargs"]["temperature"]
        
        # Set temperature ranges
        if multi_temp_param == "multi1":
            linspace_range = [0.1, 1.5]
        elif multi_temp_param == "multi2":
            linspace_range = [1.5, 3.0]
        else:
            raise ValueError(f"Unsupported multi-temp parameter: {multi_temp_param}")

        # Generate captions for each temperature value in the range
        for temp in np.linspace(linspace_range[0], linspace_range[1], num=5):
            temp = round(temp, 3)
            ref["generation_kwargs"]["temperature"] = temp

            # Generate captions for each image individually with current temperature
            for inputs in processed_inputs:
                generated_ids = ref["model"].generate(
                    **inputs,
                    eos_token_id=ref["processor"].tokenizer.eos_token_id,
                    **ref["generation_kwargs"]
                )
                generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]

                # Decode generated text without batch processing
                temp_generated_text = ref["processor"].batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]


                # Append the generated text and corresponding model name
                generated_texts.append(temp_generated_text)
                model_names.append(f"{model_name}_{temp}t".replace('.', '_'))

        # Reset temperature to original value
        ref["generation_kwargs"]["temperature"] = multi_temp_param
    else:
        # Single temperature generation for each image
        for inputs in processed_inputs:
            generated_ids = ref["model"].generate(
                **inputs,
                eos_token_id=ref["processor"].tokenizer.eos_token_id,
                **ref["generation_kwargs"]
            )

            # Decode generated text
            generated_text = ref["processor"].decode(
                generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Append results
            generated_texts.append(generated_text)
            model_names.append(model_name)

    return generated_texts, model_names


#----------------------------------------------------------------------------------------------------------------------------------------------------------

def read_json(file_path: str) -> dict:
    """Read the JSON file and return the data as a dictionary."""
    if Path(file_path).exists():
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except:
            return {}
    else:
        return {}

def write_json(file_path: str, data: dict):
    """Write the updated dictionary to the JSON file."""
    if not Path(file_path).exists():
        print("creating ",file_path, ' status: ',os.system('touch'+str(file_path)))

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def append_to_json_file(file_path: str, new_data: dict):
    """
    Append new data to the existing JSON file. 
    NOTE: only first level of key-value pairs considered
    """
    existing_data = read_json(file_path)
    for key, value in new_data.items():
        if key in existing_data:
            existing_data[key].update(value)
        else:
            existing_data[key] = value
    write_json(file_path, existing_data)

def parse_args(args):
    global HUGGING_FACE_TOKEN
    global GOOGLE_API_KEY
    global OPENAI_API_KEY
    global TEST_MODE
    global BATCH_SIZE

    caption_models = args.caption_models
    images_dir = args.images_dir
    params_file_path = args.params_file_path
    output_file = args.output_file

    HUGGING_FACE_TOKEN = args.HUGGING_FACE_TOKEN
    GOOGLE_API_KEY = args.GOOGLE_API_KEY
    OPENAI_API_KEY = args.OPENAI_API_KEY

    TEST_MODE = args.TEST_MODE

    BATCH_SIZE = args.BATCH_SIZE
    if BATCH_SIZE is not None:
        try:
            BATCH_SIZE=int(BATCH_SIZE)
        except:
            BATCH_SIZE=BATCH_SIZE_default
    else:
        BATCH_SIZE = BATCH_SIZE_default
    
    if caption_models is None:
        raise Exception("caption_model is not provided")
    if caption_models.find(',')!=-1:
        caption_models = caption_models.split(',')
    else:
        caption_models = [caption_models]
    for i in range(len(caption_models)):
        caption_models[i]=caption_models[i].strip()

    if images_dir is not None:
        try:
            images_dir = Path(images_dir)
        except:
            print(traceback.format_exc())
            raise Exception
    else:
        raise Exception("images_dir is not provided")

    if params_file_path is not None:
        try:
            params_file_path = Path(params_file_path)
        except:
            print(traceback.format_exc())
            raise Exception
    else:
        raise Exception("params_file_path is not provided")

    if output_file is not None:
        try:
            output_file = Path(output_file)
        except:
            raise Exception
    else:
        raise Exception("output_file is not provided")


    if TEST_MODE is not None:
        try:
            TEST_MODE=int(TEST_MODE)
        except:
            TEST_MODE=1
    else:
        TEST_MODE = 0

    return caption_models, images_dir, params_file_path, output_file



class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # img = Image.open(img_path).convert("RGB")  # Open and convert to RGB to ensure consistency
        return img_path#, img  # Return the image and its path

def collate_fn(batch):
    # paths, images  = zip(*batch)  # Unpack the list of tuples into two lists
    # return list(paths), list(images)
    return batch


#----------------------------------------------------------------------------------------------------------------------------------------------------------


def process_images_batched(img_paths: List[Path] = None, img_names: List[str] = None, imgs: List[Image.Image] = None) -> Dict[str, Dict[str, str]]:
    """Process a batch of images: open, generate captions, and return data entries for each image."""
    
    # Validate inputs & Prepare images and names if only paths are provided
    if img_paths:
        imgs = [Image.open(img_path) for img_path in img_paths]
        img_names = [img_path.name for img_path in img_paths]
    elif img_paths is None and imgs is None:
        raise Exception("Either img_paths or imgs must be provided")
    elif imgs and img_names is None:
        raise Exception("img_names must be provided when imgs are used directly")

    # Initialize the results dictionary
    batch_results = {}
    
    try:
        # For each model, perform batched inference
        for model_name in ref_storage:
            # Call the batched captioning method for the current model
            captions, model_names = ref_storage[model_name]["method"](model_name=model_name, images=imgs)

            for i, img_caption in enumerate(captions):
                idx1 = i%len(img_names)
                img_name = img_names[idx1]
                if img_name not in batch_results:
                    batch_results[img_name] = {}
                batch_results[img_name][model_names[i]] = captions[i]
                
    except Exception as e:
        print("An error occurred:", e)
        print(traceback.format_exc())
        raise

    return batch_results
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_models', help="flip_split")
    parser.add_argument('--images_dir', help="data_dir")
    parser.add_argument('--params_file_path', help="output_file should (must) be equal to the one from params_file")
    parser.add_argument('--output_file', help="caption_model name as in params_caption files or built-in if supported")

    parser.add_argument('--HUGGING_FACE_TOKEN', help="HUGGING_FACE_TOKEN as string")
    parser.add_argument('--GOOGLE_API_KEY', help="GOOGLE_API_KEY as string")
    parser.add_argument('--OPENAI_API_KEY', help="OPENAI_API_KEY as string") 
    parser.add_argument('--BATCH_SIZE', help="BATCH_SIZE")

    parser.add_argument('--TEST_MODE', help="TEST_MODE as int")

    caption_models, images_dir, params_file_path, output_file = parse_args(parser.parse_args())



    global ref_storage
    global image_captions
    if not params_file_path.exists():
        print(params_file_path, params_file_path.exists())
        raise Exception("job with no params file prevented")
    
    load_exp_params(models_list = caption_models, params_file=params_file_path)

    # Paths and constants
    # image_captions = read_json(output_file) # continue process for broken pipes NOTE: need to account for in the main loop as well


    # Depending on the image locations (all images in single folder / images are placed together with flips)
    #  change/uncomment one of the definition bellow:
    # if all images are in single folder:
    image_paths_names = [images_dir.joinpath(img_name) for img_name in os.listdir(images_dir) if img_name.endswith('.png')]
    # if images are per flip folder:
    # image_paths_names = []
    # for flip_challenge_dir in os.listdir(images_dir):
    #     for flip_file in os.listdir(flip_challenge_dir):
    #         if flip_file.endswith('.png'):
    #             image_paths_names.append(images_dir.joinpath(flip_challenge_dir, flip_file))

    if TEST_MODE:
        print("TEST_MODE, sample 2 BATCH_SIZEs (adds on memory stress for activation) images")
        image_paths_names=image_paths_names[:2*BATCH_SIZE]

    dataset = ImagePathDataset(image_paths_names)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    


    for batch_n, batch in enumerate(tqdm(data_loader, desc="Processing images")):
        batch_results = {}
        results = []
        # for img_path_name, image in batch:
        results.append(process_images_batched(img_paths=batch))
        torch.cuda.empty_cache()
        
        # Combine the batch results into one dictionary
        batch_results = {}
        for result in results:
            if batch_n == 0:
                print(f"some latest results:: {result}")
            batch_results.update(result)
        
        # Append to the JSON file after each batch
        append_to_json_file(output_file, batch_results)

        

    print("DONE")

if __name__ == '__main__':
    main()


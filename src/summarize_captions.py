from pathlib import Path
import os
import json
import traceback
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import ast
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

import random

import warnings
from transformers import pipeline
from qwen_vl_utils import process_vision_info

BATCH_SIZE_default = 4
MAX_WORKERS_NUM=4


HUGGING_FACE_TOKEN =None
GOOGLE_API_KEY=None
OPENAI_API_KEY=None
WANDB_API_KEY=None
context_dir=None
stable_context = None
TEST_MODE=0
VERBOSE=0
TASK_FORMAT="single_sentence"
gemini_safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

ref_storage={}

def load_image_captions(caption_models=[], caption_file_paths=[]):
    image_captions = {}
    for caption_model, caption_file_path in zip(caption_models,caption_file_paths):
        if caption_model == 'built_in':
            continue
        these_image_captions = read_json(caption_file_path)
        for key,value in these_image_captions.items():
            if key in image_captions:
                image_captions[key][caption_model]=value[caption_model]
            else:
                image_captions[key]={
                    caption_model:value[caption_model]
                }
        if VERBOSE:
            print(f"\nLOADED: \n {caption_model} \n ")
            print(f"\tFROM {caption_file_path} \n ")

    return image_captions

def load_exp_params(models_list = ['LlavaNeXT'], params_file = None):
    global ref_storage
    if TEST_MODE:
        print(f"loading models_list: {models_list}")
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
        if "device_map" in model_load_params and model_load_params["device_map"] == "split_model":
            from utils import split_model
            model_load_params["device_map"] = split_model() 
        return model_load_params
    
    
    if params_file is not None:
        params=read_json(params_file)
    else:
        raise Exception("no param file provided")
        
    for model_name in models_list:

        if "Phi_3_5" in model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {
                    "device_map":"auto",  
                    "torch_dtype":torch.bfloat16,  
                    "trust_remote_code":True,
                }
            
            ref_storage[model_name]={
                "model_id":params[model_name]["model_id"],
                "model":AutoModelForCausalLM.from_pretrained( 
                            params[model_name]["model_id"],
                            **model_load_params
                        ) ,
                "processor":AutoTokenizer.from_pretrained(params[model_name]["model_id"]) ,
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "instructions":params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context":params[model_name]["context_n"],

                "method": reason_Phi_3_5,
                "support_image_input": False # there is Phi_3.5_vision available
            }

            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))

        if "GRIN_MoE" in model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {
                "device_map":"auto",  
                "torch_dtype":torch.bfloat16,  
                "trust_remote_code":True,  
                }
            
            ref_storage[model_name]={
                "model_id":params[model_name]["model_id"],
                "model":AutoModelForCausalLM.from_pretrained( 
                            params[model_name]["model_id"],
                            **model_load_params
                        ) ,
                "processor":AutoTokenizer.from_pretrained(params[model_name]["model_id"]),
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "instructions":params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context":params[model_name]["context_n"],

                "method": reason_GRIN_MoE,
                "support_image_input": False
            }
            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))

        if "Qwen_2_5" in model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {
                    "torch_dtype":"auto",
                    "device_map":"auto"
                }
            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model":AutoModelForCausalLM.from_pretrained(
                            params[model_name]["model_id"],
                            **model_load_params
                        ),
                "processor":AutoTokenizer.from_pretrained(params[model_name]['model_id'],padding_side="left"),
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "instructions":params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context":params[model_name]["context_n"],

                "method": reason_Qwen_2_5,
                "support_image_input": False
            }
            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))

        if "Qwen_2_VL" in model_name:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                model_load_params = {
                    "torch_dtype":"auto", 
                    "attn_implementation":"flash_attention_2",
                    "device_map":"auto"
                }

            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model": Qwen2VLForConditionalGeneration.from_pretrained(
                    params[model_name]["model_id"], 
                    **model_load_params
                ),
                "processor":AutoProcessor.from_pretrained(params[model_name]["model_id"],padding_side="left"),
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "instructions":params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context":params[model_name]["context_n"],

                "method": reason_Qwen_2_VL,
                "support_image_input": True
            }
            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))
        
        if "nvm_Llama3" in model_name:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                from transformers import BitsAndBytesConfig
                model_load_params = {
                    "device_map":"auto", 
                    "torch_dtype":torch.bfloat16
                }

            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model":AutoModelForCausalLM.from_pretrained(
                            params[model_name]["model_id"],
                            **model_load_params
                        ),
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "instructions": params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context": params[model_name]["context_n"],
                "processor":AutoTokenizer.from_pretrained(params[model_name]["model_id"], padding_side="left"),

                "method": reason_nvm_Llama3,
                "support_image_input": False 
                }
            ref_storage[model_name]['processor'].pad_token_id = ref_storage[model_name]['processor'].eos_token_id

            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))
        
        if "NVLM" in model_name:
            from transformers import AutoModel, AutoTokenizer
            from utils import build_transform, find_closest_aspect_ratio, dynamic_preprocess, load_image
            if TEST_MODE:
                print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                from utils import split_model
                model_load_params = {
                    "torch_dtype":torch.bfloat16,
                    "use_flash_attn":False,
                    "trust_remote_code":True,
                    "device_map":split_model()
                }

            if TEST_MODE:
                print(f"NVLM device_map: : {model_load_params['device_map']}")

            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model":AutoModel.from_pretrained(
                            params[model_name]["model_id"],
                            **model_load_params
                        ).eval(),
                "processor":AutoTokenizer.from_pretrained(
                                params[model_name]["model_id"], 
                                trust_remote_code=True,
                                use_fast=False,padding_side="left"),
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "instructions": params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context": params[model_name]["context_n"],

                "method": reason_NVLM,
                "support_image_input": False 
                }

            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))
        
        if "meta_Llama_3_2" in model_name:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            from huggingface_hub import login
            login(token=HUGGING_FACE_TOKEN)

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                from transformers import BitsAndBytesConfig
                model_load_params = {
                    "device_map":"auto", 
                    "torch_dtype":torch.bfloat16,
                    # "quantization_config":BitsAndBytesConfig(load_in_8bit=True)
                }


            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model":MllamaForConditionalGeneration.from_pretrained(
                            params[model_name]["model_id"],
                            **model_load_params
                        ),
                "processor":AutoProcessor.from_pretrained(params[model_name]["model_id"],padding_side="left"),
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]['question'],
                "instructions": params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context": params[model_name]["context_n"],

                "method": reason_meta_Llama_3_2,
                "support_image_input": True 
                }
            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))

        if "meta_Llama_3_1" in model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import login
            login(token=HUGGING_FACE_TOKEN)

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                # from transformers import BitsAndBytesConfig
                model_load_params = {
                    "device_map":"auto", 
                    "torch_dtype":torch.bfloat16,
                    # "quantization_config":BitsAndBytesConfig(load_in_8bit=True)
                }

            # if "quantization_config" in model_load_params:

            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model":AutoModelForCausalLM.from_pretrained(
                            params[model_name]["model_id"],
                            **model_load_params
                        ),
                "processor":AutoTokenizer.from_pretrained(params[model_name]["model_id"]),
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]['question'],
                "instructions": params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context": params[model_name]["context_n"],

                "method": reason_meta_Llama_3_1,
                "support_image_input": False 
                }
            ref_storage[model_name]['processor'].pad_token_id = ref_storage[model_name]['processor'].eos_token_id

            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))

    return 


def get_context(cnt:int=5, model_name:str="Qwen_2_5_8bit", strategy:str="balanced"):
    global stable_context
    if stable_context is not None:
        if TEST_MODE:
            print(f"returning stable_context")
        return stable_context
    context = {}
    if context_dir is None:
        if TEST_MODE:
            print(f"returning empty context")
        return context

    ref = ref_storage[model_name]["context"]

    context_path = context_dir.joinpath(ref["context_reasoning_model"], ref["context_captioning_model"])
    if not context_path.exists():
        if TEST_MODE:
            print(f"returning empty context")
        return context

    if strategy == "balanced":
        negative_samples = read_json(context_path.joinpath("negative_samples.json"))
        positive_samples = read_json(context_path.joinpath("positive_samples.json"))

        context["positive_context"] = random.sample(positive_samples, cnt)
        context["negative_context"] = random.sample(negative_samples, cnt)

    elif strategy == "positive":
        positive_samples = context_path.joinpath("positive_samples.json")
        context["positive_context"] = random.sample(positive_samples, cnt)

    elif strategy == "negative":
        negative_samples = context_path.joinpath("negative_samples.json")
        context["negative_context"] = random.sample(negative_samples, cnt)
    else:
        raise NotImplementedError


    stable_context = context
    if TEST_MODE:
        print(f"returning stable_context: {stable_context}")
    
    return stable_context




        






#----------------------------------------------------------------------------------------------------------------------------------------------------------

def reason_GRIN_MoE(task_captions, model_name):
    

    
    if isinstance(task_captions, str):
        message_content = " ".join(ref_storage[model_name]["question"]) + "\n " + ref_storage[model_name]["instructions"] + "\n " + task_captions
        
    elif isinstance(task_captions, list):
        captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
        random.shuffle(captions)
        captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
        captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
        captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

        message_content = ref_storage[model_name]["question"][0] + "\n " + \
            captions_to_img_names + \
            ref_storage[model_name]["question"][1] + "\n " + \
            ref_storage[model_name]["question"][2] + "\n " + \
            f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
            ref_storage[model_name]["question"][3] + "\n " + \
            ref_storage[model_name]["question"][4] + "\n " + \
            ref_storage[model_name]["instructions"] 

    def get_res_out(generated_text, message_content):
        if TEST_MODE:
            print(f"generated_text {generated_text}")
        # if "Solution"in generated_text:
        #     generated_text = generated_text[generated_text.index("Solution"):]
        model_response = generated_text
        # for char in model_response:
        #     if char.isdigit():
        #         model_response = char
        #         break
        return model_response
    
    messages = [ 
        {"role": "system", "content": ref_storage[model_name]["prompt"]}, 

        {
            'role': 'user',
            'content': message_content,
        }
    ] 

    pipe = pipeline( 
        "text-generation", 
        model=ref_storage[model_name]["model"], 
        tokenizer=ref_storage[model_name]["processor"]
    ) 

    output = pipe(messages, **ref_storage[model_name]["generation_kwargs"]) 

    model_response = get_res_out(output[0]['generated_text'], message_content)    
    
    return model_response, model_name


def reason_Phi_3_5(task_captions, model_name):

    
    if isinstance(task_captions, str):
        message_content = " ".join(ref_storage[model_name]["question"]) + \
            "\n " + ref_storage[model_name]["instructions"] + \
            "\n " + task_captions
        
        
    elif isinstance(task_captions, list):
        captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
        random.shuffle(captions)
        captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
        captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
        captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

        message_content = ref_storage[model_name]["question"][0] + "\n " + \
            captions_to_img_names + \
            ref_storage[model_name]["question"][1] + "\n " + \
            ref_storage[model_name]["question"][2] + "\n " + \
            f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
            ref_storage[model_name]["question"][3] + "\n " + \
            ref_storage[model_name]["question"][4] + "\n " + \
            ref_storage[model_name]["instructions"] 

    def get_res_out(generated_text, message_content):
        if TEST_MODE:
            print(f"generated_text {generated_text}")
        # if "Solution"in generated_text:
        #     generated_text = generated_text[generated_text.index("Solution"):]
        model_response = generated_text
        # for char in model_response:
        #     if char.isdigit():
        #         model_response = char
        #         break
        return model_response
    
    # messages = [ 
    #     {"role": "system", "content": "You are a helpful AI assistant."}, 
    # ] 
    messages=[
        {"role": "system", "content": ref_storage[model_name]["prompt"]},

        {
            'role': 'user',
            'content': message_content,
        }
    ]

    pipe = pipeline( 
        "text-generation", 
        model=ref_storage[model_name]["model"], 
        tokenizer=ref_storage[model_name]["processor"], 
    ) 

    output = pipe(messages, **ref_storage[model_name]["generation_kwargs"]) 

    model_response = get_res_out(output[0]['generated_text'], message_content)    
    
    return model_response, model_name


# def reason_meta_Llama_3_1_pipeline(task_captions, model_name):
#     if isinstance(task_captions, str):
#         message_content = " ".join(ref_storage[model_name]["question"]) + \
#             "\n " + ref_storage[model_name]["instructions"] + \
#             "\n " + task_captions
        
        
#     elif isinstance(task_captions, list):
#         captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
#         random.shuffle(captions)
#         captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
#         captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
#         captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

#         message_content = ref_storage[model_name]["question"][0] + "\n " + \
#             captions_to_img_names + \
#             ref_storage[model_name]["question"][1] + "\n " + \
#             ref_storage[model_name]["question"][2] + "\n " + \
#             f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
#             ref_storage[model_name]["question"][3] + "\n " + \
#             ref_storage[model_name]["question"][4] + "\n " + \
#             ref_storage[model_name]["instructions"] 

#     messages = [
#         {"role": "system", "content": ref_storage[model_name]["prompt"]},
#         {"role": "user", "content": message_content},
#     ]

#     def get_res_out(generated_text, message_content):
#         if TEST_MODE:
#             print(f"generated_text {generated_text}")
#         if "Solution"in generated_text:
#             generated_text = generated_text[generated_text.index("Solution"):]
#         model_response = generated_text
#         for char in generated_text:
#             if char.isdigit():
#                 model_response = char
#                 break
#         return model_response

#     outputs = ref_storage[model_name]["model"](
#         messages,
#         max_new_tokens=256,
#         temperature=0.2
#     )
#     outs = outputs[0]["generated_text"][-1]
#     print(outs['content'])
#     model_response=get_res_out(outs['content'], message_content)
    
#     return model_response, model_name


def reason_meta_Llama_3_1(tasks, model_name):
    ref = ref_storage[model_name]
    message_contents = []
    for task_captions in tasks:

        if isinstance(task_captions, str):
            message_content = " ".join(ref["question"]) + "\n " + ref["instructions"] + "\n " + task_captions
            
        elif isinstance(task_captions, list):
            captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
            random.shuffle(captions)
            captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
            captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
            captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

            message_content = ref["question"][0] + "\n " + \
                captions_to_img_names + \
                ref["question"][1] + "\n " + \
                ref["question"][2] + "\n " + \
                f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
                ref["question"][3] + "\n " + \
                ref["question"][4] + "\n " + \
                ref["instructions"] 

        message_contents.append(message_content)

    def get_res_out(generated_text, message_content):
        if TEST_MODE:
            print(f"generated_text {generated_text}")
        # if "Solution"in generated_text:
        #     generated_text = generated_text[generated_text.index("Solution"):]
        while "Summary:" in generated_text:
            generated_text = generated_text[generated_text.index("Summary:") + 8:]
        model_response = generated_text
        # for char in generated_text:
        #     if char.isdigit():
        #         model_response = char
        #         break
        return model_response

    conversations = [
        [   {"role": "system", "content": ref["prompt"]},
            {"role": "user", "content": message_content}
        ]
        for message_content in message_contents
    ]

    # Convert conversations to model input format and process in batch
    texts = [ref["processor"].apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
    model_inputs = ref["processor"](texts, return_tensors="pt", padding=True).to("cuda:0")
    
    # Generate responses in batch
    generated_ids = ref["model"].generate(
        **model_inputs,
        **ref["generation_kwargs"]
    )

    # Trim the generated output by removing prompt part and decode in batch
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = ref["processor"].batch_decode(generated_ids, skip_special_tokens=True)

    # Extract model response for each generated response
    model_responses = [get_res_out(response, message_content) for response, message_content in zip(responses, message_contents)]

    return model_responses, [model_name] * len(model_responses)


def reason_meta_Llama_3_2(task_captions, model_name):
    # def stack(imgs):
    #     widths, heights = zip(*(i.size for i in imgs))
    #     new_width = max(widths)
    #     new_height = sum(heights)
    #     new_im = Image.new('RGB', (new_width, new_height))
    #     offset = 0
    #     for im in imgs:
    #         x = 0
    #         x = int((new_width - im.size[0])/2)
    #         new_im.paste(im, (x, offset)) # is the upper left corner
    #         offset += im.size[1]
    #     return new_im

    # if isinstance(task_captions, str):
    #     prompt = [
    #         " ".join(ref_storage[model_name]["question"]),
    #         task_captions,
    #         ref_storage[model_name]["instructions"]
    #         ]
    # elif isinstance(task_captions, list):
    #     captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
    #     random.shuffle(captions)
    #     captions_order_left = [captions.index(i)+1 for i in task_captions[0]]
    #     captions_order_right = [captions.index(i)+1 for i in task_captions[1]]

    #     message_content = ref_storage[model_name]["question"][0] + "\n " + \
    #         f"image 1:{captions[0]}\nimage 2:{captions[1]}\nimage 3:{captions[2]}\nimage 4:{captions[3]}" + "\n " + \
    #         f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
    #         ref_storage[model_name]["question"][1] + "\n " + \
    #         ref_storage[model_name]["instructions"] 
    # elif isinstance(task_captions, dict):
    #     if "Story_1" in task_captions or "Story_2" in task_captions:
    #         prompt = [
    #             ref_storage[model_name]["question"][0],
    #             ref_storage[model_name]["instructions"]
    #             ]


    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": task_captions["Story_1"]["image_1"],
    #             },
    #             {
    #                 "type": "image",
    #                 "image": task_captions["Story_1"]["image_2"],
    #             },
    #             {
    #                 "type": "image",
    #                 "image": task_captions["Story_1"]["image_3"],
    #             },
    #             {
    #                 "type": "image",
    #                 "image": task_captions["Story_1"]["image_4"],
    #             },
    #             {"type": "text", "text": " ".join(ref_storage[model_name]["question"]) + ref_storage[model_name]["instructions"]},
    #         ],
    #     }
    # ]
    
    # input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    # inputs = processor(
    #     image,
    #     input_text,
    #     add_special_tokens=False,
    #     return_tensors="pt",
    # ).to(model.device)

    # output = model.generate(**inputs, max_new_tokens=30)
    # print(processor.decode(output[0]))
    raise NotImplementedError

def reason_nvm_Llama3(tasks, model_name, **kwargs):
    if TEST_MODE:
        print(f"reason_nvm_Llama3 is triggered for {model_name} with task_captions: \n{tasks}")
    if "context" in kwargs:
        context = kwargs["context"]

    ref = ref_storage[model_name]

    message_contents = []  # Stores message content for each caption

    for task_captions in tasks:

        if isinstance(task_captions, str):
            message_content = " ".join(ref["question"]) + "\n " + ref["instructions"] + "\n " + task_captions
            
        elif isinstance(task_captions, list):
            captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
            random.shuffle(captions)
            captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
            captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
            captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

            message_content = ref["question"][0] + "\n " + \
                captions_to_img_names + \
                ref["question"][1] + "\n " + \
                ref["question"][2] + "\n " + \
                f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
                ref["question"][3] + "\n " + \
                ref["question"][4] + "\n " + \
                ref["instructions"] 

        message_contents.append(message_content)

    def get_res_out(generated_text, message_content):
        if TEST_MODE:
            print(f"generated_text {generated_text}")
        while "Summary:" in generated_text:
            generated_text = generated_text[generated_text.index("Summary:") + 8:]
        model_response = generated_text
        # for char in model_response:
        #     if char.isdigit():
        #         model_response = char
        #         break
        return model_response

    conversations = [
        [
            {"role": "system", "content": ref["prompt"]},
            {"role": "user", "content": message_content}
        ]
        for message_content in message_contents
    ]

    tokenized_messages = ref_storage[model_name]["processor"].apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True
    ).to("cuda:0")


    all_response_token_ids = ref_storage[model_name]["model"].generate(
        tokenized_messages['input_ids'],
        attention_mask=tokenized_messages['attention_mask'],
        pad_token_id=ref_storage[model_name]["processor"].eos_token_id,
        **ref_storage[model_name]["generation_kwargs"]
    )

    all_generated_tokens = all_response_token_ids[:, len(tokenized_messages['input_ids'][0]):]
    all_generated_texts = ref_storage[model_name]["processor"].batch_decode(all_generated_tokens, skip_special_tokens=True)


    model_responses = [
        get_res_out(generated_text, message_content)
        for generated_text, message_content in zip(all_generated_texts, message_contents)
    ]

    return model_responses, [model_name] * len(model_responses)

def reason_NVLM(task_captions, model_name, **kwargs):
    if "context" in kwargs:
        context = kwargs["context"]

    # generation_config = dict(max_new_tokens=10, do_sample=False)

    if isinstance(task_captions, str):
        message_content = " ".join(ref_storage[model_name]["question"]) + \
            "\n " + ref_storage[model_name]["instructions"] + \
            "\n " + task_captions
            
    elif isinstance(task_captions, list):
        captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
        random.shuffle(captions)
        captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
        captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
        captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

        message_content = ref_storage[model_name]["question"][0] + "\n " + \
            captions_to_img_names + \
            ref_storage[model_name]["question"][1] + "\n " + \
            ref_storage[model_name]["question"][2] + "\n " + \
            f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
            ref_storage[model_name]["question"][3] + "\n " + \
            ref_storage[model_name]["question"][4] + "\n " + \
            ref_storage[model_name]["instructions"] 

    def get_res_out(response, message_content):
        if TEST_MODE:
            print(f"Model response: {response}")
        generated_text = response[0]["generated_text"][-1]
        if generated_text['role']=='assistant':
            model_response = generated_text["content"]
        else:
            for model_response in response[0]["generated_text"]:
                if model_response['role']=='assistant':
                    model_response = model_response["content"]
                    break
        return model_response
    
    try:
        response, history  = ref_storage[model_name]["model"].chat(
            ref_storage[model_name]["processor"], 
            None, 
            message_content, 
            ref_storage[model_name]["generation_kwargs"],
            history=None, 
            return_history=True
            )

        model_response=get_res_out(response, message_content)
    except:
        print(traceback.format_exc())
        print(f"\nRequest response: {response}\n")
        model_response="-1"

    return model_response, model_name

def reason_Qwen_2_VL(tasks, model_name, **kwargs):
    context=""
    if "context" in ref_storage[model_name]:
        if isinstance(ref_storage[model_name]["context"], dict):
            
            context_cnt = ref_storage[model_name]["context"]["size"]
            strategy = "balanced"#ref_storage[model_name]["context"]["strategy"]
            context_dict = get_context(cnt=context_cnt, model_name=model_name, strategy=strategy)
            context+="History:\n"
            cnt=0
            context_list = []
            for context_flip in context_dict["positive_context"]:
                context_list.append(context_flip["task_captions"] + f'\n Solution: {["Left","Right"].index(context_flip["agreed_answer"][0])+1}' + "\nYour solution is CORRECT!\n\n")
                cnt+=1

            for context_flip in context_dict["negative_context"]:
                # just flip answer options
                context_list.append(context_flip["task_captions"] + f'\n Solution: {["Right","Left"].index(context_flip["agreed_answer"][0])+1}' + "\nYour solution is WRONG!\n\n")
                cnt+=1

            random.shuffle(context_list)

            context+= "\n\n".join(context_list)

            context+="\n END of History\nNew Task:\n"


    message_contents = []  # Stores message content for each caption
    ref = ref_storage[model_name]

    for task_captions in tasks:

        if isinstance(task_captions, str):
            message_content = " ".join(ref["question"]) + "\n " + ref["instructions"] + "\n " +context+ task_captions
            
        elif isinstance(task_captions, list):
            captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
            random.shuffle(captions)
            captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
            captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
            captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

            message_content = ref["question"][0] + "\n " + \
                captions_to_img_names + \
                ref["question"][1] + "\n " + \
                ref["question"][2] + "\n " + \
                f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
                ref["question"][3] + "\n " + \
                ref["question"][4] + "\n " + \
                ref["instructions"] 

        message_contents.append(message_content)

    def get_res_out(response, message_content):
        if TEST_MODE:
            print(f"Model response: {response}")
        # while "Solution" in response:
        #     response = response[response.index("Solution") + 8:]
        while "Summary:" in response:
            response = response[response.index("Summary:") + 8:]
        model_response = response
        # for char in response:
        #     if char.isdigit():
        #         model_response = char
        #         break
        return model_response

    # Prepare the conversation input for each message content
    conversations = [
        [
            {"role": "system", "content": ref["prompt"]},
            {"role": "user", "content": message_content}
        ]
        for message_content in message_contents
    ]

    # Convert conversations to model input format and process in batch
    texts = [ref["processor"].apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
    image_inputs, video_inputs = process_vision_info(conversations)
    model_inputs = ref["processor"](
        text=texts, 
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt", 
        padding=True).to("cuda:0")
    
    # Generate responses in batch
    generated_ids = ref["model"].generate(
        **model_inputs,
        **ref["generation_kwargs"]
    )

    # Trim the generated output by removing prompt part and decode in batch
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = ref["processor"].batch_decode(generated_ids, skip_special_tokens=True)

    # Extract model response for each generated response
    model_responses = [get_res_out(response, message_content) for response, message_content in zip(responses, message_contents)]

    return model_responses, [model_name] * len(model_responses)



def reason_Qwen_2_5(tasks, model_name, **kwargs):
    context=""
    if "context" in ref_storage[model_name]:
        if isinstance(ref_storage[model_name]["context"], dict):
            
            context_cnt = ref_storage[model_name]["context"]["size"]
            strategy = "balanced"#ref_storage[model_name]["context"]["strategy"]
            context_dict = get_context(cnt=context_cnt, model_name=model_name, strategy=strategy)
            context+="History:\n"
            cnt=0
            context_list = []
            for context_flip in context_dict["positive_context"]:
                context_list.append(context_flip["task_captions"] + f'\n Solution: {["Left","Right"].index(context_flip["agreed_answer"][0])+1}' + "\nYour solution is CORRECT!\n\n")
                cnt+=1

            for context_flip in context_dict["negative_context"]:
                # just flip answer options
                context_list.append(context_flip["task_captions"] + f'\n Solution: {["Right","Left"].index(context_flip["agreed_answer"][0])+1}' + "\nYour solution is WRONG!\n\n")
                cnt+=1

            random.shuffle(context_list)

            context+= "\n\n".join(context_list)

            context+="\n END of History\nNew Task:\n"


    message_contents = []  # Stores message content for each caption
    ref = ref_storage[model_name]

    for task_captions in tasks:

        if isinstance(task_captions, str):
            message_content = " ".join(ref["question"]) + "\n " + ref["instructions"] + "\n " +context+ task_captions
            
        elif isinstance(task_captions, list):
            captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
            random.shuffle(captions)
            captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
            captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
            captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

            message_content = ref["question"][0] + "\n " + \
                captions_to_img_names + \
                ref["question"][1] + "\n " + \
                ref["question"][2] + "\n " + \
                f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
                ref["question"][3] + "\n " + \
                ref["question"][4] + "\n " + \
                ref["instructions"] 

        message_contents.append(message_content)

    def get_res_out(response, message_content):
        if TEST_MODE:
            print(f"Model response: {response}")
        # while "Solution" in response:
        #     response = response[response.index("Solution") + 8:]
        while "Summary:" in response:
            response = response[response.index("Summary:") + 8:]
        
        model_response = response
        # for char in response:
        #     if char.isdigit():
        #         model_response = char
        #         break
        return model_response

    # Prepare the conversation input for each message content
    conversations = [
        [
            {"role": "system", "content": ref["prompt"]},
            {"role": "user", "content": message_content}
        ]
        for message_content in message_contents
    ]

    # Convert conversations to model input format and process in batch
    texts = [ref["processor"].apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
    model_inputs = ref["processor"](texts, return_tensors="pt", padding=True).to("cuda:0")
    
    # Generate responses in batch
    generated_ids = ref["model"].generate(
        **model_inputs,
        **ref["generation_kwargs"]
    )

    # Trim the generated output by removing prompt part and decode in batch
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = ref["processor"].batch_decode(generated_ids, skip_special_tokens=True)

    # Extract model response for each generated response
    model_responses = [get_res_out(response, message_content) for response, message_content in zip(responses, message_contents)]

    return model_responses, [model_name] * len(model_responses)



#----------------------------------------------------------------------------------------------------------------------------------------------------------

def format_output(model_response=None, model_name=None ):
    try:
        guess_int = int(model_response)-1
        if guess_int < 0:
            guess="Failed due to an Error"
        else:
            guess = ["Left", "Right"][guess_int]
    except:
        print(traceback.format_exc())
        print("\n\nmodel_name", model_name, "\n\nmodel_response: ", model_response)
        # raise Exception
        guess = f"Failed, model_response[-100:] = {model_response[-100:]}"

    return guess

def format_input_as_single_sentence(captions_list_1:list[str], captions_list_2:list[str]):

    cpt_lst1 = [captions_list_1[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_1))]
    cpt_lst2 = [captions_list_2[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_2))]

    List_1 = f"\nimage 1: \"{cpt_lst1[0]}\" \nimage 2: \"{cpt_lst1[1]}\" \nimage 3: \"{cpt_lst1[2]}\" \nimage 4: \"{cpt_lst1[3]}\""
    List_2 = f"\nimage 1: \"{cpt_lst2[0]}\" \nimage 2: \"{cpt_lst2[1]}\" \nimage 3: \"{cpt_lst2[2]}\" \nimage 4: \"{cpt_lst2[3]}\""


    task_captions = f"Story 1: \n{List_1} \n\nStory 2: \n{List_2}" 

    return task_captions

def format_input_with_captions(captions_list_1:list[str], captions_list_2:list[str]):

    cpt_lst1 = [captions_list_1[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_1))]
    cpt_lst2 = [captions_list_2[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_2))]

    return cpt_lst1, cpt_lst2


def format_input_wtih_images(images_list_1:list[str], images_list_2:list[str], model=None):

    imgs_1_0 = Image.open(images_list_1[0]).convert('RGB')
    imgs_1_1 = Image.open(images_list_1[1]).convert('RGB')
    imgs_1_2 = Image.open(images_list_1[2]).convert('RGB')
    imgs_1_3 = Image.open(images_list_1[3]).convert('RGB')

    imgs_2_0 = Image.open(images_list_2[0]).convert('RGB')
    imgs_2_1 = Image.open(images_list_2[1]).convert('RGB')
    imgs_2_2 = Image.open(images_list_2[2]).convert('RGB')
    imgs_2_3 = Image.open(images_list_2[3]).convert('RGB')

    task_captions = {
        "Story_1": {
            "image_1": imgs_1_0,
            "image_1_path": images_list_1[0],
            "image_2": imgs_1_1,
            "image_2_path": images_list_1[1],
            "image_3": imgs_1_2,
            "image_3_path": images_list_1[2], 
            "image_4": imgs_1_3,
            "image_4_path": images_list_1[3]
        },
        "Story_2": {
            "image_1": imgs_2_0,
            "image_1_path": images_list_2[0],
            "image_2": imgs_2_1,
            "image_2_path": images_list_2[1],
            "image_3": imgs_2_2,
            "image_3_path": images_list_2[2],
            "image_4": imgs_2_3,
            "image_4_path": images_list_2[3]
        }
    }
    
    return task_captions

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
        print("creating ",file_path, ' status: ',os.system('touch '+str(file_path)))

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def append_to_json_file(file_path: str, new_data: dict):
    """Append new data to the existing JSON file."""
    existing_data = read_json(file_path)
    for key, value in new_data.items():
        if key in existing_data and isinstance(existing_data[key], dict):
            existing_data[key].update(value)
        else:
            existing_data[key] = value
    write_json(file_path, existing_data)

def parse_args(args):
    global HUGGING_FACE_TOKEN
    global GOOGLE_API_KEY
    global OPENAI_API_KEY
    global TEST_MODE
    global VERBOSE
    global TASK_FORMAT
    global WANDB_API_KEY
    global context_dir
    global BATCH_SIZE

    


    flip_challenges_dir=args.flip_challenges_dir
    reasoning_models=args.reasoning_models
    caption_models = args.caption_models
    caption_file_paths = args.caption_file_paths
    images_dir = args.images_dir
    params_file_path = args.params_file_path
    output_file = args.output_file
    context_dir=args.context_dir

    HUGGING_FACE_TOKEN = args.HUGGING_FACE_TOKEN
    GOOGLE_API_KEY = args.GOOGLE_API_KEY
    OPENAI_API_KEY = args.OPENAI_API_KEY
    WANDB_API_KEY = args.WANDB_API_KEY
    BATCH_SIZE = args.BATCH_SIZE
    if BATCH_SIZE is not None:
        try:
            BATCH_SIZE=int(BATCH_SIZE)
        except:
            BATCH_SIZE=BATCH_SIZE_default
    else:
        BATCH_SIZE = BATCH_SIZE_default

    if WANDB_API_KEY is not None:
        pass
        # global wandb
        # import wandb
        # wandb.login(WANDB_API_KEY)

    TEST_MODE = args.TEST_MODE

    VERBOSE = args.VERBOSE

    TASK_FORMAT = args.TASK_FORMAT

    if flip_challenges_dir is not None:
        try:
            flip_challenges_dir = Path(flip_challenges_dir)
        except:
            print(traceback.format_exc())
            raise Exception
    else:
        raise Exception("flip_challenges_dir is not provided")

    if reasoning_models is None:
        raise Exception("reasoning_models is not provided")
    if reasoning_models.find(',')!=-1:
        reasoning_models = reasoning_models.split(',')
    else:
        reasoning_models = [reasoning_models]
    for i in range(len(reasoning_models)):
        reasoning_models[i]=reasoning_models[i].strip()
    
    if caption_models is None:
        raise Exception("caption_model is not provided")
    if caption_models.find(',')!=-1:
        caption_models = caption_models.split(',')
    else:
        caption_models = [caption_models]
    for i in range(len(caption_models)):
        caption_models[i]=caption_models[i].strip()

    if caption_file_paths is None:
        raise Exception("caption_file_paths is not provided")
    if caption_file_paths.find('::')!=-1:
        caption_file_paths = caption_file_paths.split('::')
    else:
        caption_file_paths = [caption_file_paths]
    for i in range(len(caption_file_paths)):
        caption_file_paths[i]=caption_file_paths[i].strip()

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

    if context_dir is not None:
        try:
            context_dir = Path(context_dir)
        # Fail loudly?
        except:
            print(traceback.format_exc())
            raise Exception
   


    if TEST_MODE is not None:
        try:
            TEST_MODE=int(TEST_MODE)
        except:
            TEST_MODE=1
    else:
        TEST_MODE = 0

    if VERBOSE is not None:
        try:
            VERBOSE=int(VERBOSE)
        except:
            VERBOSE=1
    else:
        VERBOSE = 0

    if TASK_FORMAT is None:
        TASK_FORMAT = "single_sentence"

    return flip_challenges_dir, reasoning_models, caption_models, caption_file_paths, images_dir, params_file_path, output_file
#----------------------------------------------------------------------------------------------------------------------------------------------------------


def process_caption_batched(image_names = None, image_captions=None, caption_models=None) -> Dict[str, str]:
    """Process a batch of image captions based on the question defined in the param file: format input (OPTIONAL), run through LLM, format output (OPTIONAL) and return data entry.
    returns: res =  {
    image_name:{
    
    },
        <reasoning_model_1> : 
            <caption_model_1>: {
                <img_name> : new_image_caption
                <img_name> : new_image_caption
                <img_name> : new_image_caption
            }
            <caption_model_2>: {
                <img_name> : new_image_caption
                <img_name> : new_image_caption
                <img_name> : new_image_caption
            }
        <reasoning_model_2> : 
            <caption_model_1>: {
                <img_name> : new_image_caption
                <img_name> : new_image_caption
                <img_name> : new_image_caption
            }
            <caption_model_2>: {
                <img_name> : new_image_caption
                <img_name> : new_image_caption
                <img_name> : new_image_caption
            }
    }
    """
    
    if TEST_MODE:
        print("processing a batch")


    batch_results = {}
    try:
        # would be more efficient to build batch size around caption model and image names together
        for caption_model in caption_models:
            batch_formatted_tasks = []
            for img_n in image_names:
                batch_formatted_tasks.append(image_captions[img_n][caption_model])
                if img_n not in batch_results:
                    batch_results[img_n] = {}
            for model in ref_storage:
                batched_model_resps, batched_model_name = ref_storage[model]["method"](tasks=batch_formatted_tasks, model_name=model)
            
                if TEST_MODE:
                    print(f"response from the models' call method: {[f'{i}:{j}' for i,j in zip(batched_model_name,batched_model_resps)]}")
                
                for idx, model_resp in enumerate(batched_model_resps):

                    batch_results[image_names[idx]][f"{batched_model_name[idx]}__{caption_model}"] = model_resp
    except:
        raise Exception(f"Error: {traceback.format_exc()}")

    return batch_results



class ImageCaptionsDataset(Dataset):
    def __init__(self, image_names):
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        # img = Image.open(img_path).convert("RGB")  # Open and convert to RGB to ensure consistency
        return image_name#, img  # Return the image and its path

def collate_fn(batch):
    # paths, images  = zip(*batch)  # Unpack the list of tuples into two lists
    # return list(paths), list(images)
    return batch


    

def main():
    
    parser = argparse.ArgumentParser() 

    parser.add_argument('--flip_challenges_dir', help="directory with tasks used in flip challenges")
    parser.add_argument('--reasoning_models', help="reasoning model names as in params_caption files")
    parser.add_argument('--caption_models', help="caption model names as in params_caption files or built-in if supported")
    parser.add_argument('--caption_file_paths', help="refer to job template")
    parser.add_argument('--images_dir', help="directory with images used in flip challenges")

    parser.add_argument('--context_dir', help="directory with context_dir used in/for TRAIN flip challenges")

    parser.add_argument('--params_file_path', help="params_file json full path")
    parser.add_argument('--output_file', help="output_file full path")

    parser.add_argument('--HUGGING_FACE_TOKEN', help="HUGGING_FACE_TOKEN as string")
    parser.add_argument('--GOOGLE_API_KEY', help="GOOGLE_API_KEY as string")
    parser.add_argument('--OPENAI_API_KEY', help="OPENAI_API_KEY as string") 
    parser.add_argument('--WANDB_API_KEY', help="WANDB_API_KEY as string") 

    parser.add_argument('--TEST_MODE', help="TEST_MODE as int")
    parser.add_argument('--VERBOSE', help="TEST_MODE as int")
    parser.add_argument('--TASK_FORMAT',help="TASK_FORMAT string variable can be caption_lists, single_sentence or images_dict")
    parser.add_argument('--BATCH_SIZE', help="BATCH_SIZE")

    flip_challenges_dir, reasoning_models, caption_models, caption_file_paths, images_dir, params_file_path, output_file = parse_args(parser.parse_args())


    # if WANDB_API_KEY:
    #     run = wandb.init(
    #         # Set the project where this run will be logged
    #         project=reasoning_models+"_"+params_file_path.name,
    #         # Track hyperparameters and run metadata
    #         config={
    #             "learning_rate": lr,
    #             "epochs": epochs,
    #         },
    #     )
    solutions_file = output_file
    if solutions_file is None:
        raise Exception(f"solutions_file: {solutions_file}")
    # existing_solutions = read_json(solutions_file)
    existing_solutions = {}
    if "param_file" in existing_solutions and existing_solutions["param_file"] == params_file_path.name:
        solutions = existing_solutions
        print(f"previous solutions file is loaded")
    else:
        solutions = {"param_file" : params_file_path.name}
        print(f"No previous solutions file is used, new one created")
    # prepare output var
    
    for reasoning_model in reasoning_models: 
        if reasoning_model not in solutions:
            solutions[reasoning_model]={} 
        for caption_model in caption_models:
            if caption_model not in solutions[reasoning_model]:
                solutions[reasoning_model][caption_model]={
                            "total_acc":0,
                            "flips_cnt":0,
                            "flips_res":{}
                            }

    # global ref_storage
    if not params_file_path.exists():
        print(params_file_path, params_file_path.exists())
        raise Exception("job with no params file prevented")
    
    # Paths and constants
    load_exp_params(models_list = reasoning_models, params_file=params_file_path)
    if not len(ref_storage):
        raise Exception("No reasoning_models not loaded")
    image_captions=load_image_captions(caption_models=caption_models, caption_file_paths=caption_file_paths)


    batches_of_image_captions = list(image_captions.keys()) # all the images & listed caption model
    # if BATCH_SIZE>0:
    #     if BATCH_SIZE
    if TEST_MODE:
        print("TEST_MODE, sample 2 BATCH_SIZEs (adds on memory stress for activation)")
        batches_of_image_captions=batches_of_image_captions[:2*BATCH_SIZE]


    dataset = ImageCaptionsDataset(batches_of_image_captions)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    for batch_n, batch in enumerate(tqdm(data_loader, desc="Processing captions")):
        # batch_results = {}
        # results = []
        # for img_path_name, image in batch:
        # results.append(process_caption_batched(image_names=batch, image_captions=image_captions, caption_models=caption_models))
        batch_results = process_caption_batched(image_names=batch, image_captions=image_captions, caption_models=caption_models)
        torch.cuda.empty_cache()
        if TEST_MODE:
            print(f"latest batch results:: {batch_results}")
        
        # Combine the batch results into one dictionary
        # for result in results:
        #     if batch_n == 0:
        #     batch_results.update(result)
        
        # Append to the JSON file after each batch
        append_to_json_file(output_file, batch_results)

        

    print("DONE")
    print(f"output_file_path: {solutions_file}")

if __name__ == '__main__':
    main()

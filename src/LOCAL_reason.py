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

BATCH_SIZE = 10
MAX_WORKERS_NUM=4


HUGGING_FACE_TOKEN =None
GOOGLE_API_KEY=None
OPENAI_API_KEY=None
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

            # model_id = "microsoft/Phi-3.5-MoE-instruct"

            # phi_3_5_MoE_tokenizer = AutoTokenizer.from_pretrained(params[model_name]["model_id"]) 
            
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
            # model_id = "microsoft/GRIN-MoE"
            # grin_MoE_model = AutoModelForCausalLM.from_pretrained( 
            # params[model_name]["model_id"],
            # **model_load_params
            # ) 

            # grin_MoE_tokenizer = AutoTokenizer.from_pretrained(params[model_name]["model_id"]) 
            
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
                "processor":AutoTokenizer.from_pretrained(params[model_name]['model_id']),
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
            # model = Qwen2VLForConditionalGeneration.from_pretrained(
            #         params[model_name]["model_id"], 
            #         **model_load_params
            #     )

            # processor = AutoProcessor.from_pretrained(params[model_name]["model_id"])

            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model": Qwen2VLForConditionalGeneration.from_pretrained(
                    params[model_name]["model_id"], 
                    **model_load_params
                ),
                "processor":AutoProcessor.from_pretrained(params[model_name]["model_id"]),
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
                    # "quantization_config":BitsAndBytesConfig(load_in_8bit=True)
                }
            # model_8bit = AutoModelForCausalLM.from_pretrained(
            #     params[model_name]["model_id"],
            #     **model_load_params
            #     )
            
            nvm_Llama3_tokenizer = AutoTokenizer.from_pretrained(params[model_name]["model_id"])
            nvm_Llama3_tokenizer.pad_token_id = nvm_Llama3_tokenizer.eos_token_id

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
                "processor":nvm_Llama3_tokenizer,

                "method": reason_nvm_Llama3,
                "support_image_input": False 
                }

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

            # NVLM_tokenizer = AutoTokenizer.from_pretrained(
            #     params[model_name]["model_id"], 
            #     trust_remote_code=True,
            #     use_fast=False)
            # NVLM_model = AutoModel.from_pretrained(
            #     params[model_name]["model_id"],
            #     **model_load_params
            #     ).eval()

            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model":AutoModel.from_pretrained(
                            params[model_name]["model_id"],
                            **model_load_params
                        ).eval(),
                "processor":AutoTokenizer.from_pretrained(
                                params[model_name]["model_id"], 
                                trust_remote_code=True,
                                use_fast=False),
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
            
            # model = MllamaForConditionalGeneration.from_pretrained(
            #     params[model_name]["model_id"],
            #     **model_load_params
            # )
            # processor = AutoProcessor.from_pretrained(params[model_name]["model_id"])

            ref_storage[model_name] = {
                "model_id":params[model_name]["model_id"],
                "model":MllamaForConditionalGeneration.from_pretrained(
                            params[model_name]["model_id"],
                            **model_load_params
                        ),
                "processor":AutoProcessor.from_pretrained(params[model_name]["model_id"]),
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

            if "quantization_config" in model_load_params:

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
            else:
                ref_storage[model_name] = {
                    "model_id":params[model_name]["model_id"],
                    "model":pipeline(
                            "text-generation",
                            model=params[model_name]["model_id"],
                            model_kwargs={"torch_dtype": model_load_params['torch_dtype']},
                            device_map="auto",
                        ),
                    "processor":None,
                    "prompt":params[model_name]["prompt"],
                    "question":params[model_name]['question'],
                    "instructions": params[model_name]["prompt_instructions"],
                    "generation_kwargs": params[model_name]["gen_kw"],
                    "context": params[model_name]["context_n"],

                    "method": reason_meta_Llama_3_1_pipeline,
                    "support_image_input": False 
                    }
            print(f"{model_name} model LOADED IN")
            print(os.system("nvidia-smi"))

    return 

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
        if "Solution"in generated_text:
            generated_text = generated_text[generated_text.index("Solution"):]
        if TEST_MODE:
            print(f"generated_text {generated_text}")
        model_response = generated_text
        for char in model_response:
            if char.isdigit():
                model_response = char
                break
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
        if "Solution"in generated_text:
            generated_text = generated_text[generated_text.index("Solution"):]
        if TEST_MODE:
            print(f"generated_text {generated_text}")
        model_response = generated_text
        for char in model_response:
            if char.isdigit():
                model_response = char
                break
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


def reason_meta_Llama_3_1_pipeline(task_captions, model_name):
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

    messages = [
        {"role": "system", "content": ref_storage[model_name]["prompt"]},
        {"role": "user", "content": message_content},
    ]

    def get_res_out(generated_text, message_content):
        if TEST_MODE:
            print(f"generated_text {generated_text}")
        if "Solution"in generated_text:
            generated_text = generated_text[generated_text.index("Solution"):]
        model_response = generated_text
        for char in generated_text:
            if char.isdigit():
                model_response = char
                break
        return model_response

    outputs = ref_storage[model_name]["model"](
        messages,
        max_new_tokens=256,
        temperature=0.2
    )
    outs = outputs[0]["generated_text"][-1]
    print(outs['content'])
    model_response=get_res_out(outs['content'], message_content)
    
    return model_response, model_name


# def reason_meta_Llama_3_1(task_captions, model_name):
#     if isinstance(task_captions, str):
#         message_content = ref_storage[model_name]["prompt"].format(" ".join(ref_storage[model_name]["question"])) + \
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

#     input_ids = ref_storage[model_name]["processor"](message_content, return_tensors="pt").to("cuda")

#     output = ref_storage[model_name]["model"].generate(**input_ids, **ref_storage[model_name]["generation_kwargs"])

#     generated_text = ref_storage[model_name]["processor"].decode(output[0], skip_special_tokens=True)
    
#     model_response=get_res_out(generated_text, message_content)
    
#     return model_response, model_name


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



def reason_nvm_Llama3(task_captions, model_name, **kwargs):
    if TEST_MODE:
        print(f"reason_nvm_Llama3 is triggered for {model_name} with task_captions: \n{task_captions}")
    if "context" in kwargs:
        context = kwargs["context"]

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
        while "Solution" in generated_text:
            generated_text = generated_text[generated_text.index("Solution") + 8:]
        model_response = generated_text
        for char in model_response:
            if char.isdigit():
                model_response = char
                break
        return model_response

    conversation = [
        {"role": "system", "content": ref_storage[model_name]["prompt"]},
        {"role": "user", "content": message_content},
    ]

    tokenized_message = ref_storage[model_name]["processor"].apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    response_token_ids = ref_storage[model_name]["model"].generate(
        tokenized_message['input_ids'].cuda(),
        attention_mask=tokenized_message['attention_mask'].cuda(), 
        pad_token_id = ref_storage[model_name]["processor"].eos_token_id,
        **ref_storage[model_name]["generation_kwargs"]
        )
    generated_tokens =response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = ref_storage[model_name]["processor"].batch_decode(generated_tokens, skip_special_tokens=True)[0]
    model_response=get_res_out(generated_text, message_content)
    

    return model_response, model_name

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

def reason_Qwen_2_VL(task_captions, model_name, **kwargs):
    if "context" in kwargs:
        context = kwargs["context"]

    if isinstance(task_captions, str):
        question_text = " ".join(ref_storage[model_name]["question"]) + \
            "\n " + ref_storage[model_name]["instructions"] + \
            "\n " + task_captions
        message_content = [{"type": "text", "text": question_text}]

    elif isinstance(task_captions, list):
        captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
        random.shuffle(captions)
        captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
        captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
        captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

        question_text = ref_storage[model_name]["question"][0] + "\n " + \
            captions_to_img_names + \
            ref_storage[model_name]["question"][1] + "\n " + \
            ref_storage[model_name]["question"][2] + "\n " + \
            f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
            ref_storage[model_name]["question"][3] + "\n " + \
            ref_storage[model_name]["question"][4] + "\n " + \
            ref_storage[model_name]["instructions"] 

        message_content = [{"type": "text", "text": question_text}]

    def get_res_out(response, message_content):
        if TEST_MODE:
            print(f"Model response: {response}")
        while "Solution" in response:
            response = response[response.index("Solution") + 8:]
        model_response = response
        for char in response:
            if char.isdigit():
                model_response = char
                break
        return model_response
    # elif isinstance(task_captions, dict):
    #     image1_path = task_caption["Story_1"]["image1_path"]
    #     image2_path = task_caption["Story_1"]["image2_path"]
    #     image3_path = task_caption["Story_1"]["image3_path"]
    #     image4_path = task_caption["Story_1"]["image4_path"]

    #     question_text = ref_storage[model_name]["prompt"].format(" ".join(ref_storage[model_name]["question"])) + \
    #          + "\n " + ref_storage[model_name]["instructions"] 
    #         # f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \

    #     message_content = [
    #         {
    #             "type": "image",
    #             "image": image1_path,
    #         },
    #         {
    #             "type": "image",
    #             "image": image2_path,
    #         },
    #         {
    #             "type": "image",
    #             "image": image3_path,
    #         },
    #         {
    #             "type": "image",
    #             "image": image4_path,
    #         },
    #         {"type": "text", "text": question_text}
    #         ]
        
    conversation = [
        # {"role": "system", "content": ref_storage[model_name]["prompt"]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": message_content}
            ]
        }
    ]

    text = ref_storage[model_name]["processor"].apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = ref_storage[model_name]["processor"](
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = ref_storage[model_name]["model"].generate(**inputs, **ref_storage[model_name]["generation_kwargs"])
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = ref_storage[model_name]["processor"].batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    model_response = get_res_out(output_text,message_content)

    return model_response, model_name

def reason_Qwen_2_5(task_captions, model_name, **kwargs):
    if "context" in kwargs:
        context = kwargs["context"]

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

            

    def get_res_out(response, message_content):
        if TEST_MODE:
            print(f"Model response: {response}")
        while "Solution" in response:
            response = response[response.index("Solution") + 8:]
        model_response = response
        for char in response:
            if char.isdigit():
                model_response = char
                break
        return model_response

    conversation = [
        {"role": "system", "content": ref_storage[model_name]["prompt"]},
        {
        "role": "user",
        "content": message_content
        },
    ]

    text = ref_storage[model_name]["processor"].apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    model_inputs = ref_storage[model_name]["processor"]([text], return_tensors="pt").to(ref_storage[model_name]["model"].device)
    generated_ids = ref_storage[model_name]["model"].generate(
        **model_inputs,
        **ref_storage[model_name]["generation_kwargs"]
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = ref_storage[model_name]["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]
    model_response=get_res_out(response,message_content)
    

    return model_response, model_name


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

def format_input_wtih_captions(captions_list_1:list[str], captions_list_2:list[str]):

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
        print("creating ",file_path, ' status: ',os.system('touch'+str(file_path)))

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


    flip_challenges_dir=args.flip_challenges_dir
    reasoning_models=args.reasoning_models
    caption_models = args.caption_models
    caption_file_paths = args.caption_file_paths
    images_dir = args.images_dir
    params_file_path = args.params_file_path
    output_file = args.output_file

    HUGGING_FACE_TOKEN = args.HUGGING_FACE_TOKEN
    GOOGLE_API_KEY = args.GOOGLE_API_KEY
    OPENAI_API_KEY = args.OPENAI_API_KEY

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

def process_flip(flip_data=None, image_captions = None, caption_model=None, images_dir=None, input_format='single_sentence') -> Dict[str, str]:
    """Process a single FLIP challenge: format input, run through LLM, format output and return data entry.
    returns: res =  {
        <reasoning_model_1> : 
            <caption_model_1>: {
                <flip_name> : {
                    "acc": <True || False>, 
                    "guess": <'Left' || 'Right'>
                }   
            }
            <caption_model_2>: {
                <flip_name> : {
                    "acc": <True || False>, 
                    "guess": <'Left' || 'Right'>
                }   
            }
        <reasoning_model_2> : 
            <caption_model_1>: {
                <flip_name> : {
                    "acc": <True || False>, 
                    "guess": <'Left' || 'Right'>
                }   
            }
            <caption_model_2>: {
                <flip_name> : {
                    "acc": <True || False>, 
                    "guess": <'Left' || 'Right'>
                }   
            }
    }
    """
    if TEST_MODE:
        print("processing flip:")
        
    res = {} 
    executors_list = []

    if caption_model == "built_in":
        if images_dir is None:
            raise Exception("images_dir not provided")

        # flip_images = {img_name: images_dir.joinpath(img_name.split("/")[-1]+".png") \
        #     for img_name in set(list(flip_data["image_lst1"].values()) + list(flip_data["image_lst2"].values()))}
        
        # image_lst1 = [flip_images[img_name] for img_name in flip_data["image_lst1"].values()]
        # image_lst2 = [flip_images[img_name] for img_name in flip_data["image_lst2"].values()]
        try:
            image_lst1 = [images_dir.joinpath(img_name.split("/")[-1]+".png") for img_name in flip_data["image_lst1"].values()]
        except:
            print(flip_data)
            # print("\n",image_lst1)
            print(traceback.format_exc())
            raise Exception("Exception in the FIRST list of captions")
        try:
            image_lst2 = [images_dir.joinpath(img_name.split("/")[-1]+".png") for img_name in flip_data["image_lst2"].values()]
        except:
            print(flip_data)
            # print("\n",image_lst2)
            print(traceback.format_exc())
            raise Exception("Exception in the SECOND list of captions")
        formatted_task = format_input_wtih_images(images_list_1=image_lst1, images_list_2=image_lst2)
    else:
        flip_image_captions = {img_name: image_captions[img_name.split("/")[-1]+".png"][caption_model] \
            for img_name in set(list(flip_data["image_lst1"].values()) + list(flip_data["image_lst2"].values()))}
        try:
            image_lst1_captions = [flip_image_captions[img_name] for img_name in flip_data["image_lst1"].values()]
        except:
            print(flip_data)
            print("\n",flip_image_captions)
            print(traceback.format_exc())
            raise Exception("Exception in the FIRST list of captions")
        try:
            image_lst2_captions = [flip_image_captions[img_name] for img_name in flip_data["image_lst2"].values()]
        except:
            print(flip_data)
            print("\n",flip_image_captions)
            print(traceback.format_exc())
            raise Exception("Exception in the SECOND list of captions")
        if input_format == "single_sentence":
            formatted_task = format_input_as_single_sentence(captions_list_1=image_lst1_captions, captions_list_2=image_lst2_captions)
        elif input_format == "caption_lists":
            formatted_task = format_input_wtih_captions(captions_list_1=image_lst1_captions, captions_list_2=image_lst2_captions)

    
    try:
        for model in ref_storage:

            if caption_model=="built_in" and not ref_storage[model]["support_image_input"]:
                res[model] = {caption_model: "not supported"}
                continue
            res[model] = {caption_model: {}}
            model_resp, model_name = ref_storage[model]["method"](task_captions=formatted_task, model_name=model)
            if TEST_MODE:
                print(f"response from the {model_name} call method: {model_resp}")
            guess = format_output(model_resp, model_name)
            acc = guess==flip_data["agreed_answer"][0]
            res[model_name][caption_model][flip_data["name"]] = {
                "acc":acc,
                "guess":guess
            }
            # with ThreadPoolExecutor(max_workers=MAX_WORKERS_NUM) as executor:
            #     executors_list.append(executor.submit(ref_storage[model]["method"], task_captions=formatted_task, model_name=model))
        # for x in executors_list:
        #     guess = format_output(x.result()[0], x.result()[1])
        #     acc = guess==flip_data["agreed_answer"][0]
        #     res[x.result()[1]][caption_model][flip_data["name"]] = {
        #         "acc":acc,
        #         "guess":guess
        #     }
    except:
        raise Exception(f"Error: {traceback.format_exc()}")

    return res

import os
import json
from torch.utils.data import Dataset, DataLoader

class FlipChallengeDataset(Dataset):
    def __init__(self, flip_challenge_files):
        self.flip_challenge_files = flip_challenge_files

    def __len__(self):
        return len(self.flip_challenge_files)

    def __getitem__(self, idx):
        flip_file = self.flip_challenge_files[idx]
        flip_data = read_json(flip_file)
        # with open(flip_file, 'r') as f:
        #     flip_data = json.load(f)
        return flip_data  # Return a dictionary with flip data

def identity_collate_fn(batch):
    # Since batch_size=1, batch will be a list with a single item (your dictionary).
    # We return the item directly to avoid any unintended formatting.
    return batch[0]


    

def main():
    
    parser = argparse.ArgumentParser() 

    parser.add_argument('--flip_challenges_dir', help="directory with tasks used in flip challenges")
    parser.add_argument('--reasoning_models', help="reasoning model names as in params_caption files")
    parser.add_argument('--caption_models', help="caption model names as in params_caption files or built-in if supported")
    parser.add_argument('--caption_file_paths', help="refer to job template")
    parser.add_argument('--images_dir', help="directory with images used in flip challenges")
    parser.add_argument('--params_file_path', help="params_file json full path")
    parser.add_argument('--output_file', help="output_file full path")

    parser.add_argument('--HUGGING_FACE_TOKEN', help="HUGGING_FACE_TOKEN as string")
    parser.add_argument('--GOOGLE_API_KEY', help="GOOGLE_API_KEY as string")
    parser.add_argument('--OPENAI_API_KEY', help="OPENAI_API_KEY as string") 
    parser.add_argument('--TEST_MODE', help="TEST_MODE as int")
    parser.add_argument('--VERBOSE', help="TEST_MODE as int")
    parser.add_argument('--TASK_FORMAT',help="TASK_FORMAT string variable can be caption_lists, single_sentence or images_dict")

    flip_challenges_dir, reasoning_models, caption_models, caption_file_paths, images_dir, params_file_path, output_file = parse_args(parser.parse_args())

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

    # image_paths_names = image_paths_names[:4]
    # images_dir = data_dir.joinpath("short_flips_set/images_of_valid_flips") # .joinpath(flip_split+"_images")
    # flip_challenge_files is a list of JSON files
    if [flip_challenges_dir.joinpath(challenge) for challenge in os.listdir(flip_challenges_dir)[0]][0].is_dir():
        flip_challenge_files = [flip_challenges_dir.joinpath(challenge, "valid_flip_"+challenge+".json") for challenge in os.listdir(flip_challenges_dir)]
    else:
        flip_challenge_files = [flip_challenges_dir.joinpath(challenge) for challenge in os.listdir(flip_challenges_dir)]
    
    if TEST_MODE:
        print("TEST_MODE")
        flip_challenge_files=flip_challenge_files[:3]

    flip_data_loader = DataLoader(FlipChallengeDataset(flip_challenge_files), batch_size=1, shuffle=True, collate_fn=identity_collate_fn)
    

    # here one can try to implement batching 
    # for flip_data in data_loader:
    # for cnt, flip_file in tqdm(enumerate(flip_challenge_files), total=len(flip_challenge_files), desc=f"over flips"):
    #     flip_data = read_json(flip_file)

    for cnt, flip_data in tqdm(enumerate(flip_data_loader), total=len(flip_data_loader), desc=f"over flips"):
        
        if TEST_MODE:
            print(f"flip_data: {flip_data}")    

        assert flip_data["agreed_answer"][0] in ["Left", "Right"], f"agreed_answer of {flip_data['name']} is : {flip_data['agreed_answer']}"
        for caption_model in caption_models:   
            # Do not overwrite existing solutions
            if False :#flip_data["name"] in solutions[reasoning_model][caption_model]["flips_res"]:
                continue
            else:
                new_res = process_flip(flip_data=flip_data, image_captions=image_captions, caption_model=caption_model, images_dir=images_dir)
            
                for reasoning_model in new_res:
                    for caption_model in new_res[reasoning_model]:
                        solutions[reasoning_model][caption_model]["flips_res"].update(new_res[reasoning_model][caption_model])
                        solutions[reasoning_model][caption_model]["total_acc"] += new_res[reasoning_model][caption_model][flip_data["name"]]["acc"]
                        solutions[reasoning_model][caption_model]["flips_cnt"] += 1

        if TEST_MODE:
            print(f"solutions: {solutions}")
            
            if (cnt%1 == 0) or( cnt == len(flip_challenge_files)-1):
                append_to_json_file(solutions_file, solutions)

        # save every 10 iterations or so
        else:
            if (cnt%10 == 0) or( cnt == len(flip_challenge_files)-1):
                append_to_json_file(solutions_file, solutions)
            if VERBOSE:
                if cnt%(len(flip_challenge_files)//VERBOSE)==0:
                    res_snip = {}
                    for reasoning_model in new_res:
                        res_snip[reasoning_model] = {}
                        for caption_model in new_res[reasoning_model]:
                            res_snip[reasoning_model][caption_model] = {
                                "curent total_acc": solutions[reasoning_model][caption_model]["total_acc"]/solutions[reasoning_model][caption_model]["flips_cnt"],
                                "latest_caption res": new_res[reasoning_model][caption_model]
                            }
                    print(f"some latest results:: {res_snip}")

        torch.cuda.empty_cache()


    print("DONE")
    print(f"output_file_path: {solutions_file}")

if __name__ == '__main__':
    main()

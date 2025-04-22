import warnings
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
from transformers import pipeline, Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, \
    AutoModelForCausalLM, Pix2StructForConditionalGeneration, VipLlavaForConditionalGeneration,\
         LlavaNextForConditionalGeneration, InstructBlipForConditionalGeneration,\
             InstructBlipProcessor, MllamaForConditionalGeneration

# Ignore all warnings
warnings.filterwarnings("ignore")
import logging

# Suppress warnings at the logging level
logging.getLogger("transformers").setLevel(logging.ERROR)


BATCH_SIZE = 20
MAX_WORKERS_NUM=4

ref_storage = {}
image_captions = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_exp_params(models_list = ['ViPLlava'], params_file = None):
    global ref_storage
    
    if params_file is not None:
        params=read_json(params_file)
        image_captions_file_name = params["image_caption_file"]

    else:
        image_captions_file_name = "image_captions_no_params.json"
        params = {
            "ViPLlava":{
                "model_id":"llava-hf/vip-llava-7b-hf",
                "prompt" : "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{}###Assistant:",
                "question" : "Can you please describe this image?"
            },
            "LlavaNeXT":{
                "model_id":"llava-hf/llava-v1.6-mistral-7b-hf",
                "prompt" : "{}",
                "question" : "Can you please describe this image?"
            }
        }
    # ~ 16 at bfloat16 precision
    if "ViPLlava" in models_list:
        VipLlava_model = VipLlavaForConditionalGeneration.from_pretrained(params["ViPLlava"]["model_id"],  device_map="auto", torch_dtype=torch.bfloat16)
        VipLlava_processor = AutoProcessor.from_pretrained(params["ViPLlava"]["model_id"])
        VipLlava_prompt = params["ViPLlava"]["prompt"]
        VipLlava_question = params["ViPLlava"]["question"]
        if "multi_q" in params["ViPLlava"]:
            VipLlava_question+= " ".join(params["ViPLlava"]["multi_q"])
        if "gen_kw" in params["ViPLlava"]:
            gen_kw = params["ViPLlava"]["gen_kw"]
        else:
            gen_kw = {}
        VipLlava_prompt = VipLlava_prompt.format(VipLlava_question)
        ref_storage["ViPLlava"] = {
        "prompt":VipLlava_prompt,
        "question":VipLlava_question,
        "model":VipLlava_model,
        "processor":VipLlava_processor,
        "generation_kwargs": gen_kw,

        "method":caption_VipLlava
        }
        print("ViPLlava model LOADED IN")
        print(os.system("nvidia-smi"))

    # ~ 14 at bfloat16 precision
    if "LlavaNeXT" in models_list:
        LlavaNeXT_model = LlavaNextForConditionalGeneration.from_pretrained(params["LlavaNeXT"]["model_id"], device_map="auto", torch_dtype=torch.bfloat16)
        LlavaNeXT_processor = AutoProcessor.from_pretrained(params["LlavaNeXT"]["model_id"])
        LlavaNeXT_prompt = params["LlavaNeXT"]["prompt"]
        LlavaNeXT_question = params["LlavaNeXT"]["question"]
        if "multi_q" in params["LlavaNeXT"]:
            LlavaNeXT_question+= " ".join(params["LlavaNeXT"]["multi_q"])
        if "gen_kw" in params["LlavaNeXT"]:
            gen_kw = params["LlavaNeXT"]["gen_kw"]
        else:
            gen_kw = {}
        LlavaNeXT_prompt = LlavaNeXT_prompt.format(LlavaNeXT_question)
        ref_storage["LlavaNeXT"] = {
        "prompt":LlavaNeXT_prompt,
        "question":LlavaNeXT_question,
        "model":LlavaNeXT_model,
        "processor":LlavaNeXT_processor,
        "generation_kwargs": gen_kw,

        "method":caption_LlavaNext
        }
        print("LlavaNeXT model LOADED IN")
        print(os.system("nvidia-smi"))
    # DOESNOT WORK ON MULTI-GPUS
    # if "BLIP2" in models_list:
    #     BLIP2_model = InstructBlipForConditionalGeneration.from_pretrained(params["BLIP2"]["model_id"], device_map="auto", torch_dtype=torch.float16)
    #     BLIP2_processor = InstructBlipProcessor.from_pretrained(params["BLIP2"]["model_id"])
    #     BLIP2_prompt = params["BLIP2"]["prompt"]
    #     BLIP2_question = params["BLIP2"]["question"]
    #     if "multi_q" in params["BLIP2"]:
    #         BLIP2_question+= " ".join(params["BLIP2"]["multi_q"])
    #     BLIP2_prompt = BLIP2_prompt.format(BLIP2_question)
    #     ref_storage["BLIP2"] = {
    #     "prompt":BLIP2_prompt,
    #     "question":BLIP2_question,
    #     "model":BLIP2_model,
    #     "processor":BLIP2_processor,

    #     "method":caption_blip2_prompt
    #     }
    #     print("BLIP2 model LOADED IN")
    #     print(os.system("nvidia-smi"))
    
    # Cannot access without hf login
    if "Llama_3_2" in models_list:
        Llama_3_2_model = MllamaForConditionalGeneration.from_pretrained(params["Llama_3_2"]["model_id"], device_map="auto", torch_dtype=torch.bfloat16)
        Llama_3_2_processor = AutoProcessor.from_pretrained(params["Llama_3_2"]["model_id"])
        Llama_3_2_prompt = params["Llama_3_2"]["prompt"]
        Llama_3_2_question = params["Llama_3_2"]["question"]
        if "multi_q" in params["Llama_3_2"]:
            Llama_3_2_question+= " ".join(params["Llama_3_2"]["multi_q"])
        if "gen_kw" in params["Llama_3_2"]:
            gen_kw = params["Llama_3_2"]["gen_kw"]
        else:
            gen_kw = {}
        Llama_3_2_prompt = Llama_3_2_prompt.format(Llama_3_2_question)
        ref_storage["Llama_3_2"] = {
        "prompt":Llama_3_2_prompt,
        "question":Llama_3_2_question,
        "model":Llama_3_2_model,
        "processor":Llama_3_2_processor,
        "generation_kwargs": gen_kw,

        "method":caption_Llama_3_2_prompt
        }
        print("Llama_3_2 model LOADED IN")
        print(os.system("nvidia-smi"))

    # ~10GB at full precision
    if "Phi_3_5" in models_list:

        Phi_3_5_model = AutoModelForCausalLM.from_pretrained(params["Phi_3_5"]["model_id"], device_map="auto", _attn_implementation='eager', trust_remote_code=True, torch_dtype=torch.bfloat16)
        Phi_3_5_processor = AutoProcessor.from_pretrained(params["Phi_3_5"]["model_id"], num_crops=16, trust_remote_code=True)
        Phi_3_5_prompt = params["Phi_3_5"]["prompt"]
        Phi_3_5_question = params["Phi_3_5"]["question"]
        if "multi_q" in params["Phi_3_5"]:
            Phi_3_5_question+= " ".join(params["Phi_3_5"]["multi_q"])
        if "gen_kw" in params["Llama_3_2"]:
            gen_kw = params["Llama_3_2"]["gen_kw"]
        else:
            gen_kw = {}
        Phi_3_5_prompt = Phi_3_5_prompt.format(Phi_3_5_question)
        ref_storage["Phi_3_5"] = {
        "prompt":Phi_3_5_prompt,
        "question":Phi_3_5_question,
        "model":Phi_3_5_model,
        "processor":Phi_3_5_processor,
        "generation_kwargs": gen_kw,

        "method":caption_Phi_3_5_prompt
        }
        print("Phi_3_5 model LOADED IN")
        print(os.system("nvidia-smi"))
    return image_captions_file_name


# gpt2_image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=device)

# Salesforce/blip2-opt-6.7b ~ 30GB
# llava-hf/llava-v1.6-mistral-7b-hf ~ 15GB
# llava-hf/vip-llava-7b-hf ~ 15GB
# VipLlava_model 7b ~ 15GB (+ ~2Gb per GPU for cuda)
# VipLlava_model 13b ~ 30GB (+ ~2Gb per GPU for cuda)


# blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
# blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto")




# https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llava_next.md

#
# LlavaNeXT_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", device_map="auto")
# LlavaNeXT_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# GIT_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco").to(device)
# GIT_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device)

# Pix2Struct_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-large").to(device)
# Pix2Struct_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-large").to(device)



# def caption_gpt2(img: Image.Image) -> str:
#     img = img.convert("RGB")
#     res = gpt2_image_to_text(img)[0]["generated_text"]
#     return res

# def caption_blip2_no_prompt(img: Image.Image) -> str:
#     inputs = blip2_processor(images=img, return_tensors="pt").to(device)
#     generated_ids = blip2_model.generate(**inputs, #     generated_text = blip2_processor.batch_decode(generated_ids[0], skip_special_tokens=True)[0].strip()
#     return generated_text, "Blip2_uncond"



# def caption_Pix2Struct_uncond(img: Image.Image)-> str:
#     inputs = Pix2Struct_processor(images=img, return_tensors="pt").to(device)
#     # autoregressive generation
#     generated_ids = Pix2Struct_model.generate(**inputs, #     generated_text = Pix2Struct_processor.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
#     return generated_text, "Pix2Struct_uncond"

# def caption_Pix2Struct_cond(img: Image.Image)-> str:
#     text = "This picture shows: "
#     inputs = Pix2Struct_processor(text=text, images=image, return_tensors="pt", add_special_tokens=False).to(device)
#     generated_ids = Pix2Struct_model.generate(**inputs, #     generated_text = Pix2Struct_processor.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
#     return generated_text, "Pix2Struct"

# def caption_GIT(img: Image.Image) -> str:
#     pixel_values = GIT_processor(images=img, return_tensors="pt").pixel_values.to(device)
#     generated_ids = GIT_model.generate(pixel_values=pixel_values, #     generated_caption = GIT_processor.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
#     return generated_caption, "GIT"

def caption_VipLlava(img: Image.Image):
    
    # VipLlava_prompt = VipLlava_prompt.format(VipLlava_question)
    inputs = ref_storage["ViPLlava"]["processor"](text=ref_storage["ViPLlava"]["prompt"], images=img, return_tensors="pt").to(device)
    # Generate
    generate_ids = ref_storage["ViPLlava"]["model"].generate(**inputs, **ref_storage["ViPLlava"]["generation_kwargs"])
    generated_text = ref_storage["ViPLlava"]["processor"].decode(generate_ids[0], skip_special_tokens=True)
    generated_text = generated_text[generated_text.index("###Assistant:")+13:]
    return generated_text, "VipLlava"

def caption_LlavaNext(img: Image.Image):
    image = img.convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": ref_storage["LlavaNeXT"]["prompt"]},
            ],
        },
    ]
    prompt = ref_storage["LlavaNeXT"]["processor"].apply_chat_template(conversation, add_generation_prompt=True)
    inputs = ref_storage["LlavaNeXT"]["processor"](prompt, image, return_tensors="pt").to(device)
    
    output = ref_storage["LlavaNeXT"]["model"].generate(**inputs, **ref_storage["LlavaNeXT"]["generation_kwargs"])
    out_1 = ref_storage["LlavaNeXT"]["processor"].decode(output[0], skip_special_tokens=True)
    res = out_1[out_1.index("/INST]")+6:]
    # print(out_1[out_1.index("/INST]")+6:])
    return res, "LlavaNeXT"

# def caption_blip2_prompt(img: Image.Image) -> str:
#     # prompt = "Question: Can you please describe this image to a blind person? Answer:"
#     prompt = ref_storage["BLIP2"]["prompt"]
#     inputs = ref_storage["BLIP2"]["processor"](images=img, text=prompt, return_tensors="pt").to(device)
#     generated_ids = ref_storage["BLIP2"]["model"].generate(**inputs, #     generated_text = ref_storage["BLIP2"]["processor"].batch_decode(generated_ids[0], skip_special_tokens=True)[0].strip()
#     return generated_text, "BLIP2"

def caption_Llama_3_2_prompt(img: Image.Image) -> str:
    # prompt = "Question: Can you please describe this image to a blind person? Answer:"
    prompt = ref_storage["Llama_3_2"]["prompt"]
    inputs = ref_storage["Llama_3_2"]["processor"](images=img, text=prompt, return_tensors="pt").to(device)
    generated_ids = ref_storage["Llama_3_2"]["model"].generate(**inputs, **ref_storage["Llama_3_2"]["generation_kwargs"])
    generated_text = ref_storage["Llama_3_2"]["processor"].batch_decode(generated_ids[0], skip_special_tokens=True)[0].strip()
    return generated_text, "Llama_3_2"


def caption_Phi_3_5_prompt(img: Image.Image) -> str: 
    messages = [
    {"role": "user", "content": "<|image_1|>\n" + ref_storage["Phi_3_5"]["prompt"]},
    ]
    prompt = ref_storage["Phi_3_5"]["processor"].tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )
    inputs = ref_storage["Phi_3_5"]["processor"](images=[img], text=prompt, return_tensors="pt").to(device)
    generated_ids = ref_storage["Phi_3_5"]["model"].generate(**inputs, eos_token_id=ref_storage["Phi_3_5"]["processor"].tokenizer.eos_token_id, **ref_storage["Phi_3_5"]["generation_kwargs"])
    generated_text = ref_storage["Phi_3_5"]["processor"].batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
    return generated_text, "Phi_3_5"

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
    """Append new data to the existing JSON file."""
    existing_data = read_json(file_path)
    global ref_storage
    for key, value in new_data.items():
        if key in existing_data:
            existing_data[key].update(value)
        else:
            existing_data[key] = value
    write_json(file_path, existing_data)

def process_image(img_path: str = None, img_name:str = None) -> Dict[str, str]:
    """Process a single image: open, generate caption, and return data entry."""

    if img_name is None:
        img_path = Path(img_path)
        img_name = img_path.name
    if img_path is None:
        img_path = "images/"+img_name

    img = Image.open(img_path)
    executors_list = []
    res = {}
    img_seen = False
    try:
        for model in ref_storage:
            # if img_name in image_captions:
            #     if model in image_captions[img_name]:
            #         img_seen = True
            #         continue
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_NUM) as executor:
                executors_list.append(executor.submit(ref_storage[model]["method"], img))
        for x in executors_list:
            res[x.result()[1]] = x.result()[0]
    except:
        print(os.system("nvidia-smi"))
        print(traceback.format_exc())
        raise Exception
    if not len(res.keys()) and img_seen:
        res = image_captions[img_name]

    return {
        img_name : res
        }
    

def main():
    def_dir = "/scratch/kturlan/Flip"

    def_params_file_name = "params_0.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', help="work_dir")
    parser.add_argument('--data_dir', help="data_dir")
    parser.add_argument('--output_file', help="output_file must be equal to the one from params_file")
    parser.add_argument('--params_file', help="params_file json path")
    args = parser.parse_args()
    print(f"Dict format: {vars(args)}")
    work_dir = args.work_dir
    data_dir = args.data_dir
    params_file_path = args.params_file
    output_file = args.output_file

    if work_dir is not None:
        try:
            work_dir = Path(work_dir)
        except:
            work_dir = Path(def_dir)
    else:
        work_dir = Path(def_dir)

    if data_dir is not None:
        try:
            data_dir = Path(data_dir)
        except:
            data_dir = Path(def_dir)
    else:
        data_dir = Path(def_dir)
    
    if params_file_path is not None:
        try:
            params_file_path = Path(params_file_path)
        except:
            params_file_path = data_dir.joinpath(def_params_file_name)
    else:
        params_file_path = data_dir.joinpath(def_params_file_name)

    if output_file is not None:
        try:
            output_file = Path(output_file)
        except:
            output_file = None
    else:
        output_file = None

    


    global ref_storage
    global image_captions
    models_list = ['ViPLlava', 'LlavaNeXT']
    if not params_file_path.exists():
        # params_file = None
        print(params_file_path, params_file_path.exists())
        raise Exception("job with no params file prevented")
    
    image_captions_INPUT_file_name = load_exp_params(models_list = models_list, params_file=params_file_path)

    # Paths and constants
    image_captions_file = data_dir.joinpath(image_captions_INPUT_file_name)  # JSON file path
    image_captions = read_json(image_captions_file)
    for model in ref_storage:
        if model not in image_captions:
            image_captions[model]= {model: ref_storage[model]}

    image_dir = data_dir.joinpath("short_images_of_valid_flips")
    image_paths_names = [image_dir.joinpath(img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.png')]
    # image_paths_names = image_paths_names[:4]
    # image_dir = this_dir.joinpath("short_flips_set_val_flips","bafkreia5yh3tmzzdxigbtnocbllmh7upevpa5lt2e73rzdkt7awmqsqo4a")
    # image_paths_names = [image_dir.joinpath(img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.png')]

    # Batch processing with ThreadPoolExecutor
    for batch_n in tqdm(range(len(image_paths_names) // BATCH_SIZE + 1)):
        batch = image_paths_names[(batch_n) * BATCH_SIZE : (batch_n + 1) * BATCH_SIZE]
        results=[]
        for img_path_name in batch:
            results.append(process_image(img_path=img_path_name))
        # with ThreadPoolExecutor(max_workers=MAX_WORKERS_NUM) as executor:
        #     results = list(executor.map(process_image, batch))
        
        # Combine the batch results into one dictionary
        batch_results = {}
        for result in results:
            batch_results.update(result)
        
        # Append to the JSON file after each batch
        if output_file is None:
            output_file = image_captions_file
        append_to_json_file(output_file, batch_results)

    print("DONE")

if __name__ == '__main__':
    main()


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
from torchvision import transforms

import numpy as np

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import logging

# Suppress warnings at the logging level
logging.getLogger("transformers").setLevel(logging.ERROR)


BATCH_SIZE = 512
MAX_WORKERS_NUM=4

HUGGING_FACE_TOKEN =None
GOOGLE_API_KEY=None
OPENAI_API_KEY=None
TEST_MODE=0

ref_storage = {}
# !!! can be memory intense !!!
image_captions = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_exp_params(models_list = ['resnet_50'], params_file = None):
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
        
        if "resnet" in model_name:
            from transformers import AutoModelForCausalLM, AutoProcessor

            if "model_load_params" in params[model_name]:
                model_load_params = get_model_load_params(params[model_name]["model_load_params"])
            else:
                # by default use Resnet_50
                model_load_params = {
                    "repo":"pytorch/vision:v0.10.0",
                    "model_id":"resnet50",
                    "pretrained":True,
                    "processor":{
                        "Resize":256,
                        "CenterCrop":224,
                        "Normalize":{
                            "mean":[0.485, 0.456, 0.406],
                            "std":[0.229, 0.224, 0.225]
                        }
                    }
                }
            
            ref_storage[model_name] = {
                "model":torch.hub.load(repo_or_dir=model_load_params["repo"], model=params[model_name]["model_id"], pretrained=model_load_params["pretrained"]),
                    
                "processor":transforms.Compose([
                    transforms.Resize(model_load_params["processor"]["Resize"]),
                    transforms.CenterCrop(model_load_params["processor"]["CenterCrop"]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=model_load_params["processor"]["Normalize"]["mean"], std=model_load_params["processor"]["Normalize"]["std"]),
                ]),

                "method":resnet_embeddings
                }
            ref_storage[model_name]["model"] = torch.nn.Sequential(*(list(ref_storage[model_name]["model"].children())[:-1]))  # Remove the classification layer

            if torch.cuda.is_available():
                ref_storage[model_name]["model"].to('cuda')

            ref_storage[model_name]["model"].eval()

            print(f"{model_name} LOADED IN")
            print(os.system("nvidia-smi"))
    return 



##---------------------------------------------------------------------------------------------------------------------------------


def resnet_embeddings(model_name:str = "resnet50", images:List[Image.Image]=[]):# model_name=model_name, images=imgs
    input_tensors = []
    for image in images:
        try:
            input_tensor = ref_storage[model_name]["processor"](image).unsqueeze(0)  # Add batch dimension
            input_tensors.append(input_tensor)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            raise

    if not input_tensors:
        raise Exception(f"Unexpected State for:\n\t model_name={model_name}\n\t len(images): {len(images)}", )

    input_batch = torch.cat(input_tensors)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        output = ref_storage[model_name]['model'](input_batch)
        embeddings = output.squeeze().cpu().numpy()
    
    return embeddings, [model_name]*len(images)

##---------------------------------------------------------------------------------------------------------------------------------

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

    embedding_models = args.embedding_models
    images_dir = args.images_dir
    params_file_path = args.params_file_path
    output_file = args.output_file

    HUGGING_FACE_TOKEN = args.HUGGING_FACE_TOKEN
    GOOGLE_API_KEY = args.GOOGLE_API_KEY
    OPENAI_API_KEY = args.OPENAI_API_KEY

    TEST_MODE = args.TEST_MODE
    
    if embedding_models is None:
        raise Exception("caption_model is not provided")
    if embedding_models.find(',')!=-1:
        embedding_models = embedding_models.split(',')
    else:
        embedding_models = [embedding_models]
    for i in range(len(embedding_models)):
        embedding_models[i]=embedding_models[i].strip()

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

    return embedding_models, images_dir, params_file_path, output_file



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


def process_images_batched(img_paths: List[Path] = None) -> Dict[str, Dict[str, str]]:
    """Process a batch of images: open, generate captions, and return data entries for each image."""

    # Initialize the results dictionary
    batch_results = {}

    imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]

    try:
        # For each model, perform batched inference
        for model_name in ref_storage:
            
            # Call the batched captioning method for the current model
            embeddings, model_names = ref_storage[model_name]["method"](model_name=model_name, images=imgs)

            for i, img_path in enumerate(img_paths):
                batch_results[img_path.name] = embeddings[i].tolist()

    except Exception as e:
        print("An error occurred:", e)
        print(traceback.format_exc())
        raise

    return batch_results
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_models', help="e.g. resnet50")
    parser.add_argument('--images_dir', help="data_dir")
    parser.add_argument('--params_file_path', help="output_file should (must) be equal to the one from params_file")
    parser.add_argument('--output_file', help="caption_model name as in params_caption files or built-in if supported")

    parser.add_argument('--HUGGING_FACE_TOKEN', help="HUGGING_FACE_TOKEN as string")
    parser.add_argument('--GOOGLE_API_KEY', help="GOOGLE_API_KEY as string")
    parser.add_argument('--OPENAI_API_KEY', help="OPENAI_API_KEY as string") 

    parser.add_argument('--TEST_MODE', help="TEST_MODE as int")

    embedding_models, images_dir, params_file_path, output_file = parse_args(parser.parse_args())

    existing_solutions = read_json(output_file)

    global ref_storage
    global image_captions
    if not params_file_path.exists():
        print(params_file_path, params_file_path.exists())
        raise Exception("job with no params file prevented")
    
    load_exp_params(models_list = embedding_models, params_file=params_file_path)

    image_paths_names = [images_dir.joinpath(img_name) for img_name in os.listdir(images_dir) if img_name.endswith('.png')]
    if TEST_MODE:
        print("TEST_MODE, sample 5 images")
        image_paths_names=image_paths_names[:5]

    image_paths_names_filtered = []
    for image_paths in image_paths_names:
        img_name = image_paths.name
        if img_name not in existing_solutions:
            image_paths_names_filtered.append(image_paths)


    dataset = ImagePathDataset(image_paths_names_filtered)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    

    for batch_n, batch in enumerate(tqdm(data_loader, desc="Processing images")):
        
        # for img_path_name, image in batch:
        batch_results = process_images_batched(img_paths=batch)
        
        if batch_n == 0:
            print(f"some latest results:: {batch_results}")
        
        # Append to the JSON file after each batch
        append_to_json_file(output_file, batch_results)
        torch.cuda.empty_cache()

        

    print("DONE")

if __name__ == '__main__':
    main()


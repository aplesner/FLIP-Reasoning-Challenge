import base64
from pathlib import Path
import os
import json
import traceback
import time
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import ast
import argparse
import random

import warnings


# BATCH_SIZE = 20
MAX_WORKERS_NUM=4


HUGGING_FACE_TOKEN =None
GOOGLE_API_KEY=None
OPENAI_API_KEY=None
context_dir=None
stable_context=None
TEST_MODE=0
VERBOSE=0
TASK_FORMAT="single_sentence"


ref_storage={}

def stack_images(imgs):
    widths, heights = zip(*(i.size for i in imgs))
    new_width = max(widths)
    new_height = sum(heights)
    new_im = Image.new('RGB', (new_width, new_height))
    offset = 0
    for im in imgs:
        x = 0
        x = int((new_width - im.size[0])/2)
        new_im.paste(im, (x, offset)) # is the upper left corner
        offset += im.size[1]
    return new_im

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

def load_exp_params(models_list = ['Gei'], params_file = None, TEST_MODE=0):
    ref_storage = {}
    
    if TEST_MODE:
        print(f"loading models_list: {models_list}")

    if params_file is not None:
        params=read_json(params_file)
    else:
        raise Exception("no param file provided")
        
    for model_name in models_list:
        if "ChatGPT" in model_name:
            from openai import OpenAI
            client = OpenAI(api_key = OPENAI_API_KEY)
            
            if "model_load_params" in params[model_name]:
                model_load_params = params[model_name]["model_load_params"]
            
            ref_storage[model_name]={
                "model_id":params[model_name]["model_id"],
                "model":client,
                "processor":None ,
                "prompt":params[model_name]["prompt"],
                "question":params[model_name]["question"],
                "instructions":params[model_name]["prompt_instructions"],
                "generation_kwargs": params[model_name]["gen_kw"],
                "context":params[model_name]["context_n"],

                "method": reason_GPT,
                "support_image_input": True
            }

            print(f"{model_name} model LOADED IN")
    return ref_storage


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


def reason_Gemini(task_captions=None, model_name="Gemini_1_5_pro_002",**kwargs):

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


    if isinstance(task_captions, str):
        question_text = " ".join(ref_storage[model_name]["question"]) + \
            "\n " + ref_storage[model_name]["instructions"] + \
            "\n " + context + task_captions

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

    elif isinstance(task_captions, dict):
        
        story1_images = [task_captions["Story_1"][img] for img in ["image_1","image_2","image_3","image_4"]]
        story2_images = [task_captions["Story_2"][img] for img in ["image_1","image_2","image_3","image_4"]]

        question_text = [
            ref_storage[model_name]["question"][0] + "\n ",
            ref_storage[model_name]["question"][1] + "\n ",
            "Story 1:\n",
            story1_images[0], story1_images[1],story1_images[2], story1_images[3],
            " and Story 2:\n",
            story2_images[0], story2_images[1],story2_images[2], story2_images[3],
            ref_storage[model_name]["question"][2] + "\n ",
            ref_storage[model_name]["question"][3] + "\n ",
            ref_storage[model_name]["instructions"] 
        ]
            # ref_storage[model_name]["question"][4] + "\n " + \


    def get_res_out(response, message_content):
        response=response.text.strip()
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

    prompt = question_text 
    response = ref_storage[model_name]["model"].generate_content(
        prompt,
        **ref_storage[model_name]["generation_kwargs"])
    try:
        response = get_res_out(response, prompt)
    except:
        response = -1

    return response, model_name

def reason_GPT(task_captions=None, model_name="ChatGPT_4_unknown",**kwargs):

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


    if isinstance(task_captions, str):
        question_text = " ".join(ref_storage[model_name]["question"]) + \
            "\n " + ref_storage[model_name]["instructions"] + \
            "\n " + context + task_captions

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

    elif isinstance(task_captions, dict):
        
        story1_images = [task_captions["Story_1"][img] for img in ["image_1","image_2","image_3","image_4"]]
        story2_images = [task_captions["Story_2"][img] for img in ["image_1","image_2","image_3","image_4"]]

        question_text = [
            ref_storage[model_name]["question"][0] + "\n ",
            ref_storage[model_name]["question"][1] + "\n ",
            "Story 1:\n",
            story1_images[0], story1_images[1],story1_images[2], story1_images[3],
            " and Story 2:\n",
            story2_images[0], story2_images[1],story2_images[2], story2_images[3],
            ref_storage[model_name]["question"][2] + "\n ",
            ref_storage[model_name]["question"][3] + "\n ",
            ref_storage[model_name]["instructions"] 
        ]

    def get_res_out(response, message_content):
        response_content=response.choices[0].message.content
        if TEST_MODE:
            print(f"Model response: {response}")
        while "Solution" in response_content:
            response_content = response_content[response_content.index("Solution") + 8:]
        model_response = response_content
        for char in response_content:
            if char.isdigit():
                model_response = char
                break
        return model_response

    response = ref_storage[model_name]["model"].chat.completions.create(
        model=ref_storage[model_name]["model_id"],
        messages=[
            {"role": "system", "content": ref_storage[model_name]["prompt"]},
            {"role": "user", "content": question_text}
        ],
        **ref_storage[model_name]["generation_kwargs"]
    )
    # print(f"\n\nGPT RESPONSE: {response}\n\n")
    response = get_res_out(response, question_text)
    # try:
    #     response = get_res_out(response, question_text)
    # except:
    #     response = -1

    return response, model_name



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

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

    return [cpt_lst1, cpt_lst2]

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
    global context_dir


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

        input_format=TASK_FORMAT
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
            if TEST_MODE:
                print(f"task_captions={formatted_task}")
            model_resp, model_name = ref_storage[model]["method"](task_captions=formatted_task, model_name=model)
            if TEST_MODE:
                print(f"response from the {model_name} call method: {model_resp}")
            guess = format_output(model_resp, model_name)
            acc = guess==flip_data["agreed_answer"][0]
            res[model_name][caption_model][flip_data["name"]] = {
                "acc":acc,
                "guess":guess
            }
            
    except:
        raise Exception(f"Error: {traceback.format_exc()}")
    else:   
        return res

    
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
    parser.add_argument('--TEST_MODE', help="TEST_MODE as int")
    parser.add_argument('--VERBOSE', help="TEST_MODE as int")
    parser.add_argument('--TASK_FORMAT',help="TASK_FORMAT string variable can be caption_lists, single_sentence or images_dict")

    flip_challenges_dir, reasoning_models, caption_models, caption_file_paths, images_dir, params_file_path, output_file = parse_args(parser.parse_args())

    solutions_file = output_file
    if solutions_file is None:
        raise Exception(f"solutions_file: {solutions_file}")
    
    if not TEST_MODE:
        # print(f"solutions: {solutions}")
        existing_solutions = read_json(solutions_file)
    else:
        existing_solutions = {}

    # existing_solutions = {}
    if "param_file" in existing_solutions and existing_solutions["param_file"] == params_file_path.name:
        solutions = existing_solutions
        if "stable_context" in solutions:
            global stable_context
            stable_context = solutions["stable_context"]
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

    if flip_challenges_dir.joinpath(os.listdir(flip_challenges_dir)[0]).is_dir():
        flip_challenge_files = [flip_challenges_dir.joinpath(challenge, "valid_flip_"+challenge+".json") for challenge in os.listdir(flip_challenges_dir)]
    else:
        flip_challenge_files = [flip_challenges_dir.joinpath(challenge) for challenge in os.listdir(flip_challenges_dir)]
    
    if TEST_MODE:
        print("TEST_MODE")
        flip_challenge_files=flip_challenge_files[:3]

    # flip_data_loader = DataLoader(FlipChallengeDataset(flip_challenge_files), batch_size=1, shuffle=True, collate_fn=identity_collate_fn)
    
    # for cnt, flip_data in tqdm(enumerate(flip_data_loader), total=len(flip_data_loader), desc=f"over flips"):
    for cnt, flip_file in tqdm(enumerate(flip_challenge_files), total=len(flip_challenge_files), desc=f"over flips"):
        flip_data = read_json(flip_file)
        if TEST_MODE:
            print(f"flip_data: {flip_data}")    

        assert flip_data["agreed_answer"][0] in ["Left", "Right"], f"agreed_answer of {flip_data['name']} is : {flip_data['agreed_answer']}"
        for caption_model in caption_models:   
            # Do not overwrite existing solutions
            if flip_data["name"] in solutions[reasoning_model][caption_model]["flips_res"]:
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
                    append_to_json_file(solutions_file, solutions)
                    # if (cnt%10 == 0) or( cnt == len(flip_challenge_files)-1):
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

                    if stable_context is not None:
                        if "stable_context" not in solutions:
                            solutions["stable_context"] = stable_context
                            append_to_json_file(solutions_file, solutions)

    print("DONE")
    print(f"output_file_path: {solutions_file}")

if __name__ == '__main__':
    main()
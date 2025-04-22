import base64
from pathlib import Path
import os
import json
import traceback
import time
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import argparse
import random
from openai import OpenAI


class FlipChallenger:
    def __init__(self, config):
        self.openai_api_key = config.openai_api_key
        self.test_mode = config.test_mode
        self.verbose = config.verbose
        self.task_format = config.task_format
        self.context_dir = config.context_dir
        self.stable_context = None
        self.model_storage = {}

    def load_exp_params(self, models_list, params_file):
        """Load experiment parameters for models"""
        if self.test_mode:
            print(f"loading models_list: {models_list}")

        if params_file is not None:
            params = self.read_json(params_file)
        else:
            raise Exception("no param file provided")
            
        for model_name in models_list:
            if "ChatGPT" in model_name:
                client = OpenAI(api_key=self.openai_api_key)
                
                model_load_params = {}
                if "model_load_params" in params[model_name]:
                    model_load_params = params[model_name]["model_load_params"]
                
                self.model_storage[model_name] = {
                    "model_id": params[model_name]["model_id"],
                    "model": client,
                    "processor": None,
                    "prompt": params[model_name]["prompt"],
                    "question": params[model_name]["question"],
                    "instructions": params[model_name]["prompt_instructions"],
                    "generation_kwargs": params[model_name]["gen_kw"],
                    "context": params[model_name]["context_n"],
                    "method": self.reason_GPT,
                    "support_image_input": True
                }

                print(f"{model_name} model LOADED IN")
        return self.model_storage

    def get_context(self, cnt=5, model_name="ChatGPT", strategy="balanced"):
        """Get context for prompting"""
        if self.stable_context is not None:
            if self.test_mode:
                print(f"returning stable_context")
            return self.stable_context
        
        context = {}
        if self.context_dir is None:
            if self.test_mode:
                print(f"returning empty context")
            return context

        ref = self.model_storage[model_name]["context"]

        context_path = self.context_dir.joinpath(ref["context_reasoning_model"], ref["context_captioning_model"])
        if not context_path.exists():
            if self.test_mode:
                print(f"returning empty context")
            return context

        if strategy == "balanced":
            negative_samples = self.read_json(context_path.joinpath("negative_samples.json"))
            positive_samples = self.read_json(context_path.joinpath("positive_samples.json"))

            context["positive_context"] = random.sample(positive_samples, cnt)
            context["negative_context"] = random.sample(negative_samples, cnt)

        elif strategy == "positive":
            positive_samples = self.read_json(context_path.joinpath("positive_samples.json"))
            context["positive_context"] = random.sample(positive_samples, cnt)

        elif strategy == "negative":
            negative_samples = self.read_json(context_path.joinpath("negative_samples.json"))
            context["negative_context"] = random.sample(negative_samples, cnt)
        else:
            raise NotImplementedError

        self.stable_context = context
        if self.test_mode:
            print(f"returning stable_context: {self.stable_context}")
        
        return self.stable_context

    def reason_GPT(self, task_captions=None, model_name="ChatGPT_4_unknown", **kwargs):
        """Process task with OpenAI's GPT models"""
        context = ""
        if "context" in self.model_storage[model_name]:
            if isinstance(self.model_storage[model_name]["context"], dict):
                context_cnt = self.model_storage[model_name]["context"]["size"]
                strategy = "balanced"
                context_dict = self.get_context(cnt=context_cnt, model_name=model_name, strategy=strategy)
                context += "History:\n"
                cnt = 0
                context_list = []
                for context_flip in context_dict["positive_context"]:
                    context_list.append(context_flip["task_captions"] + f'\n Solution: {["Left","Right"].index(context_flip["agreed_answer"][0])+1}' + "\nYour solution is CORRECT!\n\n")
                    cnt += 1

                for context_flip in context_dict["negative_context"]:
                    # just flip answer options
                    context_list.append(context_flip["task_captions"] + f'\n Solution: {["Right","Left"].index(context_flip["agreed_answer"][0])+1}' + "\nYour solution is WRONG!\n\n")
                    cnt += 1

                random.shuffle(context_list)
                context += "\n\n".join(context_list)
                context += "\n END of History\nNew Task:\n"

        if isinstance(task_captions, str):
            question_text = " ".join(self.model_storage[model_name]["question"]) + \
                "\n " + self.model_storage[model_name]["instructions"] + \
                "\n " + context + task_captions

        elif isinstance(task_captions, list):
            captions = list(dict.fromkeys(task_captions[0]+task_captions[1]))
            random.shuffle(captions)
            captions_order_left = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[0]]
            captions_order_right = [["A","B","C","D","E","F","G","H"][captions.index(i)] for i in task_captions[1]]
            captions_to_img_names = "\n".join([f'{["A","B","C","D","E","F","G","H"][i]}: {captions[i]}' for i in range(len(captions))])

            question_text = self.model_storage[model_name]["question"][0] + "\n " + \
                captions_to_img_names + \
                self.model_storage[model_name]["question"][1] + "\n " + \
                self.model_storage[model_name]["question"][2] + "\n " + \
                f"Order 1: {captions_order_left} and Order 2: {captions_order_right}" + "\n " + \
                self.model_storage[model_name]["question"][3] + "\n " + \
                self.model_storage[model_name]["question"][4] + "\n " + \
                self.model_storage[model_name]["instructions"] 

        elif isinstance(task_captions, dict):
            story1_images = [task_captions["Story_1"][img] for img in ["image_1","image_2","image_3","image_4"]]
            story2_images = [task_captions["Story_2"][img] for img in ["image_1","image_2","image_3","image_4"]]

            question_text = [
                self.model_storage[model_name]["question"][0] + "\n ",
                self.model_storage[model_name]["question"][1] + "\n ",
                "Story 1:\n",
                story1_images[0], story1_images[1], story1_images[2], story1_images[3],
                " and Story 2:\n",
                story2_images[0], story2_images[1], story2_images[2], story2_images[3],
                self.model_storage[model_name]["question"][2] + "\n ",
                self.model_storage[model_name]["question"][3] + "\n ",
                self.model_storage[model_name]["instructions"] 
            ]

        def get_res_out(response, message_content):
            response_content = response.choices[0].message.content
            if self.test_mode:
                print(f"Model response: {response}")
            while "Solution" in response_content:
                response_content = response_content[response_content.index("Solution") + 8:]
            model_response = response_content
            for char in response_content:
                if char.isdigit():
                    model_response = char
                    break
            return model_response

        
        response = self.model_storage[model_name]["model"].chat.completions.create(
            model=self.model_storage[model_name]["model_id"],
            messages=[
                {"role": "system", "content": self.model_storage[model_name]["prompt"]},
                {"role": "user", "content": question_text}
            ],
            **self.model_storage[model_name]["generation_kwargs"]
        )
        
        try:
            response = get_res_out(response, question_text)
        except:
            response = -1

        return response, model_name

    def load_image_captions(self, caption_models=[], caption_file_paths=[]):
        """Load image captions from files"""
        image_captions = {}
        for caption_model, caption_file_path in zip(caption_models, caption_file_paths):
            if caption_model == 'built_in':
                continue
            these_image_captions = self.read_json(caption_file_path)
            for key, value in these_image_captions.items():
                if key in image_captions:
                    image_captions[key][caption_model] = value[caption_model]
                else:
                    image_captions[key] = {
                        caption_model: value[caption_model]
                    }
            if self.verbose:
                print(f"\nLOADED: \n {caption_model} \n ")
                print(f"\tFROM {caption_file_path} \n ")

        return image_captions

    def format_output(self, model_response=None, model_name=None):
        """Format model output to standardized format"""
        try:
            guess_int = int(model_response) - 1
            if guess_int < 0:
                guess = "Failed due to an Error"
            else:
                guess = ["Left", "Right"][guess_int]
        except:
            if self.test_mode:
                print(traceback.format_exc())
                print("\n\nmodel_name", model_name, "\n\nmodel_response: ", model_response)
            guess = f"Failed, model_response[-100:] = {model_response[-100:]}"

        return guess

    def format_input_as_single_sentence(self, captions_list_1, captions_list_2):
        """Format input as a single sentence"""
        cpt_lst1 = [captions_list_1[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_1))]
        cpt_lst2 = [captions_list_2[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_2))]

        List_1 = f"\nimage 1: \"{cpt_lst1[0]}\" \nimage 2: \"{cpt_lst1[1]}\" \nimage 3: \"{cpt_lst1[2]}\" \nimage 4: \"{cpt_lst1[3]}\""
        List_2 = f"\nimage 1: \"{cpt_lst2[0]}\" \nimage 2: \"{cpt_lst2[1]}\" \nimage 3: \"{cpt_lst2[2]}\" \nimage 4: \"{cpt_lst2[3]}\""

        task_captions = f"Story 1: \n{List_1} \n\nStory 2: \n{List_2}" 

        return task_captions

    def format_input_with_captions(self, captions_list_1, captions_list_2):
        """Format input with captions"""
        cpt_lst1 = [captions_list_1[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_1))]
        cpt_lst2 = [captions_list_2[cnt].replace('\n', '').replace('"', "'") for cnt in range(len(captions_list_2))]

        return [cpt_lst1, cpt_lst2]

    def format_input_with_images(self, images_list_1, images_list_2):
        """Format input with images"""
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

    def process_flip(self, flip_data, image_captions, caption_model, images_dir):
        """Process a single FLIP challenge: format input, run through LLM, format output and return data entry"""
        if self.test_mode:
            print("processing flip:")
            
        res = {} 

        if caption_model == "built_in":
            if images_dir is None:
                raise Exception("images_dir not provided")
            try:
                image_lst1 = [images_dir.joinpath(img_name.split("/")[-1]+".png") for img_name in flip_data["image_lst1"].values()]
            except:
                print(flip_data)
                print(traceback.format_exc())
                raise Exception("Exception in the FIRST list of captions")
            try:
                image_lst2 = [images_dir.joinpath(img_name.split("/")[-1]+".png") for img_name in flip_data["image_lst2"].values()]
            except:
                print(flip_data)
                print(traceback.format_exc())
                raise Exception("Exception in the SECOND list of captions")
            formatted_task = self.format_input_with_images(images_list_1=image_lst1, images_list_2=image_lst2)
        else:
            flip_image_captions = {img_name: image_captions[img_name.split("/")[-1]+".png"][caption_model] \
                for img_name in set(list(flip_data["image_lst1"].values()) + list(flip_data["image_lst2"].values()))}
            try:
                image_lst1_captions = [flip_image_captions[img_name] for img_name in flip_data["image_lst1"].values()]
            except:
                print(flip_data)
                print("\n", flip_image_captions)
                print(traceback.format_exc())
                raise Exception("Exception in the FIRST list of captions")
            try:
                image_lst2_captions = [flip_image_captions[img_name] for img_name in flip_data["image_lst2"].values()]
            except:
                print(flip_data)
                print("\n", flip_image_captions)
                print(traceback.format_exc())
                raise Exception("Exception in the SECOND list of captions")

            input_format = self.task_format
            if input_format == "single_sentence":
                formatted_task = self.format_input_as_single_sentence(captions_list_1=image_lst1_captions, captions_list_2=image_lst2_captions)
            elif input_format == "caption_lists":
                formatted_task = self.format_input_with_captions(captions_list_1=image_lst1_captions, captions_list_2=image_lst2_captions)

        try:
            for model in self.model_storage:
                if caption_model == "built_in" and not self.model_storage[model]["support_image_input"]:
                    res[model] = {caption_model: "not supported"}
                    continue
                res[model] = {caption_model: {}}
                # if self.test_mode:
                #     print(f"task_captions={formatted_task}")
                model_resp, model_name = self.model_storage[model]["method"](task_captions=formatted_task, model_name=model)
                if self.test_mode:
                    print(f"response from the {model_name} call method: {model_resp}")
                guess = self.format_output(model_resp, model_name)
                acc = guess == flip_data["agreed_answer"][0]
                res[model_name][caption_model][flip_data["name"]] = {
                    "acc": acc,
                    "guess": guess
                }
                
        except:
            raise Exception(f"Error: {traceback.format_exc()}")
        else:   
            return res

    def read_json(self, file_path):
        """Read the JSON file and return the data as a dictionary."""
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as file:
                    return json.load(file)
            except:
                return {}
        else:
            return {}

    def write_json(self, file_path, data):
        """Write the updated dictionary to the JSON file."""

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def append_to_json_file(self, file_path, new_data):
        """Append new data to the existing JSON file."""
        existing_data = self.read_json(file_path)
        for key, value in new_data.items():
            if key in existing_data and isinstance(existing_data[key], dict):
                existing_data[key].update(value)
            else:
                existing_data[key] = value
        self.write_json(file_path, existing_data)

    def run(self, flip_challenges_dir, reasoning_models, caption_models, caption_file_paths, 
             images_dir, params_file_path, output_file):
        """Main execution function"""
        solutions_file = output_file
        if solutions_file is None:
            raise Exception(f"solutions_file: {solutions_file}")
        
        if not self.test_mode:
            existing_solutions = self.read_json(solutions_file)
        else:
            existing_solutions = {}

        if "param_file" in existing_solutions and existing_solutions["param_file"] == params_file_path.name:
            solutions = existing_solutions
            if "stable_context" in solutions:
                self.stable_context = solutions["stable_context"]
            print(f"previous solutions file is loaded")
        else:
            solutions = {"param_file": params_file_path.name}
            print(f"No previous solutions file is used, new one created")
        
        # prepare output var
        for reasoning_model in reasoning_models: 
            if reasoning_model not in solutions:
                solutions[reasoning_model] = {} 
            for caption_model in caption_models:
                if caption_model not in solutions[reasoning_model]:
                    solutions[reasoning_model][caption_model] = {
                        "total_acc": 0,
                        "flips_cnt": 0,
                        "flips_res": {}
                    }

        if not params_file_path.exists():
            print(params_file_path, params_file_path.exists())
            print(f"{params_file_path} does not exist")
            raise Exception("job with no params file prevented")
        
        # Load models and captions
        self.load_exp_params(models_list=reasoning_models, params_file=params_file_path)
        if not len(self.model_storage):
            raise Exception("No reasoning_models loaded")
        
        image_captions = self.load_image_captions(caption_models=caption_models, caption_file_paths=caption_file_paths)

        if flip_challenges_dir.joinpath(os.listdir(flip_challenges_dir)[0]).is_dir():
            flip_challenge_files = [flip_challenges_dir.joinpath(challenge, "valid_flip_"+challenge+".json") for challenge in os.listdir(flip_challenges_dir)]
        else:
            flip_challenge_files = [flip_challenges_dir.joinpath(challenge) for challenge in os.listdir(flip_challenges_dir)]
        
        if self.test_mode:
            print("TEST_MODE")
            flip_challenge_files = flip_challenge_files[:3]

        for cnt, flip_file in tqdm(enumerate(flip_challenge_files), total=len(flip_challenge_files), desc=f"over flips"):
            # if cnt > 100:
            #     break

            flip_data = self.read_json(flip_file)
            if self.test_mode:
                print(f"flip_data: {flip_data}")    

            assert flip_data["agreed_answer"][0] in ["Left", "Right"], f"agreed_answer of {flip_data['name']} is : {flip_data['agreed_answer']}"
            
            for caption_model in caption_models:   
                # Do not overwrite existing solutions
                reasoning_model = reasoning_models[0]  # Default to first reasoning model for check
                
                if flip_data["name"] in solutions[reasoning_model][caption_model]["flips_res"]:
                    if self.test_mode:
                        print(f"FLIP {flip_data['name']} already processed")
                    continue
                else:
                    new_res = self.process_flip(
                        flip_data=flip_data, 
                        image_captions=image_captions, 
                        caption_model=caption_model, 
                        images_dir=images_dir
                    )
                
                    for reasoning_model in new_res:
                        for caption_model in new_res[reasoning_model]:
                            solutions[reasoning_model][caption_model]["flips_res"].update(new_res[reasoning_model][caption_model])
                            solutions[reasoning_model][caption_model]["total_acc"] += new_res[reasoning_model][caption_model][flip_data["name"]]["acc"]
                            solutions[reasoning_model][caption_model]["flips_cnt"] += 1

                    if self.test_mode:
                        print(f"solutions: {solutions}")
                        
                        if (cnt%1 == 0) or (cnt == len(flip_challenge_files)-1):
                            self.append_to_json_file(solutions_file, solutions)
                    else:
                        self.append_to_json_file(solutions_file, solutions)
                        
                        if self.verbose:
                            if cnt%(len(flip_challenge_files)//self.verbose) == 0:
                                res_snip = {}
                                for reasoning_model in new_res:
                                    res_snip[reasoning_model] = {}
                                    for caption_model in new_res[reasoning_model]:
                                        res_snip[reasoning_model][caption_model] = {
                                            "curent total_acc": solutions[reasoning_model][caption_model]["total_acc"]/solutions[reasoning_model][caption_model]["flips_cnt"],
                                            "latest_caption res": new_res[reasoning_model][caption_model]
                                        }
                                print(f"some latest results:: {res_snip}")

                        if self.stable_context is not None:
                            if "stable_context" not in solutions:
                                solutions["stable_context"] = self.stable_context
                                self.append_to_json_file(solutions_file, solutions)

        print("DONE")
        print(f"output_file_path: {solutions_file}")


class Config:
    def __init__(self):
        self.openai_api_key = None
        self.test_mode = 0
        self.verbose = 0
        self.task_format = "single_sentence"
        self.context_dir = None


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

    parser.add_argument('--OPENAI_API_KEY', help="OPENAI_API_KEY as string") 
    parser.add_argument('--TEST_MODE', help="TEST_MODE as int")
    parser.add_argument('--VERBOSE', help="VERBOSE as int")
    parser.add_argument('--TASK_FORMAT', help="TASK_FORMAT string variable can be caption_lists, single_sentence or images_dict")

    args = parser.parse_args()
    
    # Parse and validate arguments
    flip_challenges_dir = Path(args.flip_challenges_dir) if args.flip_challenges_dir else None
    if flip_challenges_dir is None:
        raise Exception("flip_challenges_dir is not provided")
    
    reasoning_models = args.reasoning_models.split(',') if ',' in args.reasoning_models else [args.reasoning_models]
    reasoning_models = [model.strip() for model in reasoning_models]
    
    caption_models = args.caption_models.split(',') if ',' in args.caption_models else [args.caption_models]
    caption_models = [model.strip() for model in caption_models]
    
    caption_file_paths = args.caption_file_paths.split('::') if '::' in args.caption_file_paths else [args.caption_file_paths]
    caption_file_paths = [path.strip() for path in caption_file_paths]
    
    images_dir = Path(args.images_dir) if args.images_dir else None
    if images_dir is None:
        raise Exception("images_dir is not provided")
    
    params_file_path = Path(args.params_file_path) if args.params_file_path else None
    if params_file_path is None:
        raise Exception("params_file_path is not provided")
    
    output_file = Path(args.output_file) if args.output_file else None
    if output_file is None:
        raise Exception("output_file is not provided")
    
    context_dir = Path(args.context_dir) if args.context_dir else None
    
    # Configure settings
    config = Config()
    config.openai_api_key = args.OPENAI_API_KEY
    config.test_mode = int(args.TEST_MODE) if args.TEST_MODE is not None else 0
    config.verbose = int(args.VERBOSE) if args.VERBOSE is not None else 0
    config.task_format = args.TASK_FORMAT if args.TASK_FORMAT else "single_sentence"
    config.context_dir = context_dir
    
    # Create the challenger and run
    challenger = FlipChallenger(config)
    challenger.run(
        flip_challenges_dir=flip_challenges_dir,
        reasoning_models=reasoning_models,
        caption_models=caption_models,
        caption_file_paths=caption_file_paths,
        images_dir=images_dir,
        params_file_path=params_file_path,
        output_file=output_file
    )


if __name__ == '__main__':
    main()
import numpy as np
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import json
import traceback

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


def process_flip_batch(flip_data_batch=None, image_captions = None, caption_model=None, images_dir=None, input_format='single_sentence'):
    

    batch_results = {}
    batch_formatted_tasks = []

    for flip_data in flip_data_batch:
        # Process flip_data based on caption_model type
        try:
            # Prepare captions for the model if it's not the built-in caption model
            flip_image_captions = {
                img_name: image_captions[img_name.split("/")[-1] + ".png"][caption_model].replace("\n", " ")
                for img_name in list(dict.fromkeys(list(flip_data["image_lst1"].values())+list(flip_data["image_lst2"].values())))
            }
        except:
            print(traceback.format_exc())
            print(f"flip_image_captions failed to generate: for {flip_data}")
            raise
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

        # Format the task based on input format
        if input_format == "single_sentence":
            formatted_task = format_input_as_single_sentence(captions_list_1=image_lst1_captions, captions_list_2=image_lst2_captions)
        elif input_format == "caption_lists":
            formatted_task = format_input_with_captions(captions_list_1=image_lst1_captions, captions_list_2=image_lst2_captions)
        
        batch_formatted_tasks.append(formatted_task)
    
    return batch_formatted_tasks

# def get_story_stacks_from_flip(flip_file):
#     flip_data = read_json(flip_file)
#     image_names_lst1 = [img_name.split("/")[-1]+".png" for img_name in flip_data["image_lst1"].values()]
#     image_names_lst2 = [img_name.split("/")[-1]+".png" for img_name in flip_data["image_lst2"].values()]
    
#     # IMAGE STACKS:
#     # list of unique image names used in this flip
#     image_lst_set = list(set(image_names_lst1+image_names_lst2))
#     image_lst1_idxs= [image_lst_set.index(i) for i in image_names_lst1]
#     image_lst2_idxs= [image_lst_set.index(i) for i in image_names_lst2]
#     imgs = [ Image.open(images_dir.joinpath(i)) for i in image_lst_set ]
#     # imgs_comb_left = stack([imgs[i] for i in image_lst1_idxs])
#     # imgs_comb_right = stack([imgs[i] for i in image_lst2_idxs])

#     # IMAGE CAPTIONS
#     image_captions_left = {tested_caption: [image_captions[i][tested_caption] for i in image_names_lst1]}
#     image_captions_right = {tested_caption: [image_captions[i][tested_caption] for i in image_names_lst2]}
#     return imgs_comb_left, imgs_comb_right, image_captions_left, image_captions_right


this_dir = Path(__file__).parent
Thesis_dir = this_dir.parent.parent
print(f"this_dir: {this_dir}")
print(f"Thesis_dir: {Thesis_dir}")

images_dir = Thesis_dir.joinpath("image-captchas","Flip","data", "full_flips_set","sub_long_train_split_2","images")
flips_split = Thesis_dir.joinpath("image-captchas","Flip","data", "full_flips_set","sub_long_train_split_2","tasks")

INPUT_captions_file_name = "sub_long_train_split_2__BLIP2_flan_t5_xxl.json"

OUTPUT_results_view_dir = this_dir.joinpath("built_contexts", "exp_4", "half_2")
INPUT_results_dir = Thesis_dir.joinpath(f"flip-slim/data/exp4")
image_captions = read_json(Thesis_dir.joinpath(f"flip-slim/data/exp4",INPUT_captions_file_name))

flip_challenge_files = {challenge[6:-5]: flips_split.joinpath(challenge) for challenge in os.listdir(flips_split)}
print(len(flip_challenge_files))
OUTPUT_results_view_dir.mkdir(exist_ok=True, parents=True)

tested_caption = None
def main():
    global tested_caption

    positive_samples = []
    negative_samples = []
    
    INPUT_result_file_name = "sub_long_train_split_2_Gemini_1_5_pro_002__BLIP2_flan_t5_xxl_with_4_images_per_story"
    tested_caption = "BLIP2_flan_t5_xxl"
    tested_model = "Gemini_1_5_pro_002"

    INPUT_result_file_path = INPUT_results_dir.joinpath(INPUT_result_file_name + ".json")
    INPUT_result_file_data = read_json(INPUT_result_file_path)

    INPUT_result_file_data = INPUT_result_file_data[tested_model][tested_caption]["flips_res"]
    
    OUTPUT_results_dir_path = OUTPUT_results_view_dir.joinpath(tested_model, tested_caption)
    OUTPUT_results_dir_path.mkdir(exist_ok=True, parents=True)
    for res_key, res_val in tqdm(INPUT_result_file_data.items(), desc=f"over res keys/vals"):
        if "/flip/" in res_key:
            flip_name = res_key[6:]
            flip_data_batch = read_json(flip_challenge_files[flip_name])
            # lazy - I know, its cool
            batch_formatted_tasks = process_flip_batch(flip_data_batch=[flip_data_batch], image_captions=image_captions, caption_model=tested_caption, images_dir=images_dir)
            agreed_answer = flip_data_batch["agreed_answer"]
            flip_context = {
                "name":res_key,
                "agreed_answer":agreed_answer,
                "task_captions": batch_formatted_tasks[0]
            }
            if res_val["acc"]:
                positive_samples.append(flip_context)
            else:
                negative_samples.append(flip_context)
    write_json(file_path = OUTPUT_results_dir_path.joinpath("positive_samples.json"), data=positive_samples)
    write_json(file_path = OUTPUT_results_dir_path.joinpath("negative_samples.json"), data=negative_samples)

if __name__=="__main__":
    main()
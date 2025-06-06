#!/bin/bash

# source /mnt/c/ETH/Thesis/VENV/bin/activate
EXPERIMENT_NAME="exp0"
data_dir="/mnt/c/ETH/Thesis/flip-slim/data"
EXPERIMENT_OUT="${data_dir}/${EXPERIMENT_NAME}"
exp_dir="${data_dir}/exp0"


# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file
flip_challenges_dir="/mnt/c/ETH/Thesis/image-captchas/Flip/data/full_flips_set/full_val_split/tasks"
images_dir="/mnt/c/ETH/Thesis/image-captchas/Flip/data/full_flips_set/full_val_split/tasks/images"
# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file

reasoning_models="ChatGPT_4o"
# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file (and caption_file_paths)
# caption_models="ViPLlava_7B,LlavaNeXT_mistral_7B,LlavaNeXT_vicuna_7B,BLIP2_2_7B,BLIP2_6_7B_COCO,BLIP2_flan_t5_xxl,Llama_3_2_11B"
caption_models="BLIP2_flan_t5_xxl"
# one or multiple paths to files that contain image captions for every model from $caption_models
# in the same order as in $caption_models 
# separated by comma & no space in between  
# (duplicate caption_files entry for multiple caption models in the same file)
caption_file_paths="${exp_dir}/full_val_split_images__BLIP2_6_7B_COCO,BLIP2_flan_t5_xxl.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__ViPLlava_13B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__LlavaNeXT_mistral_7B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__LlavaNeXT_vicuna_7B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__LlavaNeXT_vicuna_13B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__BLIP2_2_7B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__BLIP2_6_7B_COCO,BLIP2_flan_t5_xxl.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__BLIP2_6_7B_COCO,BLIP2_flan_t5_xxl.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__Llama_3_2_11B.json"


params_file_path="${exp_dir}/params_reason_0.json"

output_file_name="${reasoning_models}__${caption_models}.json"
output_file_path="${exp_dir}/${output_file_name}"

HUGGING_FACE_TOKEN=""
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
TEST_MODE="0"
VERBOSE="10"

filepath="/mnt/c/ETH/Thesis/flip-slim/src/API_reason.py"

echo "running script: $filepath"

/mnt/c/ETH/Thesis/VENV/bin/python $filepath \
    --flip_challenges_dir=$flip_challenges_dir \
    --reasoning_models=$reasoning_models \
    --caption_models=$caption_models \
    --caption_file_paths=$caption_file_paths \
    --images_dir=$images_dir \
    --params_file_path=$params_file_path \
    --output_file=$output_file_path \
    --HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN \
    --GOOGLE_API_KEY=$GOOGLE_API_KEY \
    --OPENAI_API_KEY=$OPENAI_API_KEY \
    --TEST_MODE=$TEST_MODE \
    --VERBOSE=$VERBOSE \


# cp "${EXPERIMENT_OUT}/${output_file_name}" "${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "results saved to ${output_file_path}"
echo "finished at: $(date)"
exit 0
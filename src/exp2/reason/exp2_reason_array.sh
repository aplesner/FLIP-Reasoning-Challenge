#!/bin/bash
#SBATCH --job-name=array_job            # Job name
#SBATCH --output=/itet-stor/kturlan/net_scratch/slurm/output_%A_%a.log       # Standard output and error log
#SBATCH --error=/itet-stor/kturlan/net_scratch/slurm/output_%A_%a.err # redirection of stderr
#SBATCH --array=0-4%1                   # Array range and limit of 4 concurrent jobs
#SBATCH --ntasks=1                      # Run a single task
#CommentSBATCH --partition=general             # Partition (queue) name
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#SBATCH --mem=50G                        # Memory per task
#SBATCH --mail-type=NONE            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodelist=tikgpu10 # {{NODE}} # choose specific node
#SBATCH --gres=gpu:03

set -o errexit # exit on error
mkdir -p /itet-stor/${USER}/net_scratch/slurm


echo "running on node: $(hostname)"
echo "in directory: $(pwd)"
start_datetime="$(date)"
echo "starting on: ${start_datetime}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
# rm -rf /scratch/$USER/*
cp -rf /itet-stor/kturlan/net_scratch/project_storage/FLIP /scratch/${USER}/

cd /scratch/${USER}

source FLIP/src/helper_script/project_variables.sh
DIRECTORY="/scratch/$USER/cuda_sandbox"
if [ ! -d "$DIRECTORY" ]; then
  echo "$DIRECTORY does not exist. Setting it up"
  source FLIP/src/helper_script/setup_apptainer.sh
fi

cd /scratch/${USER}

export EXPERIMENT_NAME="EXP2"
export EXPERIMENT_OUT="${SCRATCH_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
export EXPERIMENT_OUT_remote="${PROJECT_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
mkdir -p $EXPERIMENT_OUT_remote
mkdir -p $EXPERIMENT_OUT


data_dir="/scratch/${USER}/FLIP/data"
exp_dir="${data_dir}/exp2"
params_file_path="${exp_dir}/params_reason_2.json"

# flip_split="full_val_split_images"
# images_dir="${data_dir}/full_val_split_images"
# flip_challenges_dir="${data_dir}/full_val_split_tasks"

flip_split="full_val_split_half_2"
images_dir="${data_dir}/full_val_split_half_2_images"
flip_challenges_dir="${data_dir}/full_val_split_half_2_tasks"


# combinations=(("meta_Llama_3_1_70B","ViPLlava_7B"),("meta_Llama_3_1_70B","ViPLlava_7B"),("meta_Llama_3_1_70B","ViPLlava_7B"))[$SLURM_ARRAY_TASK_ID]
# reasoning_models=($combinations)[0][0]
# caption_models=($combinations)[0][1]

# FOR LOOP OVER REASONING MODELS:

declare -a all_reasoning_models=("Qwen_2_5_8bit" "meta_Llama_3_1_70B" "nvm_Llama3_51B" "nvm_Llama3_70B")

# declare -a all_reasoning_models=("Qwen_2_5_8bit" "meta_Llama_3_1_70B" "nvm_Llama3_51B" "nvm_Llama3_70B")

reasoning_models="${all_reasoning_models[$SLURM_ARRAY_TASK_ID]}"
# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file (and caption_file_paths)
# caption_models="BLIP2_flan_t5_xxl"

# caption_models="meta_Llama_3_1_70B__ViPLlava_7B,meta_Llama_3_1_70B__ViPLlava_13B,meta_Llama_3_1_70B__BLIP2_flan_t5_xxl"
caption_models="Qwen_2_5_8bit__ViPLlava_13B,Qwen_2_5_8bit__BLIP2_flan_t5_xxl,nvm_Llama3_51B__ViPLlava_13B,nvm_Llama3_51B__BLIP2_flan_t5_xxl"

# caption_models="nvm_Llama3_51B__ViPLlava_7B,nvm_Llama3_51B__ViPLlava_13B,"

# one or multiple paths to files that contain image captions for every model from $caption_models
# in the same order as in $caption_models 
# separated by comma & no space in between  
# (duplicate caption_files entry for multiple caption models in the same file)
caption_file_paths="${exp_dir}/full_val_split_half_2_Qwen_2_5_8bit__ViPLlava_13B,BLIP2_flan_t5_xxl_SUMMARY.json"
caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_half_2_Qwen_2_5_8bit__ViPLlava_13B,BLIP2_flan_t5_xxl_SUMMARY.json"
caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_half_2_nvm_Llama3_51B__ViPLlava_13B,BLIP2_flan_t5_xxl_SUMMARY.json"
caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_half_2_nvm_Llama3_51B__ViPLlava_13B,BLIP2_flan_t5_xxl_SUMMARY.json"

# caption_file_paths="${exp_dir}/full_val_split_images_meta_Llama_3_1_70B__ViPLlava_7B,ViPLlava_13B,BLIP2_flan_t5_xxl_SUMMARY.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images_meta_Llama_3_1_70B__ViPLlava_7B,ViPLlava_13B,BLIP2_flan_t5_xxl_SUMMARY.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images_meta_Llama_3_1_70B__ViPLlava_7B,ViPLlava_13B,BLIP2_flan_t5_xxl_SUMMARY.json"


# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_half_2__BLIP2_6_7B_COCO,BLIP2_2_7B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_half_2__BLIP2_flan_t5_xxl.json"



output_file_name="${flip_split}_${reasoning_models}__${caption_models}.json"
output_file_path="${EXPERIMENT_OUT}/${output_file_name}"
touch $output_file_path

HUGGING_FACE_TOKEN=""
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
TEST_MODE="0"
VERBOSE="10"
BATCH_SIZE=12

filepath="/scratch/${USER}/FLIP/src/LOCAL_reason_batched.py"

echo "reasoning_models: ${reasoning_models}"
echo "caption_models: ${caption_models}"

echo "running script: $filepath"

# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --no-cache-dir --upgrade \
#   einops 
# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --no-cache-dir --upgrade \
#   auto-gptq \
#   sentencepiece \
#   ollama
# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --no-cache-dir --upgrade \
#   optimum 
# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --no-cache-dir --upgrade \
#   qwen-vl-utils 
# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --no-cache-dir --upgrade \
#   "huggingface_hub[cli]" 
# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --no-cache-dir --upgrade \
#   timm
# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --no-cache-dir --upgrade \
#   flash_attn


while [ $BATCH_SIZE -gt 1 ]; do
    echo "Attempting with BATCH_SIZE=$BATCH_SIZE"

    apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
    --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
    /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python $filepath \
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
      --BATCH_SIZE=$BATCH_SIZE


# Check the exit status of the apptainer exec command
    if [ $? -eq 0 ]; then
        echo "Job completed successfully with BATCH_SIZE=$BATCH_SIZE"
        break
    else
        echo "Job failed with BATCH_SIZE=$BATCH_SIZE. Reducing BATCH_SIZE and retrying..."
        ((BATCH_SIZE--))
    fi
done

if [ $BATCH_SIZE -eq 1 ]; then
    echo "Job could not be completed successfully even with BATCH_SIZE=1"
    exit 1
fi

cp "${EXPERIMENT_OUT}/${output_file_name}" "${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "results copied to ${USER}@tik42x.ethz.ch:${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "finished at: $(date)"


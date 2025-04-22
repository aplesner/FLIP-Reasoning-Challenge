#!/bin/bash
#SBATCH --mail-type=NONE # disable email notifications can be [NONE, BEGIN, END, FAIL, REQUEUE, ALL]
#SBATCH --output=/itet-stor/kturlan/net_scratch/slurm/%j.out # redirection of stdout (%j is the job id)
#SBATCH --error=/itet-stor/kturlan/net_scratch/slurm/%j.err # redirection of stderr
#SBATCH --nodelist=tikgpu10 # {{NODE}} # choose specific node
#SBATCH --mem=30G
#SBATCH --nodes=1
#SBATCH --gres=gpu:02
#SBATCH --cpus-per-task=4
#CommentSBATCH --exclude=tikgpu[08-10]
#CommentSBATCH --cpus-per-task=4
#CommentSBATCH --nodelist=tikgpu01 # example: specify a node
#CommentSBATCH --account=tik-internal # example: charge a specific account
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb' # example: specify a gpu


set -o errexit # exit on error
mkdir -p /itet-stor/${USER}/net_scratch/slurm

echo "running on node: $(hostname)"
echo "in directory: $(pwd)"
echo "starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# cp -rf /itet-stor/kturlan/net_scratch/project_storage/FLIP /scratch/${USER}/
# mkdir -p /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/LOCAL_reason.py /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/utils.py /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/caption_images.py /scratch/${USER}/FLIP/src
cd /scratch/${USER}

source FLIP/src/helper_script/project_variables.sh
export EXPERIMENT_NAME="EXP0"
export EXPERIMENT_OUT="${SCRATCH_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
export EXPERIMENT_OUT_remote="${PROJECT_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
mkdir -p $EXPERIMENT_OUT_remote
mkdir -p $EXPERIMENT_OUT



data_dir="${SCRATCH_STORAGE_DIR}/data"
exp_dir="${data_dir}/exp0"
# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file
flip_split="full_val_split_half_2_tasks"
flip_challenges_dir="${data_dir}/${flip_split}"
images_dir="${data_dir}/full_val_split_half_2_images"
# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file
# meta_Llama_3_1_70B ~ 136_664 MiB at bfloat16
# Qwen_2_VL_8bit ~ 88_000 MiB at 8bit
reasoning_models="nvm_Llama3_70B"

# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file (and caption_file_paths)
caption_models="LlavaNeXT_mistral_7B,LlavaNeXT_vicuna_7B,LlavaNeXT_vicuna_13B"

# caption_models="ViPLlava_7B,ViPLlava_13B,LlavaNeXT_mistral_7B,LlavaNeXT_vicuna_7B,LlavaNeXT_vicuna_13B,BLIP2_2_7B,BLIP2_6_7B_COCO,BLIP2_flan_t5_xxl,Llama_3_2_11B"
# one or multiple paths to files that contain image captions for every model from $caption_models
# in the same order as in $caption_models 
# separated by comma & no space in between  
# (duplicate caption_files entry for multiple caption models in the same file)
caption_file_paths="${exp_dir}/full_val_split_half_2__LlavaNeXT_mistral_7B,LlavaNeXT_vicuna_7B,LlavaNeXT_vicuna_13B.json"
caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_half_2__LlavaNeXT_mistral_7B,LlavaNeXT_vicuna_7B,LlavaNeXT_vicuna_13B.json"
caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_half_2__LlavaNeXT_mistral_7B,LlavaNeXT_vicuna_7B,LlavaNeXT_vicuna_13B.json"

# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__ViPLlava_13B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__LlavaNeXT_mistral_7B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__LlavaNeXT_vicuna_7B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__LlavaNeXT_vicuna_13B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__BLIP2_2_7B.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__BLIP2_6_7B_COCO,BLIP2_flan_t5_xxl.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__BLIP2_6_7B_COCO,BLIP2_flan_t5_xxl.json"
# caption_file_paths="${caption_file_paths}::${exp_dir}/full_val_split_images__Llama_3_2_11B.json"


params_file_path="${exp_dir}/params_reason_0.json"

output_file_name="${flip_split}_${reasoning_models}__${caption_models}.json"
output_file_path="${EXPERIMENT_OUT}/${output_file_name}"
touch $output_file_path

HUGGING_FACE_TOKEN=""
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
TEST_MODE="0"
VERBOSE="10"

filepath="FLIP/src/LOCAL_reason_batched.py"

echo "running script: $filepath"


# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox python --version

# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox nvcc --version

# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install flash_attn==2.2.1

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
    --VERBOSE=$VERBOSE 


cp "${EXPERIMENT_OUT}/${output_file_name}" "${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "results copied to ${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "finished at: $(date)"
exit 0
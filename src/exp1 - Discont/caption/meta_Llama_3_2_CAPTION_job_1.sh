#!/bin/bash
#SBATCH --mail-type=NONE # disable email notifications can be [NONE, BEGIN, END, FAIL, REQUEUE, ALL]
#SBATCH --output=/itet-stor/kturlan/net_scratch/slurm/%j.out # redirection of stdout (%j is the job id)
#SBATCH --error=/itet-stor/kturlan/net_scratch/slurm/%j.err # redirection of stderr
#SBATCH --nodelist=tikgpu07 # {{NODE}} # choose specific node
#SBATCH --mem=30G
#SBATCH --nodes=1
#SBATCH --gres=gpu:03
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

cp -rf /itet-stor/kturlan/net_scratch/project_storage/FLIP /scratch/${USER}/
# mkdir -p /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/LOCAL_reason.py /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/utils.py /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/caption_images.py /scratch/${USER}/FLIP/src
cd /scratch/${USER}

source FLIP/src/helper_script/project_variables.sh
export EXPERIMENT_NAME="EXP1"
export EXPERIMENT_OUT="${SCRATCH_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
export EXPERIMENT_OUT_remote="${PROJECT_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
mkdir -p $EXPERIMENT_OUT_remote
mkdir -p $EXPERIMENT_OUT



data_dir="${SCRATCH_STORAGE_DIR}/data"
exp_dir="${data_dir}/exp1"
params_file_path="${exp_dir}/params_caption_1.json"
# one or multiple model (separated by comma & no space in between) \
# names as defined in the params_file
flip_split="full_val_split_images"

images_dir="${data_dir}/${flip_split}"
# one or multiple model (separated by comma & no space in between) names as defined in the params_file 
# LlavaNeXT_mistral_7B = 14_798MiB at bfloat16
# LlavaNeXT_vicuna_13B = 26_112 MiB at bfloat16
# ViPLlava_13B ~ 26_152MiB at bfloat16 precision
# ViPLlava_7B ~ 13_770MiB at bfloat16 precision
# Llama_3_2_11B ~ 21_166MiB at bfloat16
# all together ~120GB ! ACCOUNT FOR PROCESSING VRAM AS WELL ! need about 5-10GB free room!

caption_models="Llama_3_2_11B"


output_file_name="test_${flip_split}__${caption_models}.json"
output_file_path="${EXPERIMENT_OUT}/${output_file_name}"
touch $output_file_path

HUGGING_FACE_TOKEN=""
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
TEST_MODE="1"

filepath="FLIP/src/caption_images.py"

echo "running script: $filepath"

# apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
# --env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
# /scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python -m pip install --upgrade transformers


apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
--env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
/scratch/$USER/cuda_sandbox /scratch/$USER/apptainer_env/venv/bin/python $filepath \
    --caption_models=$caption_models \
    --images_dir=$images_dir \
    --params_file_path=$params_file_path \
    --output_file=$output_file_path \
    --HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN \
    --GOOGLE_API_KEY=$GOOGLE_API_KEY \
    --OPENAI_API_KEY=$OPENAI_API_KEY \
    --TEST_MODE=$TEST_MODE \

cp -f "${output_file_path}" "${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "results copied to ${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "finished at: $(date)"
exit 0





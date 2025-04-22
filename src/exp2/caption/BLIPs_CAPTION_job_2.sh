#!/bin/bash
#SBATCH --mail-type=NONE # disable email notifications can be [NONE, BEGIN, END, FAIL, REQUEUE, ALL]
#SBATCH --output=/itet-stor/kturlan/net_scratch/slurm/%j.out # redirection of stdout (%j is the job id)
#SBATCH --error=/itet-stor/kturlan/net_scratch/slurm/%j.err # redirection of stderr
#SBATCH --nodelist=tikgpu10 # {{NODE}} # choose specific node
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --gres=gpu:01
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
start_datetime="$(date)"
echo "starting on: ${start_datetime}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cp -rf /itet-stor/kturlan/net_scratch/project_storage/FLIP /scratch/${USER}/
# mkdir -p /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/LOCAL_reason.py /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/utils.py /scratch/${USER}/FLIP/src
# cp  /itet-stor/kturlan/net_scratch/project_storage/FLIP/src/caption_images.py /scratch/${USER}/FLIP/src
cd /scratch/${USER}

source FLIP/src/helper_script/project_variables.sh
export EXPERIMENT_NAME="EXP2"
export EXPERIMENT_OUT="${SCRATCH_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
export EXPERIMENT_OUT_remote="${PROJECT_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
mkdir -p $EXPERIMENT_OUT_remote
mkdir -p $EXPERIMENT_OUT


data_dir="/scratch/${USER}/FLIP/data"
exp_dir="${data_dir}/exp2"

flip_split="full_val_split_half_2"

images_dir="${data_dir}/full_val_split_half_2_images"

# one or multiple model (separated by comma & no space in between) names as defined in the params_file 
# blip2-flan-t5-xxl ~ 36.426GB in 8bit
# BLIP2_6_7B_COCO ~ 8_201MiB in 8 bit
# BLIP2_2_7B ~ 7_934 MiB at bloat16
caption_models="BLIP2_flan_t5_xxl"

params_file_path="${exp_dir}/params_caption_2.json"

output_file_name="${flip_split}__${caption_models}.json"
output_file_path="${EXPERIMENT_OUT}/${output_file_name}"
touch $output_file_path

HUGGING_FACE_TOKEN=""
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
TEST_MODE="0"

filepath="FLIP/src/caption_images.py"

echo "running script: $filepath"
echo "caption_models: $caption_models"
echo "params_file_path: $params_file_path"
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

cp "${EXPERIMENT_OUT}/${output_file_name}" "${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "results copied to ${USER}@tik42x.ethz.ch:${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "finished at: $(date)"
exit 0





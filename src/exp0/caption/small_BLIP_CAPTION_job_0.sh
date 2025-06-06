#!/bin/bash
#SBATCH --mail-type=NONE # disable email notifications can be [NONE, BEGIN, END, FAIL, REQUEUE, ALL]
#SBATCH --output=/itet-stor/kturlan/net_scratch/slurm/%j.out # redirection of stdout (%j is the job id)
#SBATCH --error=/itet-stor/kturlan/net_scratch/slurm/%j.err # redirection of stderr
#SBATCH --nodelist=tikgpu02 # {{NODE}} # choose specific node
#SBATCH --mem=30G
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
echo "starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cp -r -u /itet-stor/kturlan/net_scratch/project_storage/FLIP /scratch/${USER}/
cd /scratch/${USER}/FLIP

source ./src/helper_script/project_variables.sh
export EXPERIMENT_NAME="EXP0"
export EXPERIMENT_OUT="${SCRATCH_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
export EXPERIMENT_OUT_remote="${PROJECT_STORAGE_DIR}/${EXPERIMENT_NAME}/OUTPUT"
mkdir -p $EXPERIMENT_OUT_remote
mkdir -p $EXPERIMENT_OUT


data_dir="${SCRATCH_STORAGE_DIR}/data"

flip_split="full_val_split_images"

images_dir="${data_dir}/${flip_split}"

# one or multiple model (separated by comma & no space in between) names as defined in the params_file 
# BLIP2_2_7B ~ 7_934 MiB at bloat16
caption_models="BLIP2_2_7B"

params_file_path="${SCRATCH_STORAGE_DIR}/data/params_caption_0.json"

output_file_name="${flip_split}__${caption_models}.json"
output_file_path="${EXPERIMENT_OUT}/${output_file_name}"
touch $output_file_path

HUGGING_FACE_TOKEN=""
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
TEST_MODE="0"

filepath="src/caption_images.py"

echo "running script: $filepath"
apptainer exec --nv --bind "/scratch/$USER:/scratch/$USER" \
--env LC_ALL=C,USER=$USER_NAME,TMPDIR=$TMPDIR,PYTHONUSERBASE=$PYTHONUSERBASE,PYTHONNOUSERSITE=$PYTHONNOUSERSITE,PIP_CACHE_DIR=$PIP_CACHE_DIR,PYTHONPATH=$PYTHONPATH,JUPYTER_DATA_DIR=$JUPYTER_DATA_DIR,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE,TORCH_HOME=$TORCH_HOME \
flip_container.sif python3 $filepath \
    --caption_models=$caption_models \
    --images_dir=$images_dir \
    --params_file_path=$params_file_path \
    --output_file=$output_file_path \
    --HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN \
    --GOOGLE_API_KEY=$GOOGLE_API_KEY \
    --OPENAI_API_KEY=$OPENAI_API_KEY \
    --TEST_MODE=$TEST_MODE \

cp "${EXPERIMENT_OUT}/${output_file_name}" "${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "results copied to ${EXPERIMENT_OUT_remote}/${output_file_name}"
echo "finished at: $(date)"
exit 0




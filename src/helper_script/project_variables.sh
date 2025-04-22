#!/bin/bash



export USER_NAME="kturlan"
export PROJECT_NAME="FLIP"
export PROJECT_STORAGE_DIR="/itet-stor/${USER_NAME}/net_scratch/project_storage/${PROJECT_NAME}"
export SCRATCH_STORAGE_DIR="/scratch/${USER_NAME}/${PROJECT_NAME}"

export REMOTE_SERVER="tik42x.ethz.ch"
export CODE_STORAGE_DIR="/home/${USER_NAME}/code/${PROJECT_NAME}"
export DATA_STORAGE_DIR="${PROJECT_STORAGE_DIR}/data"
export MODEL_STORAGE_DIR="${PROJECT_STORAGE_DIR}/models"
export SINGULARITY_STORAGE_DIR="${PROJECT_STORAGE_DIR}/singularity"




mkdir -p /scratch/$USER_NAME/apptainer_env/venv/.local
mkdir -p /scratch/$USER_NAME/apptainer_env/.local
mkdir -p /scratch/$USER_NAME/apptainer_env/pip_cache
mkdir -p /scratch/$USER_NAME/apptainer_env/site_packages
mkdir -p /scratch/$USER_NAME/apptainer_env/jupyter_data
mkdir -p /scratch/$USER_NAME/apptainer_env/hf_cache
mkdir -p /scratch/$USER_NAME/apptainer_env/hf_cache
mkdir -p /scratch/$USER_NAME/apptainer_env/hf_cache
mkdir -p /scratch/$USER_NAME/apptainer_env/torch_cache

export PYTHONUSERBASE=/scratch/$USER_NAME/apptainer_env/.local
export TMPDIR=/scratch/$USER_NAME/apptainer_env/venv/.local
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/scratch/$USER_NAME/apptainer_env/pip_cache
export PYTHONPATH=$PYTHONPATH:/scratch/$USER_NAME/apptainer_env/site_packages
export JUPYTER_DATA_DIR=/scratch/$USER_NAME/apptainer_env/jupyter_data
export HF_HOME=/scratch/$USER_NAME/apptainer_env/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER_NAME/apptainer_env/hf_cache
export TORCH_HOME=/scratch/$USER_NAME/apptainer_env/torch_cache
export HUGGINGFACE_HUB_CACHE=/scratch/$USER_NAME/apptainer_env/hf_cache

mkdir -p $HF_HOME/accelerate && echo "compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: 'all'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false" > $HF_HOME/accelerate/default_config.yaml


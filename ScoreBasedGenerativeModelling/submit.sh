#!/bin/bash
#SBATCH -J train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 # per node
#SBATCH -p gpu2
#SBATCH --cpus-per-task=8
#SBATCH --output=/homefs/home/winklep4/Work/JaxLightning/pcluster_logs/%A/%A.%a-output.txt
#SBATCH --error=/homefs/home/winklep4/Work/JaxLightning/pcluster_logs/%A/%A.%a-error.txt
### SBATCH --time 1-00:00:00
#SBATCH --signal=SIGUSR1@90

. ~/.bashrc
nvidia-smi

eval "$(micromamba shell hook --shell bash)"
micromamba activate jax
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1

wandb login --host https://genentech.wandb.io
wandb login --relogin
wandb artifact cache cleanup 50G

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "${PWD}"

### Set the wandb agent command -------------------------------
# Set the wandb agent command
WANDB_AGENT_CMD="
wandb agent ludwig-winkler/protein-correction_testing/3hngzz7v
"

# Check if an argument is provided
if [ -n "$1" ]; then
    WANDB_AGENT_CMD="$WANDB_AGENT_CMD --count $1"
fi

# Run the wandb agent command
$WANDB_AGENT_CMD

### Set the wandb agent command -------------------------------

python main.py
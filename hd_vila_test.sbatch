#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A cstdl
#SBATCH --partition develbooster
#SBATCH --gres gpu
#SBATCH --time=02:00:00

source /p/project/ccstdl/zhou12/miniconda3/etc/profile.d/conda.sh
conda activate videorl
python3 download_videos.py
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=11000M
#SBATCH --time=2-23:00
#SBATCH --account=def-rzheng
python crnn_entrypoint.py
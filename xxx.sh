#!/bin/bash

git pull
rm slurm*
sbatch job_train_fb.sh
sbatch job_train_wn.sh
cd ..
# scp -r fwolf@lisa.surfsara.nl:~/results/scratch/rgvae/data/model/GVAE_fb15k_110e_14l_20201110.pt /home/wolf/Desktop/results

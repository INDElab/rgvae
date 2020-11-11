#!/bin/bash

git pull
rm slurm*
sbatch lp1.sh
sbatch lp2.sh
sbatch lp3.sh
sbatch lp4.sh

# scp -r fwolf@lisa.surfsara.nl:~/results/scratch/rgvae/data/model/GVAE_fb15k_110e_14l_20201110.pt /home/wolf/Desktop/results

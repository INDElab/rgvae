#!/bin/bash

git pull
rm slurm*
sbatch lp1.sh
sbatch lp2.sh
sbatch lp3.sh
sbatch lp4.sh

# scp -r /home/wolf/Desktop/results/fb15k/GVAE_fb15k_110e_14l_20201110.pt fwolf@lisa.surfsara.nl:~/rgvae/data/model/GVAE_fb15k_110e_14l_20201110.pt 

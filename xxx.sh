#!/bin/bash

git pull
rm slurm*
sbatch job_train_fb.sh
sbatch job_train_wn.sh
cd ..
scp -r fwolf@lisa.surfsara.nl:/results /home/wolf/Desktop

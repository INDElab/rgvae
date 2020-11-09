#!/bin/bash

git pull
rm slurm*
sbtach job1.sh

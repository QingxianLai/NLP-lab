#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=1:00:00
#PBS -l mem=2GB
#PBS -N tryQSUB
#PBS -j oe

cd ./NLUDR/NLP-lab 

module purge

module load cuda/7.0.28
module load theano/0.7.0

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python ./test_GPU.py

#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=2G
#PBS -N lab3.2.1Tesla
#PBS -j oe
#PBS -m abe
#PBS -M ql516@nyu.edu

cd ./NLUDR/NLP-lab/lab3/code

module purge

module load cuda/7.0.28
module load theano/0.7.0
module load scikit-learn/intel/0.16.1

THEANO_FLAGS=device=gpu,floatX=float32 python lstmlm.py config.yaml


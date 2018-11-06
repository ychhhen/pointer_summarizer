#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=60:00:00
#PBS -q v100_normal_q
#PBS -A BDTScascades
#PBS -W group_list=cascades
#PBS -M ashishb@vt.edu
#PBS -m bea

cd $PBS_O_WORKDIR
echo `pwd`

python training_ptr_gen/train.py 

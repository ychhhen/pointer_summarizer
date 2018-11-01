#module load cuda/9.0.176
#module load Anaconda/5.1.0
#source activate pytorch
python training_ptr_gen/train.py >& log/training_log


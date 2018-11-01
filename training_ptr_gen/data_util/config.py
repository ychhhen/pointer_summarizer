import os

from pathlib2 import Path

root_dir = os.path.expanduser("../dataset/news/output/")

train_data_path = os.path.join(root_dir, "finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "finished_files/test.bin")
vocab_path = os.path.join(root_dir, "finished_files/vocab")
log_root = os.path.join(root_dir, "log")

# Hyperparameters
hidden_dim = 256
emb_dim = 128
batch_size = 8
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 50000

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu = True

lr_coverage = 0.15

# Logging
log_level = 'DEBUG'
log_file = os.path.join(log_root, 'training_{}.log')


def reset_path_variables():
    global train_data_path, eval_data_path, decode_data_path, vocab_path
    global log_root, log_file
    train_data_path = os.path.join(root_dir, "finished_files/chunked/train_*")
    eval_data_path = os.path.join(root_dir, "finished_files/val.bin")
    decode_data_path = os.path.join(root_dir, "finished_files/test.bin")
    vocab_path = os.path.join(root_dir, "finished_files/vocab")
    log_root = os.path.join(root_dir, "log")
    log_file = os.path.join(log_root, 'training_{}.log')


def reset_log_variables():
    global log_root, log_file
    path = Path(log_root)
    path.mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_root, 'training_{}.log')

import torch
from dataclasses import dataclass


## Get Config
@dataclass
class config:
    dataset_path = "./data"
    batch_size = 32
    num_epochs = 10
    lr = 10**-4  # 1e-3
    seq_len = 350
    d_model = 512
    lang_src = "en"
    lang_tgt = "it"
    tokenizer_file = "tokenizer_{0}.json"
    num_workers = 4
    pin_memory = False
    weight_decay = 1e-4
    eps = 1e-9
    label_smoothing = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

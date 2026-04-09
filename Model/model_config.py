import torch
import torch.optim as optim

SEED = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100

BATCH_SIZE = 32

OPTIMIZERS = {
    "adam": optim.Adam,
    "adamw": optim.AdamW
}

LABEL_THRESHOLD = 0.4
GNN_AUTO_LABEL_F1_THRESHOLD = 0.85

PARAM_GRID = {
    "lr":[1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],
    "hidden_dims":[
        (256, 128, 64), 
        (128, 64, 32)
        ],
    "dropout":[
        (0.1, 0.1, 0.2), 
        (0.1, 0.3, 0.5)
        ],
}

MIN_DELTA = 1e-4
EARLY_STOPPING_PATIENCE = 10

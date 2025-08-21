import os
import sys
import matplotlib.pyplot as plt

def build_config(batch: int, image_size: int, n_classes: int):

    config_dict = {
        "patch_size": 4,
        "embed_dim": 128,
        "image_size": image_size,
        "batch": batch,
        "no_of_classes": n_classes,
        "num_of_blocks": 6,
        "num_of_heads": 12,
        "intermediate_size": 4 * 128,
    }

    return config_dict

def plot_loss_acc(train_history, val_history):
    pass

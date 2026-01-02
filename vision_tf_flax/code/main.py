"""
main.py
Execution of system
"""

import os
import sys
import argparse
import logging
try:
    import flax
except ImportError:
    os.system("pip install flax")
    import flax
from flax import nnx
try:
    import jax
except ImportError:
    os.system("pip install jax")
    import jax
import jax.numpy as jnp
try:
    import optax
except ImportError:
    os.system("pip install optax")
    import optax
import tqdm
from sklearn.model_selection import train_test_split
import jax.numpy as jnp

# User defined
from dataset import Dataloader
from vit import ViT
from utils import *
from training import *

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="Base path containing the images in the described folder format")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=1e-5)
    parser.add_argument("--momemtum", type=float, default=0.9)
    opt = parser.parse_args()

    return opt

def main(opt):
    # Check if the base path exists.
    try:
        os.path.exists(opt.base_path)
    except OSError:
        print("Directory does not exist")    

    # Create the logging dir if not.
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir, exist_ok=True)

    # Load the NPY files as JAX NUMPY
    train_val_images = jnp.load(os.path.join(opt.base_path, "train_val_data.npy"))
    train_val_labels = jnp.load(os.path.join(opt.base_path, "train_val_labels.npy"))

    # Normalize the images
    train_val_images = train_val_images / 255.0

    # Get the number of classes
    n_classes = jnp.unique_counts(train_val_labels)[0].shape[0]

    # Split Training and validation splits
    x_train, x_val, y_train, y_val = train_test_split(train_val_images, train_val_labels, test_size=0.8, random_state=42)

    total_steps = opt.epochs * (x_train.shape[0] // opt.batch)


    print("#"*50)
    print(f"Total Samples: {x_train.shape[0]}")
    print("#"*50)

    # Load the train and val dataloader
    train_dataloader = Dataloader(
        x_train, y_train, batch=opt.batch, shuffle=True
    )

    val_dataloader = Dataloader(
        x_val, y_val, batch=opt.batch, shuffle=False
    )

    # Build the config file to build the model
    config_file = build_config(batch=opt.batch, image_size=train_val_images.shape[1], n_classes=n_classes)
    
    # Create the model instance 
    model = ViT(config=config_file)

    # Configure Optimizer
    lr_schedule = optax.linear_schedule(opt.lr0, opt.lrf, total_steps)

    optim = nnx.Optimizer(
        model, optax.adam(lr_schedule), wrt=nnx.Param
    )

    # Define validation metrics
    eval_metrics = nnx.MultiMetric(
        loss = nnx.metrics.Average("loss"),
        accuracy = nnx.metrics.Accuracy(),
    )

    # Define dictionary to track train and validation metrics.
    train_history = {
        "train_loss_step": [],
        "train_loss_epoch": [] 
    }

    val_history = {
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(opt.epochs):
        train_loss = train_one_epoch(model, optim, opt.batch, epoch, opt.epochs, train_dataloader, train_history)
        print(train_loss)
        eval_model(model, epoch, opt.epochs, eval_metrics, val_dataloader, val_history)

    print(len(train_history["train_loss_step"]))
    print(len(val_history["val_loss"]))

if __name__ == "__main__":
    opt = arg_parser()
    main(opt)



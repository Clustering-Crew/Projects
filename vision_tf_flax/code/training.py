import os
import optax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tqdm
import dataset

tqdm_bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"

def loss_fn(model: nnx.Module, images: jax.Array, labels: jax.Array):
    logits = model(images)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, logits

@nnx.jit
def train_step(model: nnx.Module, optim: nnx.Optimizer, images: jax.Array, labels: jax.Array):

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)
    
    optim.update(model, grads)
    return loss

@nnx.jit
def eval_step(model: nnx.Module, images: jax.Array, labels: jax.Array, eval_metrics: nnx.MultiMetric):
    loss, logits = loss_fn(model, images, labels)
    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels
    )

def train_one_epoch(model: nnx.Module, optim: nnx.Optimizer, batch: int, epoch: int, epochs: int, train_dataloader: dataset.Dataloader, train_history: dict[str, list]):
    train_loss = []
    model.train()

    with tqdm.tqdm(
        desc=f"[Train] epoch: {epoch} / {epochs}, ",
        total=len(train_dataloader),
        bar_format=tqdm_bar_format,
        leave=True
    ) as pbar:
        for images, labels in train_dataloader:
            loss = train_step(model, optim, images, labels)
            train_history["train_loss_step"].append(loss.item())
            train_loss.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
        
        return sum(train_loss)/len(train_loss)

def eval_model(model: nnx.Module, epoch: int, epochs: int, eval_metrics: nnx.MultiMetric, val_dataloader: dataset.Dataloader, val_history: dict[str, list]):
    model.eval()
    eval_metrics.reset()

    for val_images, val_labels in val_dataloader:
        loss = eval_step(model, val_images, val_labels, eval_metrics)
    
    for metric, value in eval_metrics.compute().items():
        val_history[f"val_{metric}"].append(value)

    print(f"[Val] epoch: {epoch + 1} / {epochs}")
    print(f"Loss: {val_history['val_loss'][-1]:0.4f}")
    print(f"Accuracy: {val_history['val_accuracy'][-1]:0.4f}")

import os
import sys
import jax
import jax.numpy as jnp

class Dataloader:
    def __init__(self, x, y, batch, shuffle):
        self.x = x
        self.y = y
        self.batch = batch
        self.shuffle = shuffle
        self.n_samples = self.x.shape[0]
        self.total_batches = self.n_samples // self.batch
        self.indices = jnp.arange(self.n_samples)
        self.key = jax.random.PRNGKey(0)
        if self.shuffle:
            jax.random.permutation(self.key, self.indices, axis=0)
    
    def __len__(self):
        return self.total_batches
    
    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            jax.random.permutation(self.key, self.indices, axis=0)

        return self
    
    def __next__(self):
        if self.current_batch >= self.total_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch
        end_idx = min(start_idx + self.batch, self.n_samples)

        x_batch = self.x[start_idx:end_idx, :, :, :]
        y_batch = self.y[start_idx:end_idx]

        self.current_batch += 1

        return x_batch, y_batch
    
    def get_batch(self):
        pass

    def reset(self):
        self.current_batch = 0
        if self.shuffle:
            jax.random.permutation(self.key, self.indices, axis=0)
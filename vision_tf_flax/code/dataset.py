import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

# Numpy Dataloader:
class Dataloader:
    """
        Custom dataloader to load the data from NPY files.

        Input:
        x: Input features
        y: Ground truth labels
        batch: batch size
        shuffle: data shuffle
    
    """
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
        ''' Returns the number of batches'''
        return self.total_batches
    
    def __iter__(self):
        '''Converts the dataloader as an iterator'''
        self.current_batch = 0
        if self.shuffle:
            jax.random.permutation(self.key, self.indices, axis=0)

        return self
    
    def __next__(self):
        """Loads the next the batch"""
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

# Folder Dataloader
class FolderDataloader:
    """
        Custom Dataloader to load the images directly from the folders

        Inputs
        data directory: path of the images per class.

        batch: batch size
        shuffle: enable batch shuffling
    """

    def __init__(self, data_dir: str, img_size: int, batch: int, shuffle: bool, resize: bool):
        self.data_dir = data_dir
        self.batch = batch
        self.shuffle = shuffle
        self.size = img_size
        self.resize = resize

        self.all_images = []
        self.all_labels = []
        self.global_key = jax.random.PRNGKey(42) # Used in shuffling the whole dataset
        self.batch_key = jax.random.PRNGKey(123) # Used in shuffling the batch.

        self.create_dataset()

    def create_dataset(self):
        
        for i, folder in tqdm(enumerate(self.data_dir)):
            # Collect all the image files names in the list
            self.file_names = os.listdir(os.path.join(self.data_dir, folder))

            # Add the parent directory to the file name.
            self.file_paths = [file + os.path.join(self.data_dir, folder) for file in self.file_names]

            # Map the `load_image` function to the file paths list
            self.all_images += list(map(self.load_image, self.file_paths))
            self.all_labels += [i] * len(self.file_names)
        
        # Convert the lists into JAX Numpy arrays
        self.all_images = jnp.array(self.all_images, dtype=np.float32)
        self.all_labels = jnp.array(self.all_labels, dtype=np.int8)
    
        self.num_of_batches = self.all_labels // self.batch
        
        # Global shuffle
        self.all_images = jax.random.permutation(self.global_key, self.all_images, axis=0)
        self.all_labels = jax.random.permutation(self.global_key, self.all_labels, axis=0)

        return self.all_images, self.labels


    def load_image(self, path):
        img = Image.open(path)
        
        if self.resize:
            img = img.resize((self.size, self.size))
        
        img = img / 255.0
        img = jnp.array(img, dtype=np.float32) # Convert PIL Image object to JAX numpy array

        return img

    def __iter__(self):
        self.current_batch = 0

        if self.shuffle:
            jax.random.permutation(self.batch_key, self.indices, axis=0)
        
        return self

    def __next__(self):
        if self.current_batch >= self.num_of_batches:
            raise StopIteration
        
        start_idx = self.current_batch
        end_idx = start_idx + self.batch

        images = self.all_images[start_idx:end_idx, :, :, :]
        labels = self.all_labels[start_idx:end_idx]
        
    def __len__(self):
        return self.num_of_batches







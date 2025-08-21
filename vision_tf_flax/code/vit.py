import os
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Embedding Class
class Embed(nnx.Module):

    # Embedding = Patch Embedding + Cls Token + Pos Embedding
    def __init__(self, config, rngs):
        self.patch_size = config["patch_size"]
        self.embed_dim = config["embed_dim"]
        self.image_height = config["image_size"]
        self.image_width = config["image_size"]
        self.batch_size = config["batch"]
        self.patch_count = (self.image_height * self.image_width) // self.patch_size ** 2
        self.rng = rngs

        self.proj_layer = nnx.Conv(
            in_features=3,
            out_features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            rngs=self.rng
        )

        self.cls_token = nnx.Param(
            jax.random.normal(self.rng.params(), (1, 1, self.embed_dim))
        )
        
        self.pos_embedding = nnx.Param(
            jax.random.normal(self.rng.params(), (self.batch_size, self.patch_count + 1, self.embed_dim))
        )

    def __call__(self, x):
        x = self.proj_layer(x)
        x = jnp.reshape(x, (x.shape[0], (x.shape[1] * x.shape[2]), x.shape[3]))
        cls_tokens = jnp.tile(self.cls_token, [self.batch_size, 1, 1])
        x = jnp.concatenate([x, cls_tokens], axis=1)

        x = x + self.pos_embedding
        return x

# Attention Head
class AttentionHead(nnx.Module):
    def __init__(self, embed_dim, attention_head_size, bias=True):
        self.embed_dim= embed_dim
        self.attention_head_size = attention_head_size

        # Query, Key and Value Weight matrices
        self.q_w = nnx.Linear(in_features=self.embed_dim, out_features=self.attention_head_size, use_bias=True, rngs=nnx.Rngs(0))
        self.k_w = nnx.Linear(in_features=self.embed_dim, out_features=self.attention_head_size, use_bias=True, rngs=nnx.Rngs(0))
        self.v_w = nnx.Linear(in_features=self.embed_dim, out_features=self.attention_head_size, use_bias=True, rngs=nnx.Rngs(0))

    def __call__(self, x):
        q_x, k_x, v_x = self.q_w(x), self.k_w(x), self.v_w(x)

        # Calculate QK^T/sqrt(dk)
        attn_out = jnp.matmul(q_x, jnp.matrix_transpose(k_x)) / jnp.sqrt(self.attention_head_size)
        # Apply softmax
        softmax_out = nnx.softmax(attn_out)
        # Obtain the attention value with value
        attn_value = jnp.matmul(softmax_out, v_x)

        return attn_value

# Multiheadattention
class MultiHeadAttention(nnx.Module):
    def __init__(self, config):
        self.embed_dim = config["embed_dim"]
        self.num_of_heads = config["num_of_heads"]

        self.attn_head_size = self.embed_dim // self.num_of_heads
        self.all_head_size = self.attn_head_size * self.num_of_heads

        self.heads = []

        for _ in range(self.num_of_heads):
            self.attn_head = AttentionHead(
                embed_dim=self.embed_dim,
                attention_head_size=self.attn_head_size
            )
            self.heads.append(self.attn_head)

        self.linear_proj = nnx.Linear(in_features=self.all_head_size, out_features=self.embed_dim, use_bias=True, rngs=nnx.Rngs(0))
        self.dropout = nnx.Dropout(0.3, rngs=nnx.Rngs(0))

    def __call__(self, x):
        attn_outputs = [head(x) for head in self.heads]

        concat_output = jnp.concatenate(attn_outputs, axis=-1)
        proj_output = self.linear_proj(concat_output)
        proj_output = self.dropout(x)

        return proj_output

# MLP class
class MLP(nnx.Module):
    def __init__(self, config):
        self.embed_dim = config["embed_dim"]
        self.intermediate_size = config["intermediate_size"]

        self.linear1 = nnx.Linear(in_features=self.embed_dim, out_features=self.intermediate_size, use_bias=True, rngs=nnx.Rngs(0))
        self.linear2= nnx.Linear(in_features=self.intermediate_size, out_features=self.embed_dim, use_bias=True, rngs=nnx.Rngs(0))
        self.dropout = nnx.Dropout(0.3, rngs=nnx.Rngs(0))

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x

# Single block
class Block(nnx.Module):
    def __init__(self, config):
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.norm = nnx.LayerNorm(num_features=self.embed_dim, rngs=nnx.Rngs(0))
        self.mha = MultiHeadAttention(self.config)
        self.mlp = MLP(self.config)

    def __call__(self, x):
        norm_out = self.norm(x)
        attn_out = self.mha(x)
        attn_out = x + attn_out
        norm_out = self.norm(attn_out)
        mlp_out = self.mlp(norm_out)
        block_out = mlp_out + norm_out

        return block_out

# Encoder
class Encoder(nnx.Module):
    def __init__(self, config):
        self.config = config
        self.num_of_blocks = config["num_of_blocks"]

        self.blocks = []

        for _ in range(self.num_of_blocks):
            block = Block(self.config)
            self.blocks.append(block)


    def __call__(self, x):
        all_attns = []

        for block in self.blocks:
            x = block(x)
        return x

# ViT Classifier
class ViT(nnx.Module):
    def __init__(self, config):
        self.embed_layer = Embed(config, nnx.Rngs(0))
        self.encoder = Encoder(config)
        self.classifier = nnx.Linear(in_features=config["embed_dim"], out_features=config["no_of_classes"], use_bias=True, rngs=nnx.Rngs(0))

    def __call__(self, x):
        x = self.embed_layer(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0, :])

        return x
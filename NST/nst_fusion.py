# -*- coding: utf-8 -*-
"""NST_fusion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yYMDzTPgNHvqh4DaZsfOvC_SQ4LERL3G
"""

import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

def image_process(img_path):
  image = load_img(img_path)
  image = img_to_array(image)
  image = cv2.resize(image, (256, 256))
  image = tf.expand_dims(image, 0)

  return image

def plot_image(title, image):
  plt.title(title)
  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.show()

def add_feats(img_features):
  #size will be (batch, size, size, channels)
  img_feats = img_features[0]

  added_feats = np.zeros((128, 128))
  for i in range(img_feats.shape[2]):
    added_feats += img_feats[:, :, i]

  added_feats = cv2.resize(added_feats, (110, 110), interpolation = cv2.INTER_AREA)
  return added_feats

def feature_fuse(vis, ir):
  size = vis.shape[0]
  fused = np.zeros((size, size))

  for i in range(size):
    for j in range(size):
      fused[i][j] = max(vis[i][j], ir[i][j])

  return fused

def read_gray(path):
  size = 110
  img = cv2.imread(path, 0)
  img = cv2.resize(img, (110, 110), interpolation = cv2.INTER_AREA)

  return img

def loss_fn(fused_feat, combination_image):
  loss = 0
  for i in range(fused_feat.shape[0]):
    for j in range(fused_feat.shape[0]):
      loss += (fused_feat[i][j] - combination_image[i][j])**2

      #loss = (loss / (fused_feat.shape[0])**2)
  return loss

@tf.function
def compute_loss_grad(fused_feat, init_fuse, combination_image):
  with tf.GradientTape() as tape:
    #loss = loss_fn(fused_feat, combination_image) + content_loss(init_fuse, combination_image)
    loss = loss_fn(fused_feat, combination_image)
  grads = tape.gradient(loss, combination_image)

  return loss, grads

def training_loop(epochs, fused_feat, init_fuse, combination_image, name, width, height, key):

  optimizer = tf.keras.optimizers.Adam(
    learning_rate = 0.01, use_ema = True
  )

  for i in range(1, epochs + 1):
    loss, grads = compute_loss_grad(
        fused_feat, init_fuse, combination_image
    )

    optimizer.apply_gradients([(grads, combination_image)])

    if i % 1000 == 0:
      print(f"Iteration {i} : loss = {loss} ")
      plt.imshow(combination_image.numpy(), cmap = 'gray')
      plt.axis('off')
      plt.show()
      comb_image = cv2.resize(combination_image.numpy(), (width, height), interpolation=cv2.INTER_CUBIC)
      plt.imsave(fname=f'/content/drive/MyDrive/image_fusion_NST/{name}_result_in_{i}.png', arr=comb_image, cmap='gray', format='png')


      #cv2.imwrite(f"/content/drive/MyDrive/image_fusion_NST/{name}_result_in_{i}.png", comb_image)

def fusion(vis_path, ir_path, name, width, height, key):
  vis_img = image_process(vis_path)
  ir_img = image_process(ir_path)

  resnet = ResNet50(
      weights = 'imagenet',
      include_top = False,
      input_shape = (256, 256, 3)
  )

  model = Model(resnet.inputs, resnet.layers[4].output)

  vis_feats = model.predict(vis_img)
  ir_feats = model.predict(ir_img)

  vis_feats = add_feats(vis_feats)
  ir_feats = add_feats(ir_feats)

  fused_feats = feature_fuse(vis_feats, ir_feats)

  plot_image("Vis Features", vis_feats)
  plot_image("IR features", ir_feats)
  plot_image("Fused features", fused_feats)

  plt.imsave("/content/drive/MyDrive/fused_features_BD.bmp", fused_feats, cmap='gray')

  vis_gray = read_gray(vis_path)
  ir_gray = read_gray(ir_path)

  init_fuse = np.mean([vis_gray, ir_gray], axis=0)

  plot_image("Visible Gray Image", vis_gray)
  plot_image("IR Gray Image", ir_gray)
  plot_image("Initial Fused Image", init_fuse)

  fused_feat = fused_feats
  init_fuse = init_fuse
  comb_image = tf.Variable(init_fuse)

  training_loop(510000, fused_feat, init_fuse, comb_image, name, width, height, key)

import os

ir_src_path = "/content/drive/MyDrive/test_data/ir/"
vis_src_path = "/content/drive/MyDrive/test_data/vis/"

vis_imgs = ['/content/drive/MyDrive/test_data/vis//VIS_meting016_r.bmp',
            '/content/drive/MyDrive/test_data/vis//VIS_meting012-1500_r.bmp',
            '/content/drive/MyDrive/test_data/vis//VIS_maninhuis_r.bmp',
            '/content/drive/MyDrive/test_data/vis//VIS_meting003_r.bmp',
            '/content/drive/MyDrive/test_data/vis//VIS_18dhvR.bmp',
            '/content/drive/MyDrive/test_data/vis//VIS.bmp',
            '/content/drive/MyDrive/test_data/vis//Kaptein_1654_Vis.bmp',
            '/content/drive/MyDrive/test_data/vis//VIS-MarnehNew_15RGB_603.tif',
            '/content/drive/MyDrive/test_data/vis//Movie_18_Vis.bmp',
            '/content/drive/MyDrive/test_data/vis//1823v.bmp',
            '/content/drive/MyDrive/test_data/vis//a_VIS-MarnehNew_24RGB_1110.tif',
            '/content/drive/MyDrive/test_data/vis//Kaptein_1123_Vis.bmp',
            '/content/drive/MyDrive/test_data/vis//4908v.bmp']
ir_imgs = ['/content/drive/MyDrive/test_data/ir//IR_meting016_g.bmp',
           '/content/drive/MyDrive/test_data/ir//IR_meting012-1500_g.bmp',
           '/content/drive/MyDrive/test_data/ir//IR_maninhuis_g.bmp',
           '/content/drive/MyDrive/test_data/ir//IR_meting003_g.bmp',
           '/content/drive/MyDrive/test_data/ir//IR_18rad.bmp',
           '/content/drive/MyDrive/test_data/ir//LWIR.bmp',
           '/content/drive/MyDrive/test_data/ir//Kaptein_1654_IR.bmp',
           '/content/drive/MyDrive/test_data/ir//LWIR-MarnehNew_15RGB_603.tif',
           '/content/drive/MyDrive/test_data/ir//Movie_18_IR.bmp',
           '/content/drive/MyDrive/test_data/ir//1823i.bmp',
           '/content/drive/MyDrive/test_data/ir//c_LWIR-MarnehNew_24RGB_1110.tif',
           '/content/drive/MyDrive/test_data/ir//Kaptein_1123_IR.bmp',
           '/content/drive/MyDrive/test_data/ir//4908i.bmp'
           ]



key = 20
name = f"test_case_{key}"
vis = cv2.imread('/content/drive/MyDrive/t01_v.bmp', 0)
width = vis.shape[1]
height = vis.shape[0]

fusion('/content/drive/MyDrive/t01_v.bmp', '/content/drive/MyDrive/t01_i.bmp', name, width, height, key)

img = cv2.imread(f"/content/drive/MyDrive/image_fusion_NST/test_case_{20}_result_in_9000.png", 0)

img.shape

plt.imshow(img, cmap='gray')
plt.imsave(fname=f'/content/drive/MyDrive/Fused_images_NST/fused_{20}.png', arr=img, cmap='gray', format='png')


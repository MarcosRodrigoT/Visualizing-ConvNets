import numpy as np
import tensorflow as tf
from imagenet_labels import IMAGENET_LABELS
import cv2
import matplotlib.pyplot as plt
import itertools


"""
Setup
"""


# The dimensions of our input image
IMG_WIDTH = 224
IMG_HEIGHT = 224
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.

# Conv2
LAYER_NAME = 'conv2_block1_out'     # Output shape: (None, 45, 45, 256)
# LAYER_NAME = 'conv2_block2_out'     # Output shape: (None, 45, 45, 256)
# LAYER_NAME = 'conv2_block3_out'     # Output shape: (None, 23, 23, 256)

# Conv3
# LAYER_NAME = 'conv3_block1_out'     # Output shape: (None, 23, 23, 512)
# LAYER_NAME = 'conv3_block2_out'     # Output shape: (None, 23, 23, 512)
# LAYER_NAME = 'conv3_block3_out'     # Output shape: (None, 23, 23, 512)
# LAYER_NAME = 'conv3_block4_out'     # Output shape: (None, 12, 12, 512) -> Default

# Conv4
# LAYER_NAME = 'conv4_block1_out'     # Output shape: (None, 12, 12, 1024)
# LAYER_NAME = 'conv4_block2_out'     # Output shape: (None, 12, 12, 1024)
# LAYER_NAME = 'conv4_block3_out'     # Output shape: (None, 12, 12, 1024)
# LAYER_NAME = 'conv4_block4_out'     # Output shape: (None, 12, 12, 1024)
# LAYER_NAME = 'conv4_block5_out'     # Output shape: (None, 12, 12, 1024)
# LAYER_NAME = 'conv4_block6_out'     # Output shape: (None, 6, 6, 1024)

# Conv5
# LAYER_NAME = 'conv5_block1_out'     # Output shape: (None, 6, 6, 2048)
# LAYER_NAME = 'conv5_block2_out'     # Output shape: (None, 6, 6, 2048)
# LAYER_NAME = 'conv5_block3_out'     # Output shape: (None, 6, 6, 2048)

# Last layers
# LAYER_NAME = 'post_bn'      # Output shape: (None, 6, 6, 2048)
# LAYER_NAME = 'post_relu'    # Output shape: (None, 6, 6, 2048)


"""
Build a feature extraction model
"""


# Build a ResNet50V2 model loaded with pre-trained ImageNet weights
inputs = tf.keras.layers.Input(shape=[224, 224, 3], dtype=tf.float32)
x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
backbone = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=True)
backbone.trainable = False
outputs = backbone(x, training=False)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.summary(line_length=150, expand_nested=True)

# Set up a model that returns the activation values for our target layer
layer = backbone.get_layer(name=LAYER_NAME)
feature_extractor = tf.keras.Model(inputs=[backbone.inputs], outputs=[layer.output])

# Load test image
image = tf.image.decode_jpeg(tf.io.read_file('test_images/forklift.jpg'))
image = image[None, ...]
image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])

# Make predictions and extract features
predictions = model(image)
features = feature_extractor(image).numpy().squeeze()


"""
Visualize the activation maps
"""


def visualize_activation_maps(activation_maps=range(0, 1), figsize=(10, 10)):
    # Build a black picture with enough space for our n x n filters of size 128 x 128, with a 5px margin in between
    n = np.sqrt(len(activation_maps))
    assert n.is_integer(), f'The square root of the number of filters must be a whole number, ' \
                           f'got num_filters={len(activation_maps)}, which square root is {n:.2f}'
    n = int(n)
    fig, ax = plt.subplots(ncols=n, nrows=n, figsize=(10, 10))
    for idx, ax_idx in zip(activation_maps, list(itertools.product(range(n), range(n)))):
        img = cv2.resize(features[:, :, idx], (IMG_WIDTH, IMG_HEIGHT))
        img = img[..., None]
        ax[ax_idx[0], ax_idx[1]].imshow(img)

    # Save results
    result_name = f'results/activation_maps_{LAYER_NAME}_{activation_maps[0]}-{activation_maps[-1]}.png'
    [axi.set_axis_off() for axi in ax.ravel()]
    fig.suptitle(f'Activation Maps for layer {LAYER_NAME} - filters from {activation_maps[0]} to {activation_maps[-1]}')
    plt.savefig(f'{result_name}')


# Plot features
visualize_activation_maps(activation_maps=range(0, 64), figsize=(10, 10))
visualize_activation_maps(activation_maps=range(64, 64+64), figsize=(10, 10))
visualize_activation_maps(activation_maps=range(128, 128+64), figsize=(10, 10))


"""
Print classification results
"""


# TOP-K classification results
k = 5
top_idx = np.squeeze(np.argsort(predictions, axis=1)[:, -k:])
top_val = np.squeeze(np.sort(predictions, axis=1)[:, -k:])
top_k_results = {key: val for key, val in zip(top_idx, top_val)}
sorted_top_k_results = dict(sorted(top_k_results.items(), key=lambda z: z[1], reverse=True))
print('###################################################################################')
for key, val in sorted_top_k_results.items():
    print(f'{IMAGENET_LABELS[key][:40]:40s} -> {val:.7f}')
print('###################################################################################')

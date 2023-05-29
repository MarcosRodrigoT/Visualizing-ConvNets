import numpy as np
import tensorflow as tf
from tensorflow import keras


"""
Setup
"""


# The dimensions of our input image
IMG_WIDTH = 180
IMG_HEIGHT = 180
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
model = keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
model.summary(line_length=150)

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=LAYER_NAME)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)


"""
Set up the gradient ascent process
"""


def compute_loss(input_image, filter_index_):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index_]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img_, filter_index_, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img_)
        loss_ = compute_loss(img_, filter_index_)
    # Compute gradients.
    grads = tape.gradient(loss_, img_)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img_ += learning_rate * grads
    return loss_, img_


"""
Set up the end-to-end filter visualization loop
"""


def initialize_image():
    # We start from a gray image with some random noise
    img_ = tf.random.uniform((1, IMG_WIDTH, IMG_HEIGHT, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img_ - 0.5) * 0.25


def visualize_filter(filter_index_):
    # We run gradient ascent for 30 steps
    iterations = 30
    learning_rate = 10.0
    img_ = initialize_image()
    for iteration in range(iterations):
        loss_, img_ = gradient_ascent_step(img_, filter_index_, learning_rate)

    # Decode the resulting input image
    img_ = deprocess_image(img_[0].numpy())
    return loss_, img_


def deprocess_image(img_):
    # Normalize array: center on 0., ensure variance is 0.15
    img_ -= img_.mean()
    img_ /= img_.std() + 1e-5
    img_ *= 0.15

    # Center crop
    img_ = img_[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img_ += 0.5
    img_ = np.clip(img_, 0, 1)

    # Convert to RGB array
    img_ *= 255
    img_ = np.clip(img_, 0, 255).astype("uint8")
    return img_


def visualize_results(filters=range(0, 1)):
    # Build a black picture with enough space for our n x n filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = np.sqrt(len(filters))
    assert n.is_integer(), f'The square root of the number of filters must be a whole number, ' \
                           f'got num_filters={len(filters)}, which square root is {n:.2f}'
    n = int(n)
    cropped_width = IMG_WIDTH - 25 * 2
    cropped_height = IMG_HEIGHT - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Compute image inputs that maximize per-filter activations for the first 64 filters of our target layer
    all_images = []
    for filter_ in filters:
        print(f'Processing filter {filter_}')
        loss, img = visualize_filter(filter_)
        all_images.append(img)

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = all_images[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j: (cropped_height + margin) * j + cropped_height,
                :,
            ] = img

    # Save results
    result_name = f'results/stitched_filters_{LAYER_NAME}_{filters[0]}-{filters[-1]}.png'
    keras.preprocessing.image.save_img(result_name, stitched_filters)


if __name__ == '__main__':
    # visualize_results(filters=range(0, 1))
    visualize_results(filters=range(0, 64))
    visualize_results(filters=range(64, 64+64))
    visualize_results(filters=range(128, 128+64))

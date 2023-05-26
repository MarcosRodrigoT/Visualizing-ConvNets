import numpy as np
import tensorflow as tf
from tensorflow import keras


"""
Setup
"""


# The dimensions of our input image
img_width = 180
img_height = 180
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = 'conv3_block4_out'


"""
Build a feature extraction model
"""


# Build a ResNet50V2 model loaded with pre-trained ImageNet weights
model = keras.applications.ResNet50V2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
model.summary(line_length=150)

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
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
    img_ = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img_ - 0.5) * 0.25


def visualize_filter(filter_index_):
    # We run gradient ascent for 20 steps
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


"""
Example (remove later)
"""
# Let's try it out with filter 0 in the target layer:
loss, img = visualize_filter(0)
# This is what an input that maximizes the response of filter 0 in the target layer would look like:
keras.preprocessing.image.save_img("results/filter_0.png", img)


if __name__ == '__main__':
    """
    Visualize the first 64 filters in the target layer
    """
    # Compute image inputs that maximize per-filter activations for the first 64 filters of our target layer
    all_imgs = []
    for filter_index in range(64):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = 8
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = all_imgs[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j: (cropped_height + margin) * j + cropped_height,
                :,
            ] = img
    keras.preprocessing.image.save_img("results/stitched_filters.png", stitched_filters)

import tensorflow as tf
import numpy as np
import imageio
from PIL import Image

# Convolutional layer
def conv2d(input_map, num_output_channels, size_kernel=5, stride=2, name='conv2d'):
    conv_layer = tf.keras.layers.Conv2D(
        filters=num_output_channels,
        kernel_size=size_kernel,
        strides=stride,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer='zeros',
        name=name
    )
    return conv_layer(input_map)

# Fully connected (dense) layer
def fc(input_vector, num_output_length, name='fc'):
    dense_layer = tf.keras.layers.Dense(
        units=num_output_length,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer='zeros',
        name=name
    )
    return dense_layer(input_vector)

# Deconvolution (transpose convolution) layer
def deconv2d(input_map, output_shape, size_kernel=5, stride=2, stddev=0.02, name='deconv2d'):
    deconv_layer = tf.keras.layers.Conv2DTranspose(
        filters=output_shape[-1],
        kernel_size=size_kernel,
        strides=stride,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=stddev),
        bias_initializer='zeros',
        name=name
    )
    return deconv_layer(input_map)

# Leaky ReLU activation
def lrelu(logits, leak=0.2):
    return tf.nn.leaky_relu(logits, alpha=leak)

# Concatenate label information
def concat_label(x, label, duplicate=1):
    x_shape = tf.shape(x)
    if duplicate < 1:
        return x
    label = tf.tile(label, [1, duplicate])
    if len(x.shape) == 2:
        return tf.concat([x, label], axis=1)
    elif len(x.shape) == 4:
        label = tf.reshape(label, [-1, 1, 1, label.shape[-1]])
        ones = tf.ones([x_shape[0], x_shape[1], x_shape[2], label.shape[-1]])
        label = label * ones
        return tf.concat([x, label], axis=3)

# Load an image and resize it
def load_image(
    image_path, 
    image_size=64, 
    image_value_range=(-1, 1), 
    is_gray=False
):
    if is_gray:
        image = imageio.v2.imread(image_path, as_gray=True).astype(np.float32)
        image = np.expand_dims(image, axis=-1)  # Expand for grayscale
    else:
        image = imageio.v2.imread(image_path).astype(np.float32)

    image = Image.fromarray(np.uint8(image))
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32)

    image = image * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image

# Save a batch of images into a grid
def save_batch_images(
    batch_images,
    save_path,
    image_value_range=(-1, 1),
    size_frame=None
):
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    images = np.clip(images, 0.0, 1.0)

    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    num_channels = batch_images.shape[3]
    frame = np.zeros((img_h * size_frame[0], img_w * size_frame[1], num_channels))

    for idx, image in enumerate(images):
        row = idx // size_frame[1]
        col = idx % size_frame[1]
        frame[
            row * img_h: (row + 1) * img_h,
            col * img_w: (col + 1) * img_w,
            :
        ] = image

    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    imageio.v2.imwrite(save_path, frame)










import os
import time
from glob import glob
import numpy as np
import tensorflow as tf
from scipy.io import savemat
from ops import *  # Assuming your ops.py is compatible or will be updated similarly

tf.keras.backend.clear_session()

class FaceAging(object):
    def __init__(self,
                 size_image=128,
                 size_kernel=5,
                 size_batch=100,
                 num_input_channels=3,
                 num_encoder_channels=64,
                 num_z_channels=50,
                 num_categories=10,
                 num_gen_channels=1024,
                 enable_tile_label=True,
                 tile_ratio=1.0,
                 is_training=True,
                 save_dir='./save',
                 dataset_name='UTKFace'):

        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name

        # Inputs
        self.input_image = tf.Variable(tf.zeros([self.size_batch, self.size_image, self.size_image, self.num_input_channels]), trainable=False)
        self.age = tf.Variable(tf.zeros([self.size_batch, self.num_categories]), trainable=False)
        self.gender = tf.Variable(tf.zeros([self.size_batch, 2]), trainable=False)
        self.z_prior = tf.Variable(tf.zeros([self.size_batch, self.num_z_channels]), trainable=False)

        # Build the model
        print('\n\tBuilding graph...')

        # Build components
        self.build_model()

        # Setup checkpoint saving
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=None, model=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=os.path.join(self.save_dir, 'checkpoint'), max_to_keep=2)
    def build_model(self):
        # Encoder
        self.z = self.encoder(self.input_image)

        # Generator
        self.G = self.generator(self.z, self.age, self.gender, self.enable_tile_label, self.tile_ratio)

        # Discriminators
        self.D_z, self.D_z_logits = self.discriminator_z(self.z, is_training=self.is_training)
        self.D_G, self.D_G_logits = self.discriminator_img(self.G, self.age, self.gender, is_training=self.is_training)
        self.D_z_prior, self.D_z_prior_logits = self.discriminator_z(self.z_prior, is_training=self.is_training, reuse_variables=True)
        self.D_input, self.D_input_logits = self.discriminator_img(self.input_image, self.age, self.gender, is_training=self.is_training, reuse_variables=True)

        # Losses
        self.define_losses()
    def define_losses(self):
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss

        self.D_z_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_z_prior_logits), logits=self.D_z_prior_logits)
        )
        self.D_z_loss_z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_z_logits), logits=self.D_z_logits)
        )
        self.E_z_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_z_logits), logits=self.D_z_logits)
        )

        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_input_logits), logits=self.D_input_logits)
        )
        self.D_img_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_G_logits), logits=self.D_G_logits)
        )
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_G_logits), logits=self.D_G_logits)
        )

        # Total variation loss
        self.tv_loss = (
            (tf.reduce_mean(tf.square(self.G[:, 1:, :, :] - self.G[:, :-1, :, :]))) +
            (tf.reduce_mean(tf.square(self.G[:, :, 1:, :] - self.G[:, :, :-1, :])))
        )
    def setup_optimizers(self, learning_rate=0.0002, beta1=0.5):
        self.learning_rate = tf.Variable(learning_rate, trainable=False)

        self.optimizer_EG = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=beta1)
        self.optimizer_Dz = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=beta1)
        self.optimizer_Di = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=beta1)
    @tf.function
    def train_step(self, batch_images, batch_label_age, batch_label_gender, batch_z_prior, weights=(0.0001, 0, 0)):
        with tf.GradientTape(persistent=True) as tape:
            # Update inputs
            self.input_image.assign(batch_images)
            self.age.assign(batch_label_age)
            self.gender.assign(batch_label_gender)
            self.z_prior.assign(batch_z_prior)

            # Forward pass
            self.build_model()

            # Full losses
            loss_EG = self.EG_loss + weights[0] * self.G_img_loss + weights[1] * self.E_z_loss + weights[2] * self.tv_loss
            loss_Dz = self.D_z_loss_prior + self.D_z_loss_z
            loss_Di = self.D_img_loss_input + self.D_img_loss_G

        # Compute gradients
        EG_vars = self.get_encoder_generator_vars()
        Dz_vars = self.get_discriminator_z_vars()
        Di_vars = self.get_discriminator_img_vars()

        grads_EG = tape.gradient(loss_EG, EG_vars)
        grads_Dz = tape.gradient(loss_Dz, Dz_vars)
        grads_Di = tape.gradient(loss_Di, Di_vars)

        # Apply gradients
        self.optimizer_EG.apply_gradients(zip(grads_EG, EG_vars))
        self.optimizer_Dz.apply_gradients(zip(grads_Dz, Dz_vars))
        self.optimizer_Di.apply_gradients(zip(grads_Di, Di_vars))

        return loss_EG, loss_Dz, loss_Di
    def train(self, num_epochs=200, learning_rate=0.0002, beta1=0.5, decay_rate=1.0, enable_shuffle=True):
        # Setup optimizers
        self.setup_optimizers(learning_rate=learning_rate, beta1=beta1)

        # Load dataset
        file_names = glob(os.path.join('./data', self.dataset_name, '*.jpg'))
        size_data = len(file_names)
        np.random.seed(2017)

        # Training loop
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)

            num_batches = size_data // self.size_batch

            for idx in range(num_batches):
                # Load batch
                batch_files = file_names[idx * self.size_batch:(idx + 1) * self.size_batch]
                batch_images, batch_label_age, batch_label_gender = self.load_batch(batch_files)
                batch_z_prior = np.random.uniform(
                    self.image_value_range[0], self.image_value_range[-1],
                    size=[self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                # Train step
                loss_EG, loss_Dz, loss_Di = self.train_step(batch_images, batch_label_age, batch_label_gender, batch_z_prior)

                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{idx+1}/{num_batches}] - EG_loss: {loss_EG:.4f}, Dz_loss: {loss_Dz:.4f}, Di_loss: {loss_Di:.4f}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.manager.save()

    def load_batch(self, batch_files):
        batch = [load_image(
            image_path=f,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1)
        ) for f in batch_files]

        if self.num_input_channels == 1:
            batch_images = np.expand_dims(np.array(batch).astype(np.float32), axis=-1)
        else:
            batch_images = np.array(batch).astype(np.float32)

        # Labels
        batch_label_age = np.ones((len(batch_files), self.num_categories), dtype=np.float32) * self.image_value_range[0]
        batch_label_gender = np.ones((len(batch_files), 2), dtype=np.float32) * self.image_value_range[0]

        for i, path in enumerate(batch_files):
            label = int(os.path.basename(path).split('_')[0])
            gender = int(os.path.basename(path).split('_')[1])

            label = self.map_age_to_label(label)
            batch_label_age[i, label] = self.image_value_range[-1]
            batch_label_gender[i, gender] = self.image_value_range[-1]

        return batch_images, batch_label_age, batch_label_gender

    def map_age_to_label(self, age):
        if 0 <= age <= 5:
            return 0
        elif 6 <= age <= 10:
            return 1
        elif 11 <= age <= 15:
            return 2
        elif 16 <= age <= 20:
            return 3
        elif 21 <= age <= 30:
            return 4
        elif 31 <= age <= 40:
            return 5
        elif 41 <= age <= 50:
            return 6
        elif 51 <= age <= 60:
            return 7
        elif 61 <= age <= 70:
            return 8
        else:
            return 9
    def sample(self, images, labels, gender, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        self.input_image.assign(images)
        self.age.assign(labels)
        self.gender.assign(gender)

        z = self.encoder(self.input_image)
        G = self.generator(z, self.age, self.gender, self.enable_tile_label, self.tile_ratio)

        # Denormalize images
        G = (G + 1.0) / 2.0  # from (-1,1) to (0,1)

        grid = self.create_image_grid(G, int(np.sqrt(self.size_batch)))
        filepath = os.path.join(sample_dir, name)

        self.save_image_grid(grid, filepath)
    def test(self, images, gender, name):
        test_dir = os.path.join(self.save_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)

        images = images[:int(np.sqrt(self.size_batch))]
        gender = gender[:int(np.sqrt(self.size_batch))]
        size_sample = images.shape[0]

        labels = np.arange(size_sample)
        labels = np.repeat(labels, size_sample)
        query_labels = np.ones((size_sample ** 2, size_sample), dtype=np.float32) * self.image_value_range[0]

        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]

        query_images = np.tile(images, [size_sample, 1, 1, 1])
        query_gender = np.tile(gender, [size_sample, 1])

        self.input_image.assign(query_images)
        self.age.assign(query_labels)
        self.gender.assign(query_gender)

        z = self.encoder(self.input_image)
        G = self.generator(z, self.age, self.gender, self.enable_tile_label, self.tile_ratio)

        # Denormalize
        G = (G + 1.0) / 2.0

        grid = self.create_image_grid(G, size_sample)
        input_grid = self.create_image_grid(query_images, size_sample)

        self.save_image_grid(input_grid, os.path.join(test_dir, 'input.png'))
        self.save_image_grid(grid, os.path.join(test_dir, name))
    def create_image_grid(self, images, grid_size):
        """Arrange batch of images into a grid."""
        images = tf.clip_by_value(images, 0.0, 1.0)
        batch, h, w, c = images.shape
        images = tf.reshape(images, (grid_size, grid_size, h, w, c))
        images = tf.transpose(images, (0, 2, 1, 3, 4))
        images = tf.reshape(images, (grid_size * h, grid_size * w, c))
        return images

    def save_image_grid(self, grid, filepath):
        """Save a grid of images to a file."""
        grid = tf.image.convert_image_dtype(grid, dtype=tf.uint8)
        encoded_image = tf.image.encode_png(grid)
        tf.io.write_file(filepath, encoded_image)

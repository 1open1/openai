# -*- coding: utf-8 -*-

"""##1) Implement a basic autoencoder and train it on a dataset like MNIST for image reconstruction."""#


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0

# Autoencoder architecture
inp = layers.Input((28,28,1))
x = layers.Conv2D(32,3,activation='relu',padding='same')(inp)
x = layers.MaxPooling2D(2,padding='same')(x)
x = layers.Conv2D(16,3,activation='relu',padding='same')(x)
x = layers.MaxPooling2D(2,padding='same')(x)
x = layers.Conv2D(16,3,activation='relu',padding='same')(x)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
x = layers.UpSampling2D(2)(x)
out = layers.Conv2D(1,3,activation='sigmoid',padding='same')(x)

autoencoder = models.Model(inp, out)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test), verbose=1)

# Visualize
decoded = autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    ax.set_title('Original')
    ax.axis('off')
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded[i].squeeze(), cmap='gray')
    ax.set_title('Reconstructed')
    ax.axis('off')
plt.show()

2) Explore different regularization techniques such as L1/L2 regularization or dropout and compare their effects on the autoencoder's performance."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load and preprocess data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

def build_autoencoder(reg_type=None, reg_rate=0.0, dropout_rate=0.0):
    input_img = layers.Input((28, 28, 1))

    if reg_type == 'l1':
        regularizer = regularizers.l1(reg_rate)
    elif reg_type == 'l2':
        regularizer = regularizers.l2(reg_rate)
    elif reg_type == 'l1_l2':
        regularizer = regularizers.l1_l2(l1=reg_rate, l2=reg_rate)
    else:
        regularizer = None

    x = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizer)(input_img)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.Conv2D(16, 3, activation='relu', padding='same', kernel_regularizer=regularizer)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    encoded = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    return models.Model(input_img, decoded)

def train_autoencoder(config):
    model = build_autoencoder(config['reg_type'], config['reg_rate'], config['dropout_rate'])
    model.compile(optimizer=tf.keras.optimizers.Adam(config['lr']), loss='binary_crossentropy')

    callbacks_list = []
    if config['early_stop']:
        callbacks_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))

    if config['data_aug']:
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
        train_gen = datagen.flow(x_train, x_train, batch_size=128)
        history = model.fit(train_gen, steps_per_epoch=len(x_train)//128, epochs=2, validation_data=(x_test, x_test), callbacks=callbacks_list, verbose=0)
    else:
        history = model.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test), callbacks=callbacks_list, verbose=0)

    return round(history.history['loss'][-1], 4), round(history.history['val_loss'][-1], 4)

# Experiment configurations
experiments = [
    {'reg_type': None, 'reg_rate': 0.0, 'dropout_rate': 0.0, 'lr': 0.001, 'early_stop': False, 'data_aug': False},
    {'reg_type': 'l1', 'reg_rate': 0.001, 'dropout_rate': 0.0, 'lr': 0.001, 'early_stop': False, 'data_aug': False},
    {'reg_type': 'l2', 'reg_rate': 0.001, 'dropout_rate': 0.0, 'lr': 0.001, 'early_stop': False, 'data_aug': False},
    {'reg_type': 'l1_l2', 'reg_rate': 0.001, 'dropout_rate': 0.0, 'lr': 0.001, 'early_stop': False, 'data_aug': False},
    {'reg_type': None, 'reg_rate': 0.0, 'dropout_rate': 0.3, 'lr': 0.001, 'early_stop': False, 'data_aug': False},
    {'reg_type': None, 'reg_rate': 0.0, 'dropout_rate': 0.0, 'lr': 0.001, 'early_stop': True, 'data_aug': False},
    {'reg_type': None, 'reg_rate': 0.0, 'dropout_rate': 0.0, 'lr': 0.001, 'early_stop': False, 'data_aug': True},
]

# Run experiments
results = []
for i, config in enumerate(experiments, 1):
    print(f"Experiment {i}: {config}")
    train_loss, val_loss = train_autoencoder(config)

    if config['reg_type']:
        reg_desc = config['reg_type'].upper()
    elif config['dropout_rate'] > 0:
        reg_desc = "Dropout"
    elif config['early_stop']:
        reg_desc = "Early Stopping"
    elif config['data_aug']:
        reg_desc = "Data Augmentation"
    else:
        reg_desc = "None"

    results.append({
        'S.no': i,
        'Regularization': reg_desc,
        'Dropout_rate': config['dropout_rate'],
        'Regularization_rate': config['reg_rate'],
        'Learning_rate': config['lr'],
        'Training_loss': train_loss,
        'Validation_loss': val_loss
    })

# Display results
print("\nResults:")
for r in results:
    print(f"{r['S.no']:2d} | {r['Regularization']:15s} | Dropout: {r['Dropout_rate']:<4} | Reg Rate: {r['Regularization_rate']:<6} | LR: {r['Learning_rate']:<6} | Train Loss: {r['Training_loss']:<6} | Val Loss: {r['Validation_loss']:<6}")

"""# 3) Implement a variational autoencoder (VAE) and train it on a dataset like MNIST to generate new images."""

import numpy as np, tensorflow as tf, keras
from keras import layers, ops

class Sampling(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seed = keras.random.SeedGenerator(1337)  # Create once here
    def call(self, inputs):
        m, s = inputs
        eps = keras.random.normal(ops.shape(m), seed=self.seed)
        return m + s * eps

# Encoder
latent_dim = 2
e_in = keras.Input((28,28,1))
x = layers.Conv2D(32,3,2,"same",activation="relu")(e_in)
x = layers.Conv2D(64,3,2,"same",activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(16,activation="relu")(x)
m = layers.Dense(latent_dim)(x)
s = layers.Dense(latent_dim,activation="softplus")(x)
z = Sampling()([m,s])
encoder = keras.Model(e_in, [m, s, z])

# Decoder
d_in = keras.Input((latent_dim,))
x = layers.Dense(7*7*64,activation="relu")(d_in)
x = layers.Reshape((7,7,64))(x)
x = layers.Conv2DTranspose(64,3,2,"same",activation="relu")(x)
x = layers.Conv2DTranspose(32,3,2,"same",activation="relu")(x)
d_out = layers.Conv2DTranspose(1,3,1,"same",activation="sigmoid")(x)
decoder = keras.Model(d_in, d_out)

# VAE
class VAE(keras.Model):
    def __init__(self, enc, dec, **kw):
        super().__init__(**kw)
        self.enc, self.dec = enc, dec
    def train_step(self, data):
        with tf.GradientTape() as tape:
            m, s, z = self.enc(data)
            r = self.dec(z)
            rl = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, r), axis=(1,2)))
            kl = tf.reduce_mean(tf.reduce_sum(0.5*(s**2+m**2 - 2*tf.math.log(s+1e-8)-1), axis=1))
            loss = rl + kl
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss}

# Data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
data = np.expand_dims(np.concatenate([x_train, x_test])/255.0, -1)

# Train
vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")
vae.fit(data, epochs=30, batch_size=128)

# Generate Variations
idx = int(input("Enter index: "))
if not 0 <= idx < len(data): raise ValueError("Index out of range")
img = data[idx:idx+1]
plt.imshow(img[0].squeeze(), cmap="gray"); plt.axis("off"); plt.show()
m, s, _ = encoder.predict(img)
noise = np.random.normal(size=(10, latent_dim))
z_vars = m + s * noise
gen = decoder.predict(z_vars)
plt.figure(figsize=(15,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(gen[i].squeeze(), cmap="gray")
    plt.axis("off")
plt.show()

"""# 4) Implement a basic autoregressive model like the Fully Visible Sigmoid Belief Network (FVSBN) and train it on a dataset like MNIST."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

class FVSBNLayer(tf.keras.layers.Layer):
    """Custom layer for FVSBN autoregressive computation."""
    def __init__(self, dim):
        super(FVSBNLayer, self).__init__()
        self.dim = dim
        self.mask = tf.constant(np.tril(np.ones((dim, dim)), k=-1), dtype=tf.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=(self.dim, self.dim), initializer="random_normal")
        self.bias = self.add_weight(name="bias", shape=(self.dim,), initializer="zeros")

    def call(self, inputs):
        masked_kernel = self.kernel * self.mask
        logits = tf.matmul(inputs, masked_kernel, transpose_b=True) + self.bias
        return logits

class FVSBN(tf.keras.Model):
    """FVSBN model for binary data."""
    def __init__(self, dim):
        super(FVSBN, self).__init__()
        self.dim = dim
        self.layer = FVSBNLayer(dim)

    def call(self, inputs):
        logits = self.layer(inputs)
        return tf.nn.sigmoid(logits)

    def compute_loss(self, inputs):
        logits = self.layer(inputs)
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(inputs, logits)

    def sample(self, num_samples):
        samples = tf.zeros((num_samples, self.dim), dtype=tf.float32)
        for i in range(self.dim):
            probs = self(samples)[:, i]
            samples_i = tfp.distributions.Bernoulli(probs=probs).sample()
            samples = tf.tensor_scatter_nd_update(samples, [[j, i] for j in range(num_samples)], tf.cast(samples_i, tf.float32))
        return samples

def load_mnist_data():
    """Load and binarize MNIST data."""
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0) > 0.5  # Binarize
    x_test = (x_test / 255.0) > 0.5
    x_train = x_train.reshape(-1, 784).astype(np.float32)
    x_test = x_test.reshape(-1, 784).astype(np.float32)
    return x_train[:10000], x_test[:1000]  # Subset for faster training

def train_fvsbn(model, data, epochs=200, batch_size=128, learning_rate=0.001):
    """Train the FVSBN model."""
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(10000).batch(batch_size)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                loss = model.compute_loss(batch)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset):.4f}")

def visualize_samples(samples, title="Generated MNIST Samples"):
    """Visualize generated samples as 28x28 images."""
    samples = samples.numpy().reshape(-1, 28, 28)
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(samples[i], cmap="binary")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load data
    dim = 28 * 28  # MNIST image size
    x_train, x_test = load_mnist_data()

    # Initialize and train model
    model = FVSBN(dim=dim)
    train_fvsbn(model, x_train)

    # Evaluate on a test sample
    test_sample = x_test[:1]
    log_prob = -model.compute_loss(test_sample) * dim
    print(f"Log probability of test image: {log_prob.numpy():.4f}")

    # Generate and visualize samples
    samples = model.sample(5)
    visualize_samples(samples)

"""# 5) Implement NADE and train it on a dataset like MNIST for image generation."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Force float32 globally
tf.keras.mixed_precision.set_global_policy('float32')
tf.config.optimizer.set_jit(False)

# --------------------------- Data Loader ---------------------------
def load_binarized_mnist(train_size=5000, test_size=500):
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 28*28)
    x_test = (x_test / 255.0).reshape(-1, 28*28)
    x_train = (x_train > 0.5).astype(np.float32)[:train_size]
    x_test = (x_test > 0.5).astype(np.float32)[:test_size]
    return x_train, x_test

# --------------------------- NADE Model ---------------------------
class NADE(tf.keras.Model):
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H
        self.W = self.add_weight(name="W", shape=(H, D), initializer="glorot_uniform")
        self.V = self.add_weight(name="V", shape=(H, D), initializer="glorot_uniform")
        self.c = self.add_weight(name="c", shape=(H,), initializer="zeros")
        self.b = self.add_weight(name="b", shape=(D,), initializer="zeros")

    @tf.function
    def call(self, x):
        batch_size = tf.shape(x)[0]
        a = tf.tile(self.c[None, :], [batch_size, 1])
        outputs = tf.TensorArray(dtype=tf.float32, size=self.D)

        for i in tf.range(self.D):
            h = tf.nn.sigmoid(a)
            logit = tf.einsum('bh,h->b', h, self.V[:, i]) + self.b[i]
            prob = tf.nn.sigmoid(logit)[:, None]
            outputs = outputs.write(i, prob)
            a = a + x[:, i:i+1] @ tf.transpose(self.W[:, i:i+1])

        return tf.transpose(outputs.stack(), [1, 0, 2])[:, :, 0]

    @tf.function
    def compute_loss(self, x):
        probs = self(x)
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, probs))

# --------------------------- Training ---------------------------
def train_model(model, x_train, epochs=30, batch_size=256, lr=2e-3, visualize_every=5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for epoch in range(epochs):
        losses = []
        for step, batch in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = model.compute_loss(batch)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(loss.numpy())

        print(f"Epoch {epoch+1}/{epochs} | Loss: {np.mean(losses):.4f}")

        # Optional visualization every few epochs
        if (epoch+1) % visualize_every == 0:
            visualize_samples(model)

def visualize_samples(model, num_samples=16):
    samples = model_sample(model, num_samples)
    plt.figure(figsize=(4, 4))
    for i in range(num_samples):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()

def model_sample(model, num_samples=16):
    D = model.D
    x = tf.zeros((num_samples, D), dtype=tf.float32)
    a = tf.tile(model.c[None, :], [num_samples, 1])

    for i in range(D):
        h = tf.nn.sigmoid(a)
        logit = tf.einsum('bh,h->b', h, model.V[:, i]) + model.b[i]
        prob = tf.nn.sigmoid(logit)
        xi = tf.cast(tf.random.uniform((num_samples,)) < prob, tf.float32)
        x = tf.concat([x[:, :i], xi[:, None], x[:, i+1:]], axis=1) # Corrected line
        a = a + xi[:, None] @ tf.transpose(model.W[:, i:i+1])
    return x.numpy()

# --------------------------- Main ---------------------------
x_train, x_test = load_binarized_mnist()
D = x_train.shape[1]

print("\nTraining NADE (Optimized)...")
nade = NADE(D=D, H=256)
train_model(nade, x_train, epochs=50, batch_size=256, lr=2e-3, visualize_every=5)

print("Final Sampling...")
visualize_samples(nade)

"""# 6) Implement MADE and train it on a dataset like MNIST for image generation."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Data ----------------
def load_binarized_mnist(train_size=5000, test_size=500):
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = ((x_train/255.0) > 0.5).astype(np.float32).reshape(-1, 28*28)
    return x_train[:train_size]

def visualize_samples(samples, title="Samples"):
    samples = samples.reshape(-1,28,28)
    plt.figure(figsize=(10,2))
    for i in range(min(8,samples.shape[0])):
        plt.subplot(1,min(8,samples.shape[0]),i+1)
        plt.imshow(samples[i],cmap="binary"); plt.axis("off")
    plt.suptitle(title); plt.show()

# ---------------- MADE ----------------
class MADE(tf.keras.Model):
    def __init__(self, D, H, seed=None):
        super().__init__()
        self.D, self.H = D,H
        rng = np.random.RandomState(seed)
        self.deg_input = np.arange(1,D+1)
        self.deg_hidden = rng.randint(1,D,size=H)
        self.deg_output = np.arange(1,D+1)

        self.W_in_hid = self.add_weight(name="W_in_hid", shape=(D,H),
                                        initializer="glorot_uniform")
        self.b_hid = self.add_weight(name="b_hid", shape=(H,), initializer="zeros")
        self.W_hid_out = self.add_weight(name="W_hid_out", shape=(H,D),
                                         initializer="glorot_uniform")
        self.b_out = self.add_weight(name="b_out", shape=(D,), initializer="zeros")

        self.mask_in_hid = (self.deg_input[:,None]<=self.deg_hidden[None,:]).astype(np.float32)
        self.mask_hid_out = (self.deg_hidden[:,None]<self.deg_output[None,:]).astype(np.float32)

    def call(self,x):
        h = tf.nn.relu(tf.matmul(x,self.W_in_hid*self.mask_in_hid)+self.b_hid)
        out = tf.matmul(h,self.W_hid_out*self.mask_hid_out)+self.b_out
        return tf.nn.sigmoid(out)

    def compute_loss(self,x):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(x,self(x)))

    def sample(self,n):
        x = np.zeros((n,self.D),np.float32)
        for i in range(self.D):
            probs = self(x)[:,i]
            x[:,i] = tfp.distributions.Bernoulli(probs=probs).sample().numpy()
        return x

# ---------------- Training ----------------
def train_model(model,x_train,epochs=3,batch_size=128,lr=1e-3):
    opt = tf.keras.optimizers.Adam(lr)
    ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size)
    for e in range(epochs):
        loss = np.mean([model.compute_loss(batch).numpy() for batch in ds])
        print(f"Epoch {e+1}/{epochs} - loss: {loss:.4f}")

# ---------------- Main ----------------
if __name__=="__main__":
    tf.random.set_seed(0); np.random.seed(0)
    x_train = load_binarized_mnist(train_size=2000)
    made = MADE(D=28*28,H=400,seed=2)
    train_model(made,x_train,epochs=2000,batch_size=128,lr=1e-3)
    visualize_samples(made.sample(8), title="MADE samples")

"""# 7) Implement a Vanilla GAN using TensorFlow or PyTorch and train it on a dataset like MNIST for image generation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Config
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 50
BATCH_SIZE = 256
NOISE_DIM = 100
LR = 2e-4
BETA_1 = 0.5

# Data
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, -1)
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

# Generator
def build_generator():
    return keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((7,7,256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(1, 5, strides=2, padding='same', use_bias=False, activation='tanh')
    ])

# Discriminator
def build_discriminator():
    return keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=(28,28,1)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

generator = build_generator()
discriminator = build_discriminator()

# Losses & Optimizers
bce = keras.losses.BinaryCrossentropy()
g_opt = keras.optimizers.Adam(LR, beta_1=BETA_1)
d_opt = keras.optimizers.Adam(LR, beta_1=BETA_1)

def d_loss(real_out, fake_out):
    real_loss = bce(tf.ones_like(real_out)*0.9, real_out)  # label smoothing
    fake_loss = bce(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss

def g_loss(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)

# Training step
@tf.function
def train_step(real_imgs):
    noise = tf.random.normal([real_imgs.shape[0], NOISE_DIM])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake_imgs = generator(noise, training=True)
        real_out = discriminator(real_imgs, training=True)
        fake_out = discriminator(fake_imgs, training=True)
        gl = g_loss(fake_out)
        dl = d_loss(real_out, fake_out)
    g_grads = gt.gradient(gl, generator.trainable_variables)
    d_grads = dt.gradient(dl, discriminator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    return gl, dl

# Plotting generated images
seed = tf.random.normal([25, NOISE_DIM])
def sample_images(epoch):
    preds = generator(seed, training=False)
    plt.figure(figsize=(5,5))
    for i in range(25):
        plt.subplot(5,5,i+1)
        img = (preds[i] + 1) / 2.0
        plt.imshow(img.numpy().squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.show()

# Training loop
def train():
    for epoch in range(1, EPOCHS+1):
        for real_batch in train_ds:
            gl, dl = train_step(real_batch)
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f'Epoch {epoch} - G Loss: {gl:.4f}, D Loss: {dl:.4f}')
            sample_images(epoch)

# Run
train()
print("Training completed!")

"""# 8) Implement Progressive GAN and train it on a dataset like MNIST

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LATENT, BATCH, EPOCHS = 100, 64, 2
STAGES = [8, 16, 32, 64]

(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.astype('float32') / 127.5 - 1)[..., None]

def G(r):
    m = tf.keras.Sequential([tf.keras.layers.Input((LATENT,)),
                             tf.keras.layers.Dense(4*4*128),
                             tf.keras.layers.Reshape((4,4,128))])
    c, f = 4, 128
    while c < r:
        m.add(tf.keras.layers.UpSampling2D())
        f = max(16, f//2)
        m.add(tf.keras.layers.Conv2D(f, 3, padding='same'))
        m.add(tf.keras.layers.LeakyReLU(0.2))
        c *= 2
    m.add(tf.keras.layers.Conv2D(1, 3, padding='same', activation='tanh'))
    return m

def D(r):
    m = tf.keras.Sequential([tf.keras.layers.Input((r,r,1))])
    f, c = 32, r
    while c > 4:
        m.add(tf.keras.layers.Conv2D(f, 3, padding='same'))
        m.add(tf.keras.layers.LeakyReLU(0.2))
        m.add(tf.keras.layers.AveragePooling2D(2))
        c //= 2
        f = min(256, f*2)
    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(1, 'sigmoid'))
    return m

bce = tf.keras.losses.BinaryCrossentropy()

def train_step(G, D, real, g_opt, d_opt):
    z = tf.random.normal([real.shape[0], LATENT])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake = G(z, training=True)
        r_logit, f_logit = D(real, training=True), D(fake, training=True)
        g_loss = bce(tf.ones_like(f_logit), f_logit)
        d_loss = (bce(tf.ones_like(r_logit), r_logit) +
                  bce(tf.zeros_like(f_logit), f_logit)) / 2

    g_grads = g_tape.gradient(g_loss, G.trainable_variables)
    d_grads = d_tape.gradient(d_loss, D.trainable_variables)

    g_opt.apply_gradients(zip(g_grads, G.trainable_variables))
    d_opt.apply_gradients(zip(d_grads, D.trainable_variables))

    return float(g_loss), float(d_loss)

for r in STAGES:
    print(f"\nTraining {r}x{r}")
    G_model, D_model = G(r), D(r)

    # Create new optimizers for each stage
    g_opt = tf.keras.optimizers.Adam(2e-4, 0.5)
    d_opt = tf.keras.optimizers.Adam(2e-4, 0.5)

    ds = tf.data.Dataset.from_tensor_slices(
        tf.image.resize(x, [r, r]).numpy()
    ).shuffle(10000).batch(BATCH)

    for epoch in range(EPOCHS):
        g_losses, d_losses = [], []
        for batch in ds.take(100):
            g_loss, d_loss = train_step(G_model, D_model, batch, g_opt, d_opt)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
        print(f"Epoch {epoch+1}: G={np.mean(g_losses):.3f} D={np.mean(d_losses):.3f}")

    # Generate samples
    samples = (G_model(tf.random.normal([9, LATENT]), training=False).numpy() + 1) / 2
    fig, axes = plt.subplots(3, 3, figsize=(4, 4))
    for img, ax in zip(samples, axes.flat):
        ax.imshow(img[..., 0], cmap='gray')
        ax.axis('off')
    plt.suptitle(f"{r}x{r}")
    plt.show()

"""# 9) Implement a style transfer algorithm using GANs.

1.   Go to Kaggle and Login your account
2.   Go to Settings and Create New Token under API section
3.   It will download kaggle.json and upload it by running the first cell in this section
4.   It makes the session to download the dataset `pix2pix-dataset`

"""

!pip install -q kaggle
from google.colab import files
files.upload()  # upload kaggle.json (from your Kaggle account)
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d vikramtiwari/pix2pix-dataset
!unzip -q pix2pix-dataset.zip -d /content/

import tensorflow as tf,os,matplotlib.pyplot as plt

IMG,BATCH=256,4
images_dir='/content/facades/facades/train'

def split(p):
    i=tf.image.decode_jpeg(tf.io.read_file(p),3)
    w=tf.shape(i)[1]//2
    a=tf.image.resize(tf.cast(i[:,w:,:],tf.float32)/255.,[IMG,IMG])
    b=tf.image.resize(tf.cast(i[:,:w,:],tf.float32)/255.,[IMG,IMG])
    return a,b

paths=[os.path.join(images_dir,f) for f in os.listdir(images_dir)]

ds=tf.data.Dataset.from_tensor_slices(paths).map(split).shuffle(512).batch(BATCH).prefetch(tf.data.AUTOTUNE)

def G():
    i=tf.keras.Input([IMG,IMG,3])
    x=tf.keras.layers.Conv2D(32,4,2,'same',activation='relu')(i)
    s1=x

    x=tf.keras.layers.Conv2D(64,4,2,'same',activation='relu')(x)
    s2=x

    x=tf.keras.layers.Conv2D(128,4,2,'same',activation='relu')(x)

    x=tf.keras.layers.UpSampling2D(2)(x)
    x=tf.keras.layers.Conv2D(64,3,1,'same',activation='relu')(x)
    x=tf.keras.layers.Concatenate()([x,s2])

    x=tf.keras.layers.UpSampling2D(2)(x)
    x=tf.keras.layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x=tf.keras.layers.Concatenate()([x,s1])

    x=tf.keras.layers.UpSampling2D(2)(x)
    o=tf.keras.layers.Conv2D(3,3,1,'same',activation='tanh')(x)

    return tf.keras.Model(i,o)

def D():
    a=tf.keras.Input([IMG,IMG,3])
    b=tf.keras.Input([IMG,IMG,3])
    x=tf.keras.layers.Concatenate()([a,b])

    x=tf.keras.layers.Conv2D(32,4,2,'same')(x)
    x=tf.keras.layers.LeakyReLU(0.2)(x)

    x=tf.keras.layers.Conv2D(64,4,2,'same')(x)
    x=tf.keras.layers.LeakyReLU(0.2)(x)

    x=tf.keras.layers.Conv2D(128,4,1,'same')(x)
    x=tf.keras.layers.LeakyReLU(0.2)(x)

    o=tf.keras.layers.Conv2D(1,4,1,'same',activation='sigmoid')(x)

    return tf.keras.Model([a,b],o)

gen,disc=G(),D()

bce=tf.keras.losses.BinaryCrossentropy()
mae=tf.keras.losses.MeanAbsoluteError()

gopt=tf.keras.optimizers.Adam(2e-4,0.5)
dopt=tf.keras.optimizers.Adam(2e-4,0.5)

@tf.function
def step(inp,real,lam=100.0):
    with tf.GradientTape() as td:
        fake=gen(inp,training=True)
        ro=disc([inp,real],training=True)
        fo=disc([inp,fake],training=True)
        dl=bce(tf.ones_like(ro),ro)+bce(tf.zeros_like(fo),fo)

    dopt.apply_gradients(zip(td.gradient(dl,disc.trainable_variables),disc.trainable_variables))

    with tf.GradientTape() as tg:
        fake=gen(inp,training=True)
        fo=disc([inp,fake],training=False)
        gl=bce(tf.ones_like(fo),fo)+lam*mae(real,fake)

    gopt.apply_gradients(zip(tg.gradient(gl,gen.trainable_variables),gen.trainable_variables))

    return dl,gl

it=iter(ds.repeat())
# Train longer
for s in range(5000):  # instead of 200
    x, y = next(it)
    dl, gl = step(x, y)
    if s % 500 == 0:
        print(f"step {s} d:{dl.numpy():.4f} g:{gl.numpy():.4f}")
# After training
x, y = next(iter(ds))
f = gen(x, training=False)

# Show only input and translated (fake) image
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(x[0])
plt.axis('off')
plt.title('Input')

plt.subplot(1, 2, 2)
plt.imshow(f[0])
plt.axis('off')
plt.title('Translated')

plt.tight_layout()
plt.show()

import tensorflow as tf, os, matplotlib.pyplot as plt

IMG, BATCH = 256, 4
images_dir = '/content/facades/facades/train'

def split(p):
    i = tf.image.decode_jpeg(tf.io.read_file(p), 3)
    w = tf.shape(i)[1] // 2
    a = tf.image.resize(tf.cast(i[:, w:, :], tf.float32)/255., [IMG, IMG])
    b = tf.image.resize(tf.cast(i[:, :w, :], tf.float32)/255., [IMG, IMG])
    return a, b

paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]

ds = tf.data.Dataset.from_tensor_slices(paths).map(split).shuffle(512).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# GENERATOR (unchanged)
# ---------------------------
def G():
    i = tf.keras.Input([IMG, IMG, 3])
    x = tf.keras.layers.Conv2D(32, 4, 2, 'same', activation='relu')(i)
    s1 = x
    x = tf.keras.layers.Conv2D(64, 4, 2, 'same', activation='relu')(x)
    s2 = x
    x = tf.keras.layers.Conv2D(128, 4, 2, 'same', activation='relu')(x)

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu')(x)
    x = tf.keras.layers.Concatenate()([x, s2])

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, 1, 'same', activation='relu')(x)
    x = tf.keras.layers.Concatenate()([x, s1])

    x = tf.keras.layers.UpSampling2D(2)(x)
    o = tf.keras.layers.Conv2D(3, 3, 1, 'same', activation='tanh')(x)
    return tf.keras.Model(i, o)

# ---------------------------
# PATCHGAN DISCRIMINATOR
# ---------------------------
def D():
    inp = tf.keras.Input([IMG, IMG, 3])
    tar = tf.keras.Input([IMG, IMG, 3])

    x = tf.keras.layers.Concatenate()([inp, tar])

    # PatchGAN: progressively downsample
    x = tf.keras.layers.Conv2D(64, 4, 2, 'same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, 4, 2, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, 4, 2, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(512, 4, 1, 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # PatchGAN output — NOT a single scalar, but an N×N map
    o = tf.keras.layers.Conv2D(1, 4, 1, 'same', activation='sigmoid')(x)

    return tf.keras.Model([inp, tar], o)

# ---------------------------
# LOSSES & TRAINING
# ---------------------------
gen, disc = G(), D()
bce = tf.keras.losses.BinaryCrossentropy()
mae = tf.keras.losses.MeanAbsoluteError()
gopt = tf.keras.optimizers.Adam(2e-4, 0.5)
dopt = tf.keras.optimizers.Adam(2e-4, 0.5)

@tf.function
def step(inp, real, lam=100.0):
    with tf.GradientTape() as td:
        fake = gen(inp, training=True)
        ro = disc([inp, real], training=True)
        fo = disc([inp, fake], training=True)
        dl = bce(tf.ones_like(ro), ro) + bce(tf.zeros_like(fo), fo)
    dopt.apply_gradients(zip(td.gradient(dl, disc.trainable_variables), disc.trainable_variables))

    with tf.GradientTape() as tg:
        fake = gen(inp, training=True)
        fo = disc([inp, fake], training=False)
        gl = bce(tf.ones_like(fo), fo) + lam * mae(real, fake)
    gopt.apply_gradients(zip(tg.gradient(gl, gen.trainable_variables), gen.trainable_variables))
    return dl, gl

# ---------------------------
# TRAIN LOOP
# ---------------------------
it = iter(ds.repeat())
for s in range(1000):  # shorter demo training
    x, y = next(it)
    dl, gl = step(x, y)
    if s % 200 == 0:
        print(f"step {s}: D={dl.numpy():.4f}, G={gl.numpy():.4f}")

# ---------------------------
# TEST AND SHOW ONLY INPUT + FAKE
# ---------------------------
x, y = next(iter(ds))
f = gen(x, training=False)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(x[0])
plt.title('Input')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(f[0])
plt.title('Translated (Fake)')
plt.axis('off')
plt.show()

"""# 10) Implement a basic transformer model using PyTorch or TensorFlow and train it on a text dataset like WikiText-2 for language modeling."""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

texts = [
    "machine learning is fascinating and powerful",
    "deep learning is a subset of machine learning",
    "neural networks can learn complex patterns",
    "transformers are great for natural language processing",
    "language models can generate realistic text"
]

corpus = " ".join(texts)
tokens = corpus.lower().split()
vocab = sorted(set(tokens))
word_index = {w: i + 1 for i, w in enumerate(vocab)}
index_word = {i + 1: w for i, w in enumerate(vocab)}
vocab_size = len(vocab) + 1

input_sequences = []
target_sequences = []
for i in range(1, len(tokens)):
    input_sequences.append([word_index[w] for w in tokens[:i]])
    target_sequences.append([word_index[w] for w in tokens[1:i+1]])

max_len = max(len(s) for s in input_sequences)
X = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len, padding="pre")
y = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_len, padding="pre")

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, max_len, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(max_len, embed_dim)

    def call(self, x):
        positions = tf.range(tf.shape(x)[-1])
        return self.token_emb(x) + self.pos_emb(positions)

def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(embed_dim)(ffn)
    return layers.LayerNormalization(epsilon=1e-6)(x + ffn)

def build_transformer_lm(vocab_size, seq_length, embed_dim=64, num_heads=2, ff_dim=128, num_layers=2):
    inputs = layers.Input(shape=(seq_length,))
    x = PositionalEmbedding(vocab_size, seq_length, embed_dim)(inputs)
    for _ in range(num_layers):
        x = transformer_block(x, embed_dim, num_heads, ff_dim)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

model = build_transformer_lm(vocab_size, X.shape[1])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X, y, epochs=100, batch_size=16, verbose=1)

def generate_text(seed_text, num_words=10):
    for _ in range(num_words):
        token_list = [word_index.get(w, 0) for w in seed_text.split()]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=X.shape[1], padding="pre")
        preds = model.predict(token_list, verbose=0)
        predicted = int(np.argmax(preds[0, -1, :]))
        seed_text += " " + index_word.get(predicted, "")
    return seed_text

print("\n--- Text Generation ---")
print(generate_text("machine learning is", num_words=25))

"""# 11) Fine-tune a pre-trained GPT model on a specific task such as sentiment analysis using a dataset like IMDB reviews."""

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2ForSequenceClassification
texts = [
 "I love this movie, it was fantastic!",
 "Absolutely wonderful experience, would recommend!",
 "Terrible movie, I hated every moment.",
 "Worst film ever, waste of time.",
 "The acting was brilliant and touching.",
 "Awful plot and bad direction.",
 "A great story and strong performances.",
 "It was boring and too long."
]
labels = [1,1,0,0,1,0,1,0]
trn_texts, tst_texts = texts[:6], texts[6:]
trn_labels, tst_labels = labels[:6], labels[6:]
mname = "distilgpt2"
tok = GPT2Tokenizer.from_pretrained(mname)
tok.pad_token = tok.eos_token
def enc(txts):
    return tok(txts, truncation=True, padding=True, max_length=64, return_tensors="tf")
trn_enc = enc(trn_texts)
tst_enc = enc(tst_texts)
trn_ds = tf.data.Dataset.from_tensor_slices((dict(trn_enc), trn_labels)).shuffle(6).batch(2)
tst_ds = tf.data.Dataset.from_tensor_slices((dict(tst_enc), tst_labels)).batch(2)
model = TFGPT2ForSequenceClassification.from_pretrained(mname, num_labels=2, from_pt=True)
model.config.pad_token_id = tok.eos_token_id
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
model.fit(trn_ds, validation_data=tst_ds, epochs=3)
print("Eval:", model.evaluate(tst_ds))
def predict(text):
    e = tok(text, return_tensors="tf", truncation=True, padding=True, max_length=64)
    logits = model(e).logits
    p = int(tf.argmax(logits, axis=-1).numpy()[0])
    return "Positive 😀" if p==1 else "Negative 😠"
for ex in ["The movie was absolutely amazing!","It was dull and disappointing.","The plot was okay, but the acting was weak."]:
    print(ex,"->",predict(ex))

"""# 12) Utilize the OpenAI API to build a question-answering application powered by GPT-3, allowing users to input questions and receive relevant answers.
# NOT FINALIZED YET (NOT PLAYGROUND VERSION)
1.   Go to OpenRouter and Login your account
2.   Go to Settings and Create API key under API keys section
3.   Copy the key and Make a secret key under the name "OPENROUTER-API-KEY" and value is copied key value.
4.   Enable the Notebook Access.


 ![adding key in Secrets .png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmMAAAJYCAYAAAAudeETAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAP+lSURBVHhe7P1/XJR1vvj/PwQHZUAdNUbM0RxpRy3GjqMrulGu9AtL3A2zhLNiR07fIuuQHbPl+E43+5jlsiInzfbt0nfB/Ugd093C3aiz0WGbtugYnoROOq2O6Zg4rDEpjMgIfv64ZoaZiwEGRDF73m+3uSnX9bp+va7Xdc1zrtfrer0G3HzzzRfwamtro7W1lQEDBjBgwADfZCGEEEIIcYlEqCcIIYQQQojLZ0CoJ2PXX389gwYNCk4phBBCCCH6XMhgTKophRBCCCEuD6mmFEIIIYToRxKMCSGEEEL0IwnGhBBCCCH6kQRjQgghhBD9SIIxIYQQQoh+JMGYEEIIIUQ/kmBMCCGEEKIfSTAmhBBCCNGPJBgTQgghhOhHEowJIYQQQvQjCcaEEEIIIfqRBGNCCCGEEP1IgjEhhBBCiH4kwZgQQgghRD+SYEwIIYQQoh9JMCaEEEII0Y8kGBNCCCGE6EcSjAkhhBBC9CMJxoQQQggh+tGAm2+++YLvj7a2NlpbWxkwYAADBgwITimEEEII0YkpN01h/k9/wrBhw/o0hrhw4QLffvstb/3hTfZ/tl89+6ogT8aEEEIIcVFGjBzB3LvvRqfT9WkgBjBgwAB0Oh1z776bESNHqGf3qTEGAw8ufZAxBoN6Vpd6u5xP5Lhx437h++PChQtcuHBBnowJIYQQImzjxo1j+owfotFoOP3taZz1Tk6fPt3t57znPFqtVr26kCIiIjj05d84deqUelafGGMwkJGZwYSEBIxGI199dZQzp0+rk3XQ2+UCSTWlEEIIIS7KxIkT+dmSxURHR/N57ef8/4teVScJKe0n85n949nqySGdPXuW3xVv5+DBg+pZfeK68ddx/6IHGDVqFAB1J+oo3VHKcYdDndTPF4jFj44H4OTJk/zHa6/z1ZGv1Em7JNWUQgghhPje++rIV+z43Q7qTtQBED86nozMjE6rHtWBWN2JOnb8bkePAzH6OhiLiIjg/kUP8GL+Bjb86pdkP/TPxMbGqpMJIYQQQlxxjjsclO4o7TYgCxWIdfcUrSt9Wk05J2UOd81NZeDAgeBtg1b18ce88R9vcP0PrmdRRgaDBg9SL9at1tZWyv9UzscffaSeJYQQQoh+djVUUwbqKtjqal5v9emTsZHXjPQHYnjfgBg+XHnzITIikkGDBxEdHd2rjyZKE7AlIYQQQnzXfXPqFMeOHevyc/rbnjWG7wudPSG7MTGxzwMx5MlYMJPJxJ133sl1113HgAEDcLlcfPDBB1RWVqqTCiGEEMKrt0/GwvFP2Uu5MfHGy/pkzEf9FKy1tZXIyEjow0CMvg7GIiIiuO/+hUybPo0BAwZw8MBBXi99jcbGRnXSK87ChQtJS0sjOjo6aHprayuffPIJW7du5ezZs0HzhBBCCHH1BmOECMjo40CMvq6mbGtr4z9ee52nV6xk5b8+RdG233wnArGbb76Zu+++m+joaM6ePcu+ffv461//yt///nciIiKYMWMG999/v3qxK15BQQE7d+5k9erV6llCCCGEuEL0aTD2XXXjjTcSExOD2+3md7/7Hc8//zwFBQWsW7cOh8NBY2MjERGSVUIIIcT3ifqpWGtrK3TxlmVv9XkP/KNGjeLHKXNITLyR5rPNuFwudZIrzowZM5gwYQKNjY1UVFRw8uRJAE6fPs3evXt56623+O///m/1YkydOpX58+dz7733Mm3aNEaMGMGxY8c4f/68OinR0dGkpqbyk5/8hLvvvpvJkyczYMAAjh8/7k9jNpuZPHky11xzDS6Xi5/+9KfcddddTJ06ldOnT/t7He5uu7713HjjjQwZMoSGhgba2towGAy43W7Onj1LdHQ0KSkp/OQnP+Gee+4JuT9CCCFEOK655hqm/MNNaDQa6p31/M++feokvTbVMhW9Xs/58+fZ/9n+S9YDv5o6EKs7UceunbsYPXo0sUNiiR0S2+se99X6tM3YGIOBJQ9mMWLkSACampp44z92UrO/hqioKMZdN47ICKXhW0+0trVRd+LEJavyfOCBB/jpT39KREQEn332Gb/+9a+7PNnR0dHk5OQwY8YMf0M+vMHsiRMn+O1vf8u+gIJ46623kpmZyUhvvgSm/+KLL9i2bRsOh4PVq1djNpv5+uuvaWlp8b9IcPbsWYqKivjkk0/C2q5vPWq+9Rw6dIjc3Fz/+n1aW1v5n//5HwoLC6V9nBBCiLBdbW3GQgVil7Jriz59MvbjOT9m8g2T/X9HRUWh0Wj4n337uP7668l6cAlJs2Yybfq0Hn2mWqZy5swZjn7V815tw9HQ0MANN9zA8OHDGT16NHfccQdmsxmNRsPhw4fVyXnkkUf40Y9+BMCRI0fYv38/DQ0N6HQ6RowYwbhx4/jf//1fTp8+zdSpU/mnf/onRo4cicfj4dChQxw6dMjfZUdcXBzXXHMNH374IbNnz2bUqFHExMQwbNgwvv32W/7+97/T1tbG/v37ufvuu8Pa7ujRoxk4cCCDBg1Sxgk7fZq6ujq++eYbPvvsM+bOnYvZbFZ+Zezfz2effUZbWxsjR45k1KhRREZGUlNTozpqIYQQIrSr6clYd8HWmdOn+eqroxiNxj57QtanwdgNN97AuOvGBU37e/3f2Ve9L+hE9dSFCxew2WyXLBg7c+YM1dXVjBgxgtGjRxMVFYVer2fatGnceeed/u0DJCcnM2/ePDQaDVarleeff55PPvmEDz74gKamJiZPnsyIESM4d+4cNTU1PPjggyQkJOB2u9m+fTvbtm3jr3/9K++99x7XX389Q4cOxeFwcPToUSwWi39MrNraWlavXs0f//hH/vjHPzJu3Liwt/vb3/6Wd999l1tvvZWhQ4dis9lYtWoVFRUVnDhxgjvvvJNRo0Zx+PBhfvGLX7Bv3z4qKiqIjo5mz549lJeXq3JICCGE6NzVEox1F4j59HVA1qfB2JkzjUycaCLaOwJ7U1MTf373P3GedHLmzBmO2O3s+3Qf1Xs/7dFn795P+ZvtS1paWtSb7DNnz57l448/pqqqigEDBjBs2DC0Wi3R0dHceOONREVFUVtbyx133MGkSZM4c+YMH3/8MaNHj2b8+PGMHz+e1tZWxo0bh06n49y5c9TX15OamopWq+XTTz/ld7/7nX9758+f5y9/+Qt/+MMf+Pjjjzlz5oz/ydjp06d57bXXOHr0qD99T7b7l7/8BYDU1FSGDh2K0+kM6itt6tSpjB07Fp1Ox5QpU7hw4QInT57k008/lTZjQggheuxqCMbCDcR8+jIg69M2Y3gb8E+f8UOiNBr2Ve/jyJEj6iTfGT/96U9JS0tj6NCh1NfXs2nTJu6//35uuukmddIOampqqKio4KGHHiIqKoo//OEPvP766+pkQXxtverq6li/fj1ff/21f97/+T//J+ztrl27FrxdWxgMhqBpAAaDgWXLlpGQkOA/z62trRw/fpw//vGPVFRU+NMKIYQQ3QlsM3b629N8e/pbdZJeGzZ0GEOHDb3kbcauG38d9y96gFGjRnUbiAUKDOJOnjzJf7z2eo8HC+/TJ2N4n4Z9abNx4IsD34k3KfE2yL/jjjuIi4vDEZDxBw4cQKfTcf311xMZGcmXX37JddddR3x8PA0NDXzyySd89dVXIT82m42zZ88ybdo0IiMjsdls1NbWBm1XzfdkzOVy8cYbbwTNu/XWW8Perq+gdvZk7PTp07z33ns0NDSg0WiIjo5m0KBB6HQ6/uEf/oHIyEg+//zzgK0LIYQQnbvABcxmMzExMQwaPIhhw4b12cc3co+rQRkV51K9YPat61u+OvIVQ4bE8uYf3gorECPgCdmQIbG8+fs3OXb0mDpJt/o8GPuuMZvNrFq1iuTkZIYPH87HH38c1DXFDTfcwMSJE2lra2Pfvn1ERUUxYcIELly4wJ/+9CfeeOMNPvnkE//n5MmTWK1Wamtr0Wg0TJ8+nSFDhnD+/PkOwyqZTCYmTJjgfwIWWE35zjvvBKX9wQ9+EPZ2fToLxgAmTZrEoUOHeOedd3jrrbf46quvmDRpErGxsQwcOFCejgkhhAjb2bNnOXPmNNeNH8+gQYP6NIa4cOEC3377LX8s28MR+6WtbTtz+jT/s+9/elzN2NvlfPq0mvK7OhzS008/zbRp07hw4QIffvgh27Zt4+zZs5hMJh5++GHGjRvHN998w+bNmxk4cCDLli1j2LBhHDp0iM2bN/ufpt1999088MADnD9/nt///vfs2bOHJ554gh/96EecP3+eP/3pT/52Y751jx07liNHjrBlyxaWLFmC2WzG4XCwfPnyoH2cOnVqj7ZLQDWl3W5nzZo1jBw5kjFjxpCens51110X1I1FdHQ0/+f//B9MJhOHDx/m6aefDtq+EEIIIS6NPg3GvqsDhU+dOpWHHnqIuLg4AFpaWmhtbSUqKorIyEja2tqorKzk5ZdfBiArK4u5c+cycOBAf0N9jUZDXFwcERER1NfXs3XrVmpqajAYDDzxxBNcd911XLhwgdOnT9PU1ERcXBwajYbW1lYqKyvZunWrv81YqGCsp9slIMgcMGAAHo8Hj8fD7373O2bOnOnvh+zMmTM0NDQwZMgQhg8fzoULF3j33XcpKipSbV0IIYQQl0KfVlNO/+F0xl13nf/vAQMG0NTkpvrTT4mLi8My3UJ0dDQajaZHn8jISA4dOnTJuraoq6vjyy+/ZOzYsQwfPpyBAwei0WiIiIjA7Xbz3nvvsX37dn/15WeffaZ0YjtuHNHR0QwdOpTY2FgA7HY7v/nNb/zVhadPn+bIkSOMHTuWESNGEB0dzZAhQ4iMjMTtdvPnP/+Z3/zmN9BNNSU93C7AV199xYQJExgxYgSRkZF4PB6qqqr4r//6L6699lri4uKIjo5Gp9MRHR3N+fPn+eijjyguLg45ioAQQggh+p48GVNJSEjw99lls9n47LPPOm0sGB0dzU033YTJZOKbb77hiy++4NChQ+pkfj1Zd1d6s93Ro0dz5MiRoBcUDAYDkydPZsyYMdjtdmpray/ZK8NCCCGECK1Pg7HvapsxIYQQQoj+0qfBmBBCCCGE6JkI9QQhhBBCCHH5SDAmhBBCCNGPJBgTQgghhOhHEowJIYQQQvQjCcaEEEIIIfqRBGNCCCGEEP1IgjEhhBBCiH4kwZgQQgghRD+SYEwIIYQQoh9JMCaEEEII0Y86BGMXLvhHRxJCCCGEEJdYh2BMCCGEEEJcPhKMCSGEEEL0IwnGhBBCCCH6kQRjQgghhBD9SIIxIYQQQoh+JMGYEEIIIUQ/kmBMCCGEEKIfSTAmhBBCCNGPJBgTQgghhOhHA26++WZ/l/ttbW2cP3+eiIgIBgwYEJyylyIiIoiIiCAyMtK/Xt8Hb4//vk9bWxutra20tbXR1tamXpUQQgghxFXnkjwZGzBgABqNhujoaLRaLYMHD0aj0QQFZIFpfcGaRqNh8ODBaLVaoqOj0Wg0fRYUCiGEEEJcifo0GIuIiGDQoEHExMQwaNAgIiMj1UnCFhkZGbSuiIg+3VUhhBBCiCtCn0U4UVFRaLVaNBqNetZF02g0aLVaoqKi1LOEEEIIIb7TLjoYi4iIIDo6+rIESlFRUURHR8tTMiGEEEJcNS4qqhk4cCBarfaiqiN7KjIyEq1Wy8CBA9WzhBBCCCG+c3odjPka2/cX30sBQgghhBDfZb0KxjQaDYMGDVJPvuwGDRokAZkQQgghvtN6HIwNHDjwigjEfAYNGiRVlkIIIYT4zupRMBYREdGvVZOdGTx4sDTqF0IIIcR3Uo8imCvpiZjalbxvQgghhBCdCXs4pKioqLC7r9BoNMz60SzGXXcd0dpo9eyQTn97mj1vldHU1KSeFbaWlhZaWlrUk4UQQgghrlhhBWMRERFotVr/312JiYkhc/E/MmbMGPWsLl24cAHbQRuvl752UeNSut3ui1peCCGEEOJyCquasidvLJomTkSv16snd2vAgAH8wPQDUu+eq57VIz3ZVyGEEEKI/tbtk7EBAwYQExMTuEyXkmbN5I4772DgwIH89cO/8m75O+okHcy6+UfcdvttDBw4kPPnz/Pen9/jow//qk4WtqamJi5c8B+WEEIIIcQVq9snY5ej24iPPvwr1Z9W09bWxsCBA7nl1luYfMNkdbKwXY59FkIIIYToC1dEMAZQ/qe3+dL2JRcuXECr1XJX6l2MvvZadbKwXK597hPTs8jbVMyO7TvYUbyOLMCyOI8Na7JJjlcnFley5IfXUbBxBenqGT01O4d1G9eRM1s947tOT+pj69jwZBo9b8jwHTE9i7wXVpM9XT1DXH69LG89uP7Snyqg4KmLvuIvgp7UxzawrXgHO7bvYMuTyRCfSu5zG1hxn9GbJpmc5wpY93CyatkrV//n6+XXZTAWERFx2cadbGtr460/vMnXX38NgG74cObNn9ejKlKfyMjInvU7lpBKzpoNbClSCvSOoi1sWJNDaoI6YV9LIe/hNCzx0FB/kpNN59CQxrzbLRgTb2Pud+faEcDweAOGsWMYpZ7RU0PiMIw1EDdEPeO7bgF3zTZhnJXCAvWsq0TaPbdhSTBz6z1p6lnisutleevB9TdqjAHDmIu+4ntNv2QFWbONxLSc5GT9SZqIhfS7SJ5kJCllPiYAhhM3zoAhfrh68StWf+drf+gyYulRQNMHmpqa2PPWHlwNDQBce+21zP/pT3q1H+Euo5+fx7bnsklJNBLrUQr0SU8sxsQUstdtY7X/18UlMMWCQQeu/cUse3I5yx9dSxFl7PlzNbZ95ex6Q72AEJdH6tNb2LF9A9nqGRdlF+9U2rBVvsMu9ayrRNkf36P6QDXv/bFMPev7Y24eW7bvYMND6hmX29Vf3m4zGdC02nnvmeUsf3I5KzeWw+53sB6wYf1DKTb1AleaK6as9L8uI5bL9VQs0Imvv+ad8ndwu90MGDCAUfGjGHnNSHWyboW179NzyFtkQXfWRtmmZSx5RCnQyx9ZwrJN5djP6TCn/ws5U9QL9pFxWmKBRldF0OTq7etZ9XwJ1UFThbh8NINj0QzWEF7PguFyUr55Fas2l+NUz7pa7C1h/TPrKdmrnvE9EqkhdrCG/n+x/eovb7oYDZxrxFkXMLGunMJnVlH47nfgqK+YstL/IseNG/cL3x8XLlygra2NAQMGMGDAAKKiosJ+wuRjGGsgISGBiIgIjh07xqG/HQqaf9M/3MTd99zD7XfezoykGYwdN45vTp2isbHRn+bv9X/HfNMUYmNjaW5upmb/ftxN7qD1hOP8+fPqSUHSc57k5lGnqNr0r2ytCu5stunYPv5z4D9w7xQjQ6nkT5/45usx3z6fBXffxs0zZmK5PoZTNXZcgQsnWEieMprIr1oZO38RGXfcyswZiejdx7E5lfUYpyeTOHYS0yaOpLXuBKejxjFufMDn2mE0O5wE7pVxTgYZ827j5hkWJsScovbIcCyzExk98CgnGnzbHcewc8dxtmdnx+kB+5f4QBb3zrmZCTFV1B7xpU8h4/753DZrJjOnTiDmnBO7d787MmKZnUhCXCRHvw7KBf+8cUOaOe5dXp+Ywvx7A9btqsWuPAhVqPe1s+ndHUMI+sQ0Ft0/l1tndLJtlPWm33s/d948s8M5CxJvJu3+DObe3L6uqCn3cpP+LPad7xD4XRy03cQxeE5/oZyvzphmc+8/jKDhsz/wRVT7ssFlrWf5rtZelpT1nqu3e/NbWfY60zQmjmzF6TxNxPhriPzqhL+Md3sO480kTTMy4txxYm7JZknqrSTGu9l30KmU+2vb99n/d9vY9vxM1OM+YQs+/971qtMwPompCSM6XCsddFemu71m9ZhnTcU0tuN1qRyviVEDj3JiUPuxOxu7zguFEcu9C7j/9luZOWMmifFujh9UXfdh5ZGyf0ZdM8eHJHmP1XefcKnybjjOfX8LvmfRF3kU+r52je/+FCjBQvKUhJDzjNOTSbwuIJ8D9yvUNdlFHqvLmy+9Py/899KA+WFdf4ofpt6PETs7y4OueMz+/JlJ4hgPrgPt149P59dgGLxl7vrEmRgGfcvRhhaGBH5/jA/M94nMvvcmRjR8xh8qA5+VGUnJzGD+7JuVfBh6Dufhrq+j9vwcHrBsiHPiSx9wjOoy21VZac/XU+3bCXWvIYxy24mLuY9dCl12baHVanscjHXWtUVERATp9y3ghhtv6LDO5uZmKv78Hp9UfeKf9siyHOLj43G5XPy/239HvbM+aJnutLW14XZ3FcBlsG57Oqa6cpY9VdTJL6cUMh4zo7GVUvKuE+JTWbEqi6R4DbR68LR5I3q3nfKXVlLkux4fLWDnnChsB2IxTdLiafagGawB3Nh+/xSrdjjJ2bSTlK76xXXXULJkLUplh4Ws53JIm6QDwOMBjQZc+2tomWKm5f2FLH/Zt90R1BQvYe2egHWpp3v3z35oOMYE5SeJw7sOy+J15NxjQhdJwH57cLyfz/KXQz2rM5P70mqSdTZKF69id8Ac/dINbJlrxL5nISuL9aQ+uYasWXo0BKy71Y393UJWvupdt3pffdTTuziGjlTb9uYfrS5qdqxi7VvK2Q889vZEHpx7S3n2xbL2MjI9mw2Pp2LUKuk8ERo0bS6qD7ZgSWyhYuFytqq32+pRVjlY02G7HcxbTfGSBE7WfoMh0aAsiwZNJHhOVVP6i/WU1YWb7wEzlLmkPbOOrCm6TspwDgU7UzAELePwHlOY59C3/4daMCYoZdZdW8KSZ8uUck8FC59Qcihn005SIm3YhpowDQrcHxu7n15FqfcXv/7OFax5MAm99wQqee6g5ugIzKMPBVwrHYVVpsO4ZjPW7SA9wUH5EyspCngSoX+0gC1zYql++SHWxyjHfshXTrvIC6Znse7hNEw6lHPhO8f11ZSuXU+Zdxvh5VEaq4uzSGiw06I3oqO9/Nor34MfpmIMWNZzvIL8J7b6n773VR6Fuq+FvC6nrGDLM0lE7dvKQ88H1gwox2F2KvdlQyf75fzwNyzb5F2uizxWl7fAcuRpVvJIEwnuA7vJf6aUmsD1dXn9KZtWrz/oO8LjXU4DuGooeWatd7nurkFlVV2at5riJWa66oq9Pd+9ZeOwt9yhvGjiL3u+6ykSPMcqyH+yvVyoKWWxhmrNJCwjNUFlwP52wD0g8Dsr8Bg9TqzbllH4vnddnZQV33ZqBpsx6wJvxw4qNi5nqzeP9PNXsy7TrCof6n1Ru/j72KXQZaSlHhLpYqTePdcfiLW1tdHS0oLH4wHvQN8/TplzUd1ZqHW777PHox8MzhOfdBKIAVRQurlQCcTQk/F4Fknx4KjcyspFmWRmLmTlq1U4NUZSH84jJWhZPaZrD1H684VkLs5k2SYrDo8W05xsUoDy3xVS+JYNN+D8qJDCze2fMltwEGl+dDFpk3S4bWUUPr7Qv92WyeaevSUURI9x3EmsL69k4cKV/Ps7wPRcsu8xEVNfRZF3vxf+vIiqOjDMyWZFyDfEaij9wgGDjUx7IHC6ngVTjNBs4+Ni0D+QS9YsPRy3sjVw3fUajHNzyJsTuGy4QhxDCPr5j5AxS4/nSLk//5ZtKsPWpMO8aAVZ8cD0FWTPN6FrqmH3s8tYmJnJwscLKbN50E/PYMUSX06nkPdwKkaNk6pXV7IwM5PMRSsp2teCOTH4bPi3ayuj8IlMbznwbndhLhldvi2rxZg4AvueQpYtyiRz0TIK37XjGWkh4/EM9GHmewezs5k/RYf7QKm/DC/bVo1La/Q2Oi9nu78MOqnaXEjh5u2U09NzqMWYEEXNG2tZtnAZv/hdVeDMYPEmDId9+7OMwkoHHq2JlKW+KyqNR/4xCT0Bef5EIWX2EZgTuvo66mmZ7vqaLd1nxxNpZOLdgedZzwKTAVyHqXo/YHKQUHlhYcXSNEw6tzLdd4732PDEWcj416zga7vbPFJox4zi2G4ljxb+vAx7swbj7FRGnShjvXfZklo3mjEzmefrX7sP8yjUfW17qOty/zvY6kBnTAq+b86bSoIW7F+U4SSLRXNN6M7U+Lfn2y/9zQvIDWo+EiqP1ZLJSk9C32qnfNMyMhd7y5HNjXZSGovuC0zb3fUXip60hzNIivdg21PIskzvudpjw60zty/X7TUYhr2/5+XNhVTVKUF5WcD3R+Hmqi6+0wAs5C5JwxSjuod94oSxKWQ/aVEvECzejNlTrSorqnvA4kWkTtLh2t9+jCtfrcKJnuT0XMzhlJV4M5OarO3fea/X4NYYSHkg25v/yWSnmdGds7WXj8eLqD6lxTh7Hp3l5CW7j12kyxKMjb72WiZNnkRERATNzc2UvVnG88+tY/3/8zz7qvfR1taGVqtl2vSQ3/a90u2+D1Haw7Q016jnhBa/gBkmDR5bGes3V2D3Tra/nc8rVgfozNwW9KXowfbuWnZ7a2mdHxZScdANOj0TAfteK9aGcwC0NFuxVrZ/GoKG1zRzV6IBXNUUryrB6v1FZn87n2f/7NuL3rH/eT2F79sBO/ZDkHaPBX2kk6pt+ZT7apcPlZO/uwYXeibeFvr1TufuWuytGkzTfRcJMCWDxDHg+uI9dqNnwSwTmmYbZS8UUhG47m0f4GjVYb49o32FPaA+ho70LEgxo222Uf6rIn/+OT8sofAvNlwuDzETIe0es/Jlv3UtpbXeW1mdlZJVb1Dj1mBMysIMMC+ZSTpwfvIK+W/7SwHlv3xFuTH6tW/3zYDz5vywhMIP7TDYxNS7AtN35DnwJquKrd4bqxPrtnzKbR40phksiA8n30PQesv96SP+Mux8t4h/35hP/q/LADvV/jLYQlOlFWtlNfZenEPXvmLWvl6DEyf2Q118PTTbKH92t3d/nFg3V3DADTr9RGX+AzOZpAX7n59tz/M6KyWriqlW1/2o9KxMd33N8san2JvBODmgqwRffv/tA4JbfgbrkBfz5mGOU76ElOkox168ijdq3WjGJ5EVGGx0l0c+ddW88oY3jw6V8MHflKC6ZoevDaqTsk8P4UbLcO+7SX2ZR6Hua9Uhr8saKg45QTeBpIAvv/SkSWhb7dS+6oT4fZTvqWD3jvbtcaic/P88gBs942e1L0eoPO7gOBXvlFO+cztFHwZc4y9V40SDYXJqUOrurr8O4heQkqjtuFxxIX85ApqEqaQSzjUYhroaqiqtNLUCnKMh4PvDWtlElyM0z5uHJT7UPWwXNS7QT76L0Hd6nxBlpbgaJzom3aqEQPp95ZT9eTelz/nKrPKdVX7QDfHjSQ6nrLTaee+FwvbvvDfW8vExQD+GJABi0UQAZxs44luurpyizfnk//KVTp6UX8L72EXqMhjrK/Hx8URHKwOG/+3Lv7GvWrkttLW1Uflf/4XLpdxRhw/XMWzYsKBlrxgz4hgBOGylHX511Ow+olzMpsCLuQFnZcCfvZbI8CHgdtR2uNk7/3S8w76Ez8nxPwUurWfCNVpojcX0YAEFGwM+aROIAaIGd/JqdF0RH3/pgbETSfPepMy3m9DjpHZPBZBE3HCg7qC/2slv/y6O1IMm/gfKjapH1McQSufbdhav4qFHV7G10nvsLgcfd6giKGPfYTfEjSEZMJtGocWJvUIdxNfwqSMwMvBuFz23BOblxgLypiqvbGu67LXFg71WHU45KbU5gBHEzQgn30N428oBF+hm5FH88gZWP5lDxlwDjR9VUaM+N0E6z8fQ59DNsdpO9kHN5eQ99bQA5jE6NCHPdQW1jq6aIvS0THd3ze7mvS9cwfmd4s3vt63qxAE65oV+/Ci0uHB82rEqpWzfIdzoGRMYbHSTRz7uvx8OcU9ooWm/eppPX+dR+Gpeq8beqmPCLN+zsXSmjdfg+fJjSlACjoodW3mvPon0pbnkPpZLTmYqScpXCQS9o9UxjzuyU/37Ioo+15CamUPuY7nkLk0n2YASvERoAtKGcf2peb8jiLslOB835mEZoTRWj+FirsG+oZQ9iE1YqtrPNCYMBqI0dHKnV9TbqVCXp70f43CB9poJ6AFnbQWlv36P+lnpZD+WS+5jOWTMTSJWtViXvnHyiSo/HN8GXu/lWG0uGJlEXtEWNqxZQU5mKoazVVT5flB3cAnvYxepy2Csr4YUihoU5X9Sdfr06aB5rgYXLS1KHB85cCBRg/rm/a1u931PPd8AI+I6e5ipYhiOFjeNp9QzgLqWEBdzCy3qk90rOrSD1dO8fNvtFfX+JTFsKOAJtc5GTh5z4KhTt5xst7vqAO5II5ZFZiCFeZP1UGfzXrQGhmvB3RhqeSctrcBACMy98KiPIZSutu3jPfYmF119pQKMHxLb6Zeb9Uxg61tlu3jLdhDPNziOOTje2f0CAA/nQrVBPdWoPNUYrfzZdb6HUsH6Zwop2+egJcbApBkppC/NY8OOLay4M3Tli6KrfOzkHLYF/tGF1pYQAUQ7X553f67Velqmu99GxUeHcUUaSUzXA2ZSErrLby9VXiSNjAUacYUKbHxpA4ONbvKo9/o+j8JWV8bBY6CbmKxUKd03DeNgD/bPfEGQnrSnt7BpzQoy7kwiKSmJW+Zns+K+TtpKhVHeLEs3ULwuj+z5t5CUlETS7RnkPp2saiNJ2NdfEMNwvM1IO2j51oHj2HFOwkVcg31DKXseWkLsZ2O9A8fRekJd5X4tTUrbuiBWXE2AdhgWgPg08l7exOonM0j9URJJSbeQvnQF6Ykhz1xoIbcTrOLFVRTuqcbhicUwMYmUe7PJe2EHW55M7aQq+RLexy7SZQnGvjl1yv9mo9E4Pqgj1xtuvAGdTmkcd9bt5tTfQ0U7Pdf9vtfgdIH2+pmd95gen0PBzp1sedIMRxuUizBUo/spMUQBjWe6eJWv1xw0uEE73NCxcM3z/hLrE2XUNwDNh9n1pLeLD9Vn1a+7CFX2KL/29AkpmOckMUEH9n3eBrG+YxjW8ZYHZmKigCY3lyL3ut62j/fYh8SFfDoXG6WBZjcuoKz+G+WXcYjx7FNHBp4NZbt8U9UhH32f/C77kdMSG6pHlzHKj4KG496/u8z3TtRZKXl+OQ8tUdrCrN2utOVIWpjdRfVEV/l4ac+hP8/nqefoMQzv6uZ+kWU6lPc/4LALDKYF6KfchSkeHF90k98hdFWOiBmEBg/uvrkVduMS5FHYnBTV2kGbwNR5kDHViMZ9gI9918W8R7hvuh6Pr23V4kwyFy1k5RtKO6Mei89h8Vwj2nqrvw1nZuZClj3rq1IMFOb1F+hoA27g5N6Oeah88tubDvTqGuwbStlr4vCb6v3zfp7Z2vWP0hhdiH1MVTrJPVNPOZD28H1Y4jzYXve2SVucycKFK9mtag998ZxYi9ez/JEl3nNZorQpnLWA7JAjKPTffaw7lyUYO/S3QzhPKsU9fvRofpb1M6ZNn8btd9zOPfPuYfDgwbS1tXH4sJ22tr4JQ7vfdyt79jlgsImfrAk1XIaetIdnYsCFY18N7DnCyVYwJOYqkX+AlLkT0fvS9bkyao96YMxMHpkfuJcWcm6fFPwLsRVAw6Cg6i892cZQBa+jmhMu0E3gFnVj+jm5bNi4mqybVdODVLDncyfEJ7L0J2Z0zTY+ftV3iyvjiBO4NpFcdbPAOfOYGAcuR7XyhXaRx9CRb9umDv3FmZ/Ywo4dBeRM8h770ARmBOUxEJ/FVKMG6g4r1USfO3GhZezU4IbTkMLUcYFnw7vdsYlkq9qX6JespuC5XNKU7rE7ZUwMaAsGgJmcyQZoPckR/5umXeV7CPetoODlAlb4z7GTmrfyee+QB3R6pV1cSD04h33NegQnWibdnhN07ennP8LMUD+OAlxcmQ7Fyp4vnDDGRNZdRvStdmp3d5HfnfGWo4QZ6nuPnqzpRjStDg6Hemp2CfR9HvVA8T5szVoSpq0gcZwG10Fre1ufcUrgc+i/29sdAQwfowv9ZKw705Ufr86Dpf52SADcqA9ZLRfe9RfA+x3RcTk9Wc8UsO6JNKVH/LCuQSOW2ckkz7qYl7Q68bkTFzom3NzxHpb7QgGrl3QMtYKEKivzZpAwFFwnlDuAYbgW3If42Nd+EYDhjOnyx1NPpbNi4xYKnmo/DmdtGfmVdjzo0E8KSuzVj/exbnQZjPVVYNTW1sZHf/0rTU1NDBgwgNHXXkvaT+aTfOstxMQqtcgRERHccOMNjDF0c3cNUzj7XvPydiqOe9AmZvHLl1aTc28yybOTSb43m7yNvyQrUYu79i2K3gco4bVPnBCfTO4LOaRPNyp9Uj1WwJLpOjzHqtnT6dtUF2f3zg9weLSYM3/JlufyyH0sjw3bVpAS0xTcd81HSts1053ryL7djD7BQvpTa7htfGCizlmLP8Du0WF5sIDcey0Y0WO+PZt1P0vGGB8LIRvitqt5rRYHOgxjNB0akJe8XoUzUk9yzgZy7rVgxIjl3lwKHrSg8zio9rVxushjCKXkT9W4Ig2kPO5dZ7yZlKWr+ZcZejSu41QdAGtxBbZmLeZFa8jLTMEcrxz76l+kYtS4qan0dn9SWcQHRzzopi6h4LF0LAlgnJ5O7sYlWJQHvKrtGrntF6uV7WLEcu8K1txlxhCvpam77rHH38a6NdmkJOrRJ6aQ/dy/kDIGXPvKlTY1Xl3lewenNMTGGUj62Tqy5xi9/Vyt4LYEDdQd8f8iPnKmERhF4lPpJGemk9aTc9jX9hdSvt+NZkwKK7YVsPrJXFasKeCXmZNocXX9S/tiy3QoNX+24cRA0gw9HKsN6uYibJVFVNg8aBMzWPNvGe3neM06UsdrcH/+Qe/W2wt9nkdH3TQCoyavIH12BukdnmgGKuXTIx60iUmYBrs4/GFAGfIGrJPuWEFaoh7izaQsXUfurF6GJ95gST81N6jsr/mJKXQTiTCvv3YllO9zBS2H9/6VOsWAflCT0iN+ONfgvMXkPpbLkjlxIZ7aXaQQ9zB9YgrZ65aQnDCK2G777Q8sK8q+b8k0o/XY+aBYuYPUnHCBdhKpT6VhjvetP5ekONWqelRW1FxohugxzFjCuqUpGFHuxStmG9Hg5MhH6vSKfruPdaPLYKy1tVU9qde++N8v2P3GLuqd9UGB0vnz5/0dug4fPpz77l/YJwFZePtezdYn8imrdaKJM5OSqTQSzc1MxXItOD4p5RfPtvcvVb3xWUr2OtEkpJDx9AZ2vpBHxmwDHLNSsmHrpYum929l+cYyqutg+A8sJCebGXWmiqIXqgnqI3B/IUXvO/AMNZH68Gq2vJBHxlSo6bQxo0pdCflbyrFjIDkzjw07t7D64VRM0U6qfptPSXdfDnW7vJ2uujj8kapA783n2e3VOKONpGTmsWHnBvIykzHgwPrb9Wz1tbm52GMI5f31bH2rBmeMd50vrSZnrpkYVzWlv8pX3jKrK2VVYTl2jx7LvTmsfkk5dvPQBmreKgzo88xJya9KqDquwTA7g7wXdrLh6QyShzgo+8QRtFneX8/zr1fTMNSsbHfnBvIyk9A32Sj79foOL2QEc1NTWUPLxFRy1mxhy5oc5VXxA2VsfVG1ZFf5rubLi2gTqY9uYKdvnzx2yosL/WW4Zkc5NS4N+hkZ5N57i9KmJtxzeAmUPfcURZV2mqIMmGclY5kQxaG3X+YtVZZ3cLFlOpT9FdjqUN4s3Bv6a7l7TkpX5VN+yIN+anr7OU6MoaG2jMLnQr8Ldkn0dR7tL6F8vwtNfBIZj6Vzyzh1gmC7P7PjAag/yJ7Ap4GVRby11wnxSWSt2cLOl1aTc7uOA5W9rKakhNfeteOOCSj7D5hpej9UVxA9uP4CVLz4PKX7GohJVJbb+UIeGTP0NAUuF8Y1mDYtAS2q4LTPOCn51cuUH8J/D9uyJodUkwbnRyXkF3fMjSDHqyg/McJbVrz73uoIKivW4reorgf9jCxWv6Ss/7bhB7AeUJ25HpaVYBWs/3UZNfUaTHNz2LBTuRcnxXmwv11EYWf3o368j3Wly05fIyIi0Gp79lixs05fA8Xp4xg2dBitbW04T54kOjqa++5fSPxopT7H1dBAa2sbI68Z2etOX91ud1hPx9oZscwe43387eZ4ZXXQo/Eg8WaSJg5TOow7ZevizY1LzNtxor/TV58EC8kGbffH0Sk95lkmhkUBLd9i+8j36n13vB2RYmXt4+1f7MHCXPdFH0MoAee46TjWvaHXapyezJgYut22PjEJ00hN18cBYW83tPZl3Y7OugkIJ9/Vwtsn4/Rk9M3qMh7mObwMMtbtIN1woMtOXxVXzj534C/rXZ3jy6GP8yjBQrLe0zfrMWgBD98e7IM3DgPu393ndzjXXwhhndPOrsEU8rblYGnuyfXcO/57WJh5G9jJrf8+2UVZCScNXHxZCfeeHayPy/tF6jIYA4iOjg5vnEevcIKxUHTDddz/wANcO+baoOm9CcZaW1s5e/asevJ3VtITBeTOiKJ64zLyA7pesDy5hbxZUUrP35eoirRHvL1Cn3x7GSu7arck+tZVn+8ZrCtKw1BXxlOrArqWic9g3S+7G0VDiO+Y2Xlse8xCwxV4PXcYcUD0mS7HpsTbeerAgQMDl+lSd2NTdqa5uZlDhw4xbtx1DBk6JGh6T8em9Hg8PXwqdmU7fn4CybMSuWHWj/mH0cOIGpPET//xn1kwVU/bsb/y6q/39usXUepj61i2KI37Zv2AIWeqeW3tO2H+MhEX4/uT705ipt7G9MlTmDPrekZqY5k0eyHZS1MwRrupeXM1f+qumYsQ3xUD3TQcsFLxhxDjh/az0GNxir7QZZsxwhhsuy+5Glz8x+uv8/Xxr9WzeuRy7vNlsXcr67eUYzs9HNPsdLIzU0n6wXAaast4+VK2VQtTI1pGxI0iylkdRlso0Ve+P/nupOzZX1D6iQOutZCamU367RYM5x1YX/1F8BimQnzXHar2jnpx5Wlp9uBp7tgjnbh43VZTAgwaNAiNJuT7Jh1MtVi4e97dYaf3qXfWs3vXbk58/XVQleWpv/+dkt+W8O2336oXCcnj8XDunDLEghBCCCHEla7bJ2N4A5xw2Q4exOnseaVZnD6O9AXpjL72WlwNSjuxY8eOcejQ4bADMXq4r0IIIYQQ/S2sJ2MAUVFRREWFN1SRRqNh1o9mMe6664jW+gYSC8+Jr0/wn++826unWy0tLf6hlYQQQgghvgvCDsboxZuVl1Nv3qBMf6qAW/iA5b/sspvMq1QyOc8twODYdQmHO/n+SH54HQsmNvBB4JAnQSS/Lx09qY/lkhL1Mfkb2/sFvDpYyPq3RYyve421r3YcVDwU/Z255KZo+HhTPmV13h7ff0QXZdPnCi6jYR+DuDSUsmE63Zvvyyu4XF1Bwqqm9OnN06rLpTf7NmqMAcOYUerJVyTT4jwKNm4gt9Meis1kPVNAwZpsktSzQhpO3DgDhvhQA4GInhoeb8AwdgydlybJ70tnAXfNNmGclcIC9azvunnzuG2qEfPsecpg2mFYcHcypoQkUnyD7urHdFM2fa7gMhr2MYhLw1s2evV9eQWXqytIj4KxtrY2mpub1ZP7XXNz81XVlUUots88RI01kjQrQz1LMXset0wxMGpgA1XqeVeTuXls2b6DDQ+pZ4jvr128U2nDVvkOu9SzLlZ/l7c9e3hvn43qP+/pplPbdrv+ZMV2wMo7PX2A8T2Q+vQWdmzfQLZ6hugTkr+916NgDG+3Eb15CnWpnDt37urryiKU/e9gqwPNuERChWPJMyegw82Bqqv8DhypIXawhh6+rCuuak7KN69i1ebyvq+i7PfyVk3J86tYvz28KkoA57uFrHqmkPJuelP/PtIMjkUzWEN4rZ9FT0n+9l63nb6G0tbWxoULF3rUGeylcO7cuYt6ezJkB3YJFpKnJDB6SDPHnU0A6BPTWHT/XG6dMZOZiWPwnP6CEw14h7NIJCEukqNfq7vnU+aN869Hj3n+IjLuuJWZM2aSOMaD68CJHnTq58Q98cfMnjCayIhdVHweOC+Zf3zwx4xurmH3r6ze/mmCt2e5PoZTNfaA7U1k9r03MaLhM/5QaQuxvz6dT0/JzGD+7JuZOcPChKHncB52EpiCeDNp92cw9+aZzJyRyJg2F190yCeVLpYxTk8mcewkpk0cSWvdCU5HjeOagUe950I5d+n33s+d3mX17uPYgvZZj3nWVIy6Zo4PuYXsf7yHW2/S4662eb/EwzxHgfs4dQIxrlqiptzLTfqz2He+Q+juEAPy++Cg9uUT9bhP2HAGDTJK9/kbbyZpmpER544Tc0s2S1JvJTHezb6DvnCkm+U7E5T/FibEnKL2SEAOXILtGqcnk3htJEe/Hh6wbIjz18W229cRhXnWVExjh9Hs6Fgek6aZGOUvM11fI92XtxQy7p/PbbO85eCcE3tQeeuaPjGJqZMNDDt3XHX+lXJqim8/HqMu8PrrppwmWEieMprIr7zTfpjK/Uaw73yHU3MyyJh3GzfPmElivJvjBwPzSH1P8On5OVXoMd8+nwV3K9tT5y8E7msrY/3HFOLcBxzD0U7zzZeno9qP3U+5j11nmsbEka04naeJGH9NUDp9Ygrz7w04n65a7L5z3YH3HIUqZ0Hnz7v2bu9Nnd9nQ09X6e66Dbr3JfnLbchzovrOU9JEcdO9NzGqubMOX7vK357e+7op31epHjXgVxs4cCCDBw9WT74smpubL/qJWIehHaZnsS4nDRM1lD6/lt2H9KQ+uYasWXo0rR48HtAM1kCri5odq1j71ihlTECdjdLFq4IaluqXbmDLXCP2PQtZWWwhZ+MKUsZqwOPBg/JL23OqmtJfrFca2YZjSi5bnklmuG03matK26fPyWPboxbYt5WHnq+A+FRWrMoiKT54e5y2sbtwFaX7AdJYXZxFwuESljxbBuRQsDOFEbW+v31CTJ+exbqH0zDpUNYfoUETCZ5jFeQ/uVUZeHt6DgVPpmDQgKfZAxoNmkgPrr2lrHqxk0bW3SyzYNNOUlRjyDu843JaFq8j5x4Tukhln5QD9uDcW8qz/u15j9lpp2WsUUnrrlHGNewsz1w1lDyztv0cTc9mw+OpGLUBx97movpgC5bEFioWLif0QCHebZ+o4ZtxZgwR3m1EAqdrKFkVuI0w8nfeaoqXJHDyUAvGBB0Abt85Cmf5EPR3rmDNg0nog/If3Ad2k/9MqdK58CXYbs6mnaRE1lCtmYRlpAZPs0e5znBjf7uQlb6G611sO/Bazli3g/QEB+VPrKQo4NrSP1rAljmxyvBhX3RyvgOukZwwy1v7/npwvJ/P8pc7O1KVB9ax4z4TDvWwN/E5FLyUQuy+rTz0fIzqOg3jXvJoATvn0F4WHy1g55woavZHYZ6iCzo37iPlFD5V5D036ntCmGUxlMDrqdWDp82bv2475S+tpMj3fe7dN9uBWEyTtEHn3vb7p1i1w5svgcd03zp2PGCi4cO1LNsU2OW1d0xHqil8aD3BTcWV+5ghaJrDm0cB93kCzmerG/u7AeVPJf25HWRMasD63LLgQal99+O9hTz0ojXMe1OI+2yX09uFdd367302GseY0Afcfzz1Vn7zaKG/02jL0g3kzjWiDcgLz6lqDngsmFs7Gwqpq/ztwb0v3PvwVajH1ZSBzp8/j9vtprW1VT3rkmltbcXtdl90INZBfBqrc9IwYaNs61p2HwL9/EfImKXHYyuj8IlMMhdnsmxTGbYmHeaFuWTE11D6hQMGG5n2QODK9CyYYoRmGx8XA0sWkTJWg7OykGWZmWRmLmT923YYaSElXR+4YNf2l1J7vGNVZcqsCehwcfgj5XJKXryApHiwv+vb3jIK99hwDzWRttDXqre3LOQuScMU46Tq1ZUszMwkc9FKij5xwtgUsp+0AJD1QAqGCCfWTcvIXJxJ5qL1lB8B3fQUFijjwXfQ3TLlvyuk8C0bbsD5USGFmwvZ/g4wfQXZ803ommrY/ewyFmZmsvDxQspsHvTTM1ixJDiPteONRH2xm7WPL2TZ2u1UoSft4QyS4j3Y9qjyTGcm4/EMlDWkkPdwKkaN6tj3tWBODO88ahPMjDhYyspFmWQuWkZhpQPPUDPzl6Z4U4SXv961YUyIouaNtSxbuIxf/K6qh8sHSiYrPQl9q51yX/4/UUiZzY12UhqL7gtM25fb9Yo3Y/ZUU/TzhWQuzmThz4uoqtNgnJtD3pzAhKG2Hax0nx1PpJGJdweeEz0LTAZwHabq/fCukc7LWy7Z95iIqa9S7S8Y5mSzYnrAZrvy+qfYm8E4Oc1bvhT6dBOGgOs5SK/vJXrMk1uCzk3pPhea8aldnJvenlM9GY9nkRQPjsqtSlnPXMjKV6twaoykPpyHr7T70puuPUSpNy+XbbLi8GgxzclWpfN642MOuEFvugtz4PR5yUzSgfPzPapADKCc7ZuV8gxOqjYXUrh5O+WA/oFcsmbp4biVrYHnsz5U+Wu3u+oAbvSYbg/aC9JunYQOJ7VvW3t8b+q5nly3yr2v5a+FLFvkLQO1bjRxyaQu9e7HnDxy5hrR1AWX7WqPGXMn921F5/nr0/29L9z78NXpooIxvE/Tzp49e1n692ppaeHs2bN931g/Po3V67IwRzuo2LqKkr0oN+8UM9pmG2+uKsHqjcidH5ZQ+KEdBpuYehc4d9dib9Vgmp7dXlCmZJA4BlxfvKc8LfP+am761lcdBtWvbiV/83r+/eWQz4g64WSXTQn+Ev3BXwpJRh24DvOBd7Dw4++/Tfm7pWzfZvVuz4m1uJDqOtBcm0iqf329MG8elnhwfvIK+W/7BuywU/7LXdS4QD/5LpKBmEjA00TDIf8RU/RKPoUv/jtbO/l1090y9r1WrA1Ke8WWZivWSivVhyDtHjN6nFRtXUtprXfZOislq96gxq3BmJQVfNN2VVP8bCk1deA8ZMcZv4CURC2eA2+yqjg4z/5yBDQJU5U8893sOxz7K1R1ckwduKopfna3tyrZiXXzOxw4DTr9RGV+mPnr49pXzNrXa3DixH7I2ePl2x2n4p1yyndup+jDgDx8qRonGgyTg0tN323Xx0nVtnzKfUPZHionv7gaJzom3Rr8HmGHbau9ESLI8V2Tf/uAijCvkc7LmwV9ZIj93V2DCz0Tb+v6SNvt5tMjHhg7kTT/F52ZjMlK0Oi7noNcxL1EfW52P7+V6nrQ39TJm5q9PafxC5hh0uCxlbF+c4V/WB/72/m8YnWAzsxtQT9ePdjeVX4AAzg/LKTioBt0erxXhUoZ1oMuiJvIvIBAKT1pEtpWO9WvhRogzk51pZWGFoAWmiqt3iGH9CyYZULTbKPshUIqAs/ntg9wtOow3x6qlS6wx8oBF+gnzwsIGtOZeb0WjlRTur8X96Ye69l1S301r272lXk7u39djRMYNVZ5B98XSHYo29uqQtdm+HWWvwG6u/eFex++Sl10MObT0tKC2+2+qDZcnfF4PLjd7ksT8EXqWLEqA/NQD/Z31rPVXx2eRNxwAD23bCygIOCTN1V5vVcTA9QV8fGXwTdU8+0m9Dip3eP9ZfvmPuweDcZ7NrFt0zryHssmfVYsDnVhDYNztw0HGoxTvDcIX4DwRfuvQfve3RRtq0UzN4Ocx3LJfSyb9NkGaAUGwsW0RdaPH4UWiE1YGpQnBRvTmDAYiNIwHNj1mR3PYCNpv9xGwXN55C5NJynagXVv50fcm2VAz4RrtOBy8HGHpgxl7DvshrgxQV8abkdt8DiOM+IYARB3i+qY8rCMUBpxxwBm0yi0OLFXqG/2NXzqCK9Fg8d5RDWGZDn1Z9r/Cjd/FW6O1QavrWfLB7JT/fsiij7XkJqZQ+5jueQuTSfZAC0AEYGlpi+361VvpyKwqgdg78c4XKC9ZkLAL+KO2+5oN+994Qq+JlO81+TbylXS+2vEW95aYzE9GHicBRSkTSAGiBrc5ZEG2V1RgyvwKd6UFEzxnT3duZh7SahyW83HX7lAO4oJIZ569Pqceq8nh620wxd4ze4jSpBgCvxabcBZGfBnGCr21OJEx4RZvjAog2njNXi+/Dioarp73vt83UFK1cvt38WRetDE/6CTIKCCPZ87QTeBJF9Q+MA0jIM92KqKcPbi3tRzPbluwX3yYPBYxnUtSjoAzEwcpQ19Le7/lDBvcZ3q7t4X7n34atVnwRjep2Tnzp2jqamJc+fOXVT1ZWtra9C6+vxpmE+8haSRTbiaNRiTAx+LGxiuBUIFgJ5vcBxzcNx7p9lddQB3pBHLIjOQwrzJeqiztRfouhJWPlOE9csGokYascxOJePJ1WzZto6scKs0fOp2UXukvaoybVoCWpzY/hxwiU3PZkPxBvKWpnNLUhJJSbeR8Vgeyar2L72RNDIW8NASIuZurHfgOFpPA+AsXsmqV63YzkQxKsFC8twMVqzZwrbnsuiscqM3y0ASw4YCTa7QX17hMAxX2keEOKaWbx04jh3nJDB+SKzyq099owKsZzq0Qg3J09J1unDz1091WfR4+QCWpRsoXpdH9vxbSEpKIun2DHKfTla1A/Hqw+0C0NIUYsB7K64mQDss+PyHcSuo+Ogwrkgjiel6wExKguqa7PU14i1vnsAvMZ9GTh5z4Kjr8kiDvV/FYRcYpyxAHxA02joETl69vpd0VW5jGTZVPecizqlhOFrcNJ5Szwj48g8KElpoUQdC3fE22dBNvo10QL90KqbBHuyf9fRtcuU+724MdSROWroJzmteq8WBDnNKOqAne7oJTbOdT9+gb+5NYejRddul8WhjurkWL0J3975w78NXq0vyOuSFCxfweDx4PB4iIiKIiIggMjLS/2JA4NuaFy5c8H/a2tpobW2lra3t0gVfah4X1a+tomj4CjbNs7DgSQsVG6sBBw1uMHxTxfKnStRLBdtj5cBPzFgSUjDP0TJBB/a3fQ0nvQ6VU/iMrwbdSMrSf2bJXBNpD2RRsreb9QdxUvaFndTxRhIfSOPcBC3UVQf8ktGT87NUjIOcWDc9S6Hv0XW8mdxVq0keGrCqXiir/4YsYjn85nLWd/Nr1v52Iave9v6RkEL20iWkTkpj0ZISqotVib16vkwZ9Q1ZMCSOVAhqowAQG6WBZnfXb+IcbcCNgZN7l7My5Da86r8hixHEzQV8++iVOnIE8E3wxF7oSf6G0uvl43NYPNeIts5K4bpCf7W8PjGXNWuSiVWnV+n1dn1idCSD6ksrlbghwJn6Due1W+9/wOFMCxbTAvRTYjDFg+N93zV5MdeIt7zFHGbXk+pG4r1RwQd/y8Ay3cSCeDMxJj0cr/C+ZNOJXt1LYtHNBlTnxldu61XlmYs5p97raXiowHZKDFFA45kj6jk95GTXfjspc41Mu0/P8MlGcFXz3hvqdN3x3ueHhQpdzMREAU1uOt3bul3UHknBMH4a6fHDmTgWXPu9zVP64t7UnYu8boN1tb++azFoYt8K9z58lerTJ2Oh+N7QPHfuHGfPnsXtdtPU1ERjYyONjY00NTXhdrs5e/asv8+wyxaIATirWf+WE2fxdj44DvoZi8mZAlDGEScwNpFs1SN8/ZLVFDyXS5rJN8X7uDo+kaU/MaNrtvFxwNtRyQ+vo+DldWT512On4tXfUF0HjBjvfwRuTDSH1UDR+aeD2Fs1GJNTSdCC44vAwM/72L3eRqnvSwaARPTKC2hd0kQFX776pcbgX1ifO3GhY8LN6qa1KeS+UMDqJcne4S8K2PJcVvvxHKqg6FVv+4RxoR7692YZRc0JFwxNYMZ8Ve7FZzHVqIG6w7wXPCfYniOcbAVjYkC7PwD0ZD1TwLon0jDhO3YtY6d2PPap47Sqab0UVv52obfLT1eqCJwHS/03dABu1IeuilLr7XZ9dBO4Rd1Qet4MEoaC60TH3+nds7LnCyeMMZF1lxF9q53a3b7r4eKukZoTrtD7OyeXDRtXk3Wz9+94M0mzk0meblQlDGZ9+yBODJgy78IYB/b9uzpU7/mEey/pKNS5SWPG9VpwOUM8CbmIc+q9ngyJuR2eaKfMnYgeF459IbfYI85XP8bWrME0Ow/LeKVqt7sK7I689/lrE8lVP1mcM4+JceByVIfOHwCcFFXZ8Aw2cdvPLRgjA5qn9OLe1O39V+1ir1sVZX/HMrVD2Z7K2C5/pPSBcO/DV6lLHox9d9Sw9XdVOCMN3LJYCQhK/lSNK9LIbb9YTfbtZvQYsdy7gjV3mTHEa2kK6IrH97jaMEbT3nDfy9asYVScidRVK0hL1EO8mZSl/4wlHjxf11IOJD+1jQ1rVrPpuTDedqwrovYYaOL1aIO+ZGi/ucRbyF2aghEwTk9nxao0TF32QmLlSB1oTKmsW5qCOd57rLervkgqi/jgiAfd1CUUPJaOJUHpnyd73RKSE0YRiw2w0TR4FPpJqax5Kg1zvDfNUgt6PDi+UP9GJPxljrppBEZNXkH67AzS54G1uAJbsxbzojXkZaZgjtdjvj2b1b9IxahxU1NZ1OmXm6KE8n0uGH8b69Zkk5KoV/oFemoNqVMM6Ac1YSP0sRunp5O7cQmWML7EwxJiGx3ztwu9Xd57I9RPzSV7jlHpN+jeFaz5ianTKpogvd2unw7LgwXk3mvB6N32lkwzWo+dD4p79/yp5s82nBhImqGHY7UBbYl6cI2ELG8fYPcE7q9S3tb9LBljfCx4Gz7r05ey4rFc5k/ppn7H26GzYZbyVlxtYDcXKuHcSzqjmxJcble8lIFZ68FuLQr9hK/X57SE1z5xQnwyuS/kkD7dqFxPjxWwZLoOz7Fq9oR6OaHHdvPx39wQb0DfacP9YEfONAKjSHwqneTMdNKAktercEbqSc7ZQI6//OVS8KAFncdBdUBwFZLv7c4xen/DfZ/w701h3n/VLva6VelYtgPyQp04hFD5G74w78NXKQnGAu3NZ9de5XXv3Af08P56nn+9moahZlIfXs2WnRvIy0xC32Sj7Nfrg3+FedtyEeKVdGdxPiWVDjxxSWSt2cLOl1aTM9eEpr6a0i1K2PbtWaUFiifMHjtKar3NdY8d7ND3Ssnr5djdWkxzc9iwcycbns7A3PReN2/81VBYXKG8Uj43h9UvbSAv0wKf1agCGSclv3qZ8kNgmJ1B3gs72bImh1STBudHJeQXO71pSrAe86CfkcXqlwLS7C2lMGRVQpjL7C+hfL8LTXwSGY+lc8s4oK6UVYXl2D16LPfmsPqlLax+OBXz0AZq3ipk7R7VpkKoePF5Svc1EJOYSs6aLex8IY+MGXqaDpSx9UXf+VT2seq4xn/sG57OIHmIg7JPHKo19lY4+duV3i5fwmvv2nHHmEh9dAM7d24g7wEzTe939xaVT2+363W8ivITI0jOzGOD7zprdVD123xKuiy3Xdhfga0OwINNVX0X9jUSsryVkL+lHDsG7/4q5c0U7QzYX29XGt0EV4oaKrxvhXoO7aOzikbCvJeE5qDq3ZOMSG4vt0lx4Ojy3PT+nFZvfJaSvU40CSlkPL1BuZ5mG+CYlZINW7t40tQzZX85oFTzhbgPhlKzo5walwb9jAxy771Feeq0N59nt1fjjDaS4i9/yRhwYP3terZ2VWUM+N/uBOxfqPpQDPveFO79V+1ir1uVuhLyf1uFI8JXtpW8iD1aRtVxdeKOQuZvD4R3H746XVSnr98fRiyzx6AFaDreydt9ZqUDWKysfbww9M0m3kzSxGFKx4KnbFT5XnX20sfrcdb16hIKQY95lolhUV3tcyjtx+p2KK/yd0afmIRppPKq/bcHq6gJcTMMJ41aWMskWEjWe7B9FHyzMk5PZkwMgJvj3b5hFkKChWSDUuXY1fH797Hl2w770FfCyocu9Gr5gDLa1fF3pafbDeyw1X/+LmG+tuvBNRKyvAUsr97f+Gw2bErF8GUpmc90FST1Ujf3kk75lwvv3Pj09Jz69XY/w3XfOnY8YMT+eiarQv7IC804PRl9s3p/ujif3Uh/bgcZ4+0dOv8OFN69Kfz7b5A+uG6DtedFb85b6PztgTDvw1cTCcb6yrzVFC8xc1Ldm7YQoksdRsK4GizZwM55w5Xe/vukSk50ZCZn02pSYqrZ+pCqpuJympJDwb+lELvfOwKKEL0g1ZQXKfWxdRRsLGBbphmtq5pyCcSE+N7Tf7qLwo3/TpEEYpdAFnmbCih4eQUpY5Q2b/0SAi3Oo2BjAVv+NQVDm50PXu2XvRBXCQnGLlIjWkbEjSLKWd2xHZkQolstzR48zR177fouc9ZWYe1BNZfoiUY0Q0YxKqYFe2XXbdcuqbMaRsSNIvasHevFtG0UQqophRBCCCH6lzwZE0IIIYToRxKMCSGEEEL0IwnGhBBCCCH6kQRjV4r7VlCwcQVh9L/f9y73ti/39q5KyvBR6x7uZEgaIYQQ3xkSjF0p9GMwjB3DKPX0y+Fyb/tyb++qNJy4cQYM8b0ZgU4IIcSVRIKxvjQ3jy3bd7DhIfWM76nLlB+pT29hx/YNZKtniIuQzYbtO9jydOdDTwshhOgbEoz1pUgNsYM1aHozQuvV6DLlh2ZwLJrBGqLUM8RFiEIzWDl/QgghLq3IcePG/cL3x4ULF2hra2PAgAHfi37GjNOTSbw2kqNtY0m7P4O5N89kZqIe9wkbzsYOqbHcu4D7b7+VmTNmkhjv5vhBJ02+udOTSRw7iWkTR9Jad4LTUeO4ZuBRTjR4EySkkHH/fG6bNZOZUycQc86J3elbGvhhKvcbwb7zHU7NySBj3m3cHGI77bren3Z6zLfPZ8Hdyvos18dwqsauDK7rE7Dtvf6JRiyzE0m4dhjNDmW9xoD9slwfw7l6e4h8UnSZHyGP1cKEmFPUHgnaM2X/5y8i4w7lOIP3X9nH60zTmDiyFafzNBHjryHyqxPBx+cdW2/qZAPDzh3vsM/KvFEBy3WXt3rMs6Ziio/k6NeBW+psukq8ub28zUhkTJuLLzqkDz7uxDEeXAcCj2sis++9iRENn/GHSlvAckZSMjOYP/tmZs6wMGHoOZyHQ5SLwPI4IxG9+zg2b3lU8uN6/mGGgSjXUf7eOoTRQ5o5HlhehRBC9JnvdaevOZt2khJpwzbUhGmQB0+b9ymO28bup1dR6utReXoW6x5Ow6QDWj140KCJBE99NaVr11NW513XmOD1O95fyPKXwbJ4HTn3mNBFgqfZg2awMuCu4/18lr9crSR+tICdc6Ko2R+FeYoOPB48Ecp23EfKKXyqCG/KsPYHgPhUVqzKIileo6TzH5+d8pdWUuSLvB4tYOccqFi4HGV0QAtZ63JJM0HN679g7RtNpD2zjqwpuq7XE6Cr/Ah1rMoKVXkSuP8e73FqgNM2dheuonR/DgU7UzAEbyXgOALct44dD5ho+HAtyzYFDuOeQt62HCxUU/jQeqxh5W0aq4uzMDeox1PsbHqA6TkUPJmCQaOUBTQaNJEeXHtLWfVimdJje2fH7aqh5Jm1QfuQcLiEJc+WedcdsO8B5cdzrIL8J7f6y0935TFtTTFZicogvT7u2oDtCCGE6FNSTRlvwnC4lJWLMsnMXEZhpQOP1kTK0hRvAgsrlqZh0rmpeWMtyxZlkrloGYV7bHjiLGT8axZ6oPx3hRS+ZcMNOD8qpHBzIdvfAabnkn2PiZj6Kop+vpDMxZks/HkRVXVgmJPNiumBO6PHPLmFqldXsjAzk8xFKynd50IzPpXsJy092h/Qk/F4Fknx4Kjc6j2+hax8tQqnxkjqw3n4jjCYnrQ1SiBme6uQtW/YYXY286focB/w5dNClm2rxqU1cus9aeoVQFf54afHPLkR66ZlLMzMZOHPS6k5rcFw6yKy45UUyYsXkBQP9ncLWZbpPT97bLiHmkhbmA6Us31zIWU2N+CkanMhhZu3Ux64GZ83PuaAG/SmuzAHTp+XzCQdOD/fgzXsvO29rAdSMEQ4sW5aRubiTDIXraf8COimp7AgHiX/H84gKd6DbY/quHVmMh7P6GQfLOQuScMU4wwqP0WfOGFsSnv58ZVHVzWlHcpjDnlzoGrXyxRursIJuG1lFG4u5OVdVeoNCiGE6CMSjDXbKH92N3YAnFg3V3DADTr9RGX+vHmY45SAYu3rvrHmnFiLV/FGrRvN+CSypoB9rxVrwzkAWpqtWCutVB+CtHss6COdVG3Lp/yQd5uHysnfXYMLPRNvC+6awPnJK+S/rewN2Nn9/Faq60F/0zzSCH9/iF/ADJMGj62M9ZsrvMcH9rfzecXqAJ2Z2x7wbdVHT9qaX5KVqMHxfiGrtnufpWiV9lgtp4/41+N8t4h/35hP/q9DPy3pLD+C0vx5PYUfeseVO7SbtZ84IHIUY7wB6vH336b83VK2b7MGHGch1XWguTaRVOxUV1ppaAFooanSirWy2r+PwcqwHnRB3ETmzWmfmp40CW2rnerXasLP24sQEwl4mmg45BtPr5qiV/IpfPHf2VqnnLeURC2eA2+yqjj4uP9yBDQJUwnZpH7ePCzxHctP+S93UeMC/eS7SKa9PFb/Zj27g8pjNY5TjXhGesdVrGxCydYGrJVWqmr7afw/IYT4HpBgzOXkPfW0APrxo9DiwvGpv5LQr2zfIdzoGTNLPcdHz4RrtNAai+nBAgo2BnzSJhADRA0O7JrAib0isAoNoJqPv3KBdhQT4nuwPzPiGAE4bKUdBiuu2X0EJxoMpsCv9Sh0T64hI1GL51A5631VhQBvWzngAt2MPIpf3sDqJ3PImGug8aMqano9OK4T517Vnh1vwB3wp33vboq21aKZm0HOY7nkPpZN+mwDtAIDoadNyyv21OJEx4RZvmeCGUwbr8Hz5ccU1fUgby/Crs/seAYbSfvlNgqeyyN3aTpJ0Q6se70BlPe8EXdLcHnZmIdlhPJSRIxqnfjLKcQmLFUtl8aEwUCUhuG+8uhy8LG6avn9fJY/spz8N1TThRBCXHISjLW2dAhWAiWNjAUacVWq5wBt3n8jVdP9khg2FPC0KE8ZgjRy8pgDR52vhT/K0539AX96Wc80ArEMm9qD/TEMR4ubxlOqNAB13v2JCAxn9FhmDafJ5UGTcAvZAU+PoIL1zxRSts9BS4yBSTNSSF+ax4YdW1hxZ+hKs+6FPtYg07PZULyBvKXp3JKURFLSbWQ8lkeyqi1a2PaXUnscdJNvIx3QL52KabAH+2e74aLPdXicxStZ9aoV25koRiVYSJ6bwYo1W9j2XBYWfOdNaUan1vKtA8ex45xUz/Dvu4eWEMs11jtwHK2nwVcem1xY1YmEEEL0GwnGulFW/w0wgri56jlAzCA0eHCHCngAKKO+AWg+zK4nl7M8xGfVrwO/FmPRzQ740yt15AjgG+rf7sH+HG3AjZbhoQKXKTFEAY1njgRM9DYif6Ycu0eHZeEKJTjwqbNS8vxyHlqitKNau70KJ3qSFmZzafqA15Pzs1SMgwLaVy3OZOHja7FexNO4XfvtMNjItPv0pE02gquG97xPg8LO24tkf7uQVY8uITNzIQt/vpVymxvdpDQWLfGdNzi5t2NZUT75KKFjMGXfmzj8pjq99/PMVqy+8jgkLnRVpxBCiH4hwVh3PnfiQkvCjDRVw2k9WdONaFodHA71JMWr5oQLdBO4JehJEzAnlw0bV5N1c+BEHRNuVjerT2PG9VpwOamhB/uz5wgnW8GQmBscVAEpcyeix4VjX2CV6EmqXyzDWVfCdqsD4pJY/Ki3qft9Kyh4uYAV/mNwUvNWPu8d8oBOH9wgvs8kETccqLdR6mtXBkAiel3Anz3kfPVjbM0aTLPzsIxXGu5X+GaGm7d44DwQFRN87HMSGRv8EqKKMoTRlucCXgQ4VEHRq9U4gVHjUv3nzZiY3XEfnilg3RNpmIKme33uxBWy/KSQ+0IBq5coIXPNCRcMTWDGPFWy+9ZRvGMbq+9VTQ/BmGju5CUCIYQQvXFZgrEbExMZYwjugCCUcNNdVpVFVNg8aBMzWPNvGaQk6tEnppC9Zh2p4zW4P/+AIt+TmqNuGoFRk1eQPjuD9HlgLf5AedL0YAG591owosd8ezbrfpaMMT4WVI3adVOWUPBYOpYEME5PZ8VLGZi1HuzWIqVqKez9KeG1T5wQn0zuCzmkTzdCgoX0xwpYMl2H51g1e94P3rZPzcvbqaoHQ/JisuKBUxpi4wwk/Wwd2XOM3n64VnBbggbqjnRe5RUiP8JXxhEnEG8hd2kKRrz5sSoN0+DglEfONAKjSHwqneTMdOVFh07t5uO/uSHegN7XcN8n7Lwtp/ZrD8RZeOSpdCwJvnNqoes40UbT4FHoJ6Wy5qk0zPEo619qQY8HxxflQAnl+1ww/jbWrckmJVGvnLen1pA6xYB+UBOBvYr5VRbxwREPuqnt5UefmEL2uiUkJ4wi1ruUUh61mDO3sOJeC0aMWO7NpSDNhLbVyZcf+Vbowt0M2nEzyZ6bSsZ8JZhLfmobG9asZtNzMrKoEEL0lUve6evsOT8m/b50Jk+exOHDds6cPq1OAj1I15d+mHo/RuzsLA9szax0pjmq2Te9idoKO0On/ZCbbpjCD398D/f8+Idcr4/kVO2f2Lp2Jyd8i578G5GTbsaScD3mGZMZ3rCTd97/jM/qhnLDNAs3WW7hjvvvYfb06xkZ4aTqt+vZ+pm3I80fpnK/8TRV75xm/JxbSLnzfu5INmOIbsNRVcKGVz7zdtwZ/v6c+GgvzRN+yE2JidyUfAf3334L5vFD8Ryz8v9u2MxffZ2fduj09QR/dV/PnTNv4IbrB/JB0VZsgyczcfIN3DTzDu6//w5uMRuIOWen/P/+P5R/7V2PWqj8+O9Q2/MyzebefxhBw2d/oNIGnzUM5YeWyVx/ww+5434lP0ac/E+qW67HMMjJZ3+oxAY47ZFM/pGFCQlmZk4ezin1elVsEZO584ejGXy0ioLf7wvoEDX8vP3iixb/Nm+5XTmn0Y5qHINHM9xfdtSa+KymiWvNNzD5hunMvvt+Zf0jwbm3lBe32WgC7B/W0voDC/9w000k/fge5byNieHMgTK2PvcH7z6oO31t4rOakwydOA3LVAu33B6w7o9KWO8rP42f8dmZa7npxsncYLmFO+6/g1vM4xjqsVP+6wJKvvDlRi2u0T9i5iQjE6dauC7Sxh8qbQyzpDJ7fAzn6mr5Q+UXgQcnhBCily5pp6+z5/yY1LmpaDQa2tra+KSqijf+o+PrWjqdjn/K/if/U7FvTp2i+LclHHc41En7V4KFZINSD+V2dOyqwS/BQrLeg+0jX/cIKP1qzTIxLApo+VY1TyXeTNLEYWjw8O3BLt5YDHd//OsDzynbRXRTYMQyewxagKbj7W8AdidkfoQrIN+62aZxejL65jCO77517HjAiP31TFZ1LI6KsPK2fd96mq/6xCRMI5XOVjs9x2HtQ0dhrTvc8hhvJmmiBmdAlyH6eD3OupCphRBC9MIlC8bUgdinez9l5+v/QVub77W0YLGxsWT///6ZsWPHwpUckInvODM5m1aTElPN1ofWt7cXE0IIIfrJJamm7GkgBtDS0sLnNbUk/OB6hg0bRrRWy8SJpstWZSmudlnkbXqQBfNv46Z4pcPZ/+urIhZCCCH6UZ834O9NIObT2NhI0f/9DceOHQNgxMiRLHkw68pr1C++gxrRDBnFqJgW7JUl5BdLNZsQQogrQ59WU15MIBZIqiyFEEII8X3RZ9WUfRWIIVWWQgghhPge6bNqyoiI4FW1tLT0KhDziYmJITIiYOyZAQMYOPAix6IRQgghhLjC9NmTsSN2O+fPn2fChAkMHDgQg8FA7JAhHPii530RjRo1in9c/DOuHXMtBFRTHjuqtCUTQgghhLha9FkwBvDVkSMXHZB1FohJezEhhBBCXI36NBjjIgMyCcSEEEII8X3T58EYvQzIJBATQgghxPfRJQnG6CQgG6Ybxv9+/r/qpOh0OrIezGKMYQxIICaEEEKI75E+e5sylMr3/4vyt8vxeDy4Ghr46K8fq5MA4HK5qK7eh8fjkUBMCCGEEN8rfdrpa2duTEzE5XJ1G2CFm04IIYQQ4mpxWYIxIYQQQggR2iWtphRCCCGEEF2TYEwIIYQQoh9JMCaEEEII0Y8kGBNCCCGE6EcSjAkhhBBC9CMJxoQQQggh+pEEY0IIIYQQ/UiCMSGEEEKIfiTBmBBCCCFEP5JgTAghhBCiH0kwJoToJQtZ/7aB1Ust6hlCCCF6QMamTEgl98E0LOOHowE8TSc58NEuioqtONVphRDt5q2meIkZrbuGkiVrKVPPF0IIEZbIcePG/cL3x4ULF2hra2PAgAHfi2BMP381m55IxaSPwdNwgr+fbiJSZ8B4w0zm3NDMXyttNKkXugJkv7CDp9JNNP3Ryt/UM8UV58o+X9ls2P4UPzU18acPe7h3tjMM/cEomv7nTYr2n1DPFUIIEabvcTVlMll3m9G1Oah4cSFLHl3O8ieX89Di9ZQf8aBNnE/2bPUyV4aowRo0MRo06hniinRln68oNIM1xA7uzd5VU/L8KtZvr1bPEEII0QPf42rKLDbsSMN4vJyFTxUFz5qdw7qfGHFWrKRwT8D0hBQybjej1wAeJzUfvkdFbcfKTH1iGmk/nkAs3nR/LqXikH8u5lkmtGdsVJ21kD13IrHnDlOxrYwaAIykZKZiHhEFtOA8YOW9P9coVabxZpImDuOW+3JJGmqj7NW3OdzyLbaPvPNVjNOTGcNxrI5Y0u5OYYIWcB+m4k9l1NSpEseb29PQgvPzckrftwfNT5qoxX2wisbJGaTeqCcq4NgCj7nx0FsUvR2wrFdQvriP8UHlbqr9+dKFwHynkcP/VUaZOt97s/8Xk85Ln5jCbTcHlInAc93D86XQY56fRsq4WABavqmhfEcF6i13uV3COw59YhKmkbeQ8VgSsbYytr17GM8pG1X+vO1uXwLKcq1TVbYT/ees43KhGef49lNZxlpR0V5OA47n5DXt5ajTdXdXHvwCrzdoPFpB2Vvq8xOcD43HP6Di99WqbYaTRgghQvseB2PJ5G3LxRLjwLplPYUfdv71CGBZvI6ce0zoIsHT7EEzWAN4cLyfz/KXfU8G9KQ+uYasWXql/ZkHNBqg1UXNjlWsfcsJpLG6OIsEp52WsUZ0kYCvzc30LNY9nIZJpyzsidCgiQTPsQryn9xKta+NTuCOddFeJ2fTTlIibdiGmjAN8uBp0yj747ax++lVlHq/6PR3rmDNg0noNcqxoVG26z6wm/xnSpUgcd5qipck8M2hFkYl6MB3bB471vfBcqcRrX+fu8iXVo+SL4M1qnwJLZx878n+nzzQiGGSHo3/5Hhwfvgblm2qUDYYbjr1ufbtW6sb+7uFrHy1ur1Nlf9ouj5fxKeyYlUWSfEaaPXgQTkOXDWUPLOWsjrC2y7hHUfammKyEoP2DndtCUueLQveF493XzTAaRu7C1dRup/2snzYu4y/bNtoHGNCH9F+DJ56K795tBBf7gXTk/bMOrKm6JTj9pdTO+UvraRob8Dx7DvJqKlGtAH54zlVTekv1nvzJ8zyANDd9YbqnATmQ9A5sZCzcQUpY4PTqPdLCCE68z2uprRSVFaDW2Mg+YktFG9aTc69FozqZADTc8m+x0RMfRVFP19I5uJMFv68iKo6MMzJZsV0JZl+/iNkzNLjOVJO4eMLycxcyLJNZdiadJgXrSArvn2V2vFGor7YzdrHF7Js7XaqsJC7JA1TjJOqV1eyMDOTzEUrKfrECWNTyH7SAnt/z8ubC6mqUwKqss2FFL7ye6raV9tRvAnD4VJWLsokM3MZhZUOPFoTKUtTvAmSyUpPQt9qp3zTMjIXZ5L5RCFlNjfaSWksui9wZVoMccd44+fKsa18y45HYyT5zlGcfGu9ss9PlFDj1mCYNo9U71L+fLGVUfhEJpmLM9vzZWEuGQH5EsSX765qSjvkew55c+jx/hsTWqjatIyFmZks/HkpNac16Gemkh20D92n0z+QS9YsPRy3sjVw3+o1GOd6961H50tP2sMZJMV7sL9byLJFmWQuWkbhHhtunZmMf81CH+52/bo+jqpdL1O4uQon4LaVUbi5kJd3KXuXvHgBSfEo+5LpLTt7bLiHmkhbmB64kQ604420/NV3DCsprXWjiUsmdalenVQxO5v5U3S4D/jK6UKWbavGpTVy6z1pAQm1GKeMwrEnOH88Iy3+/Am/PPiuNxfVr3e83nKeTgk6J7Y9qnzQmcl4PEPZ5pJFpIzV4Kz0pVnI+rftMNJCSnonxyyEEAG+x8EYON9ay1Obyqg+7kY7xkxKZh4bXtvBthdySU1oT5d2jwV9pJOqbfmU+6qCDpWTv7sGF3om3pYM6FmQYkbbbKP8V0VYvb+GnR+WUPgXGy6Xh5iJ7evEVU3xs6XU1IHzkB3nvHlY4sH5ySvk+6v47JT/chc1LtBPvovkuhqqKq00tQKco6HSirXLKi+g2Ub5s7u91SVOrJsrOOAGnd63M8epeKec8p3bKfI9HayzUvJSNU40GCb7QiqFc98r7PbmgX37BxxwA/U1bPe1G6orY99hNwwd7g1s2/PlzVUlwfnyoR0Gm5h6l2/twXz5Xv2b9f5tKvlejeNUI56R9Hz/977a/hT00G5e2eeEyFGM8QbU4aXTs2CWCU2zjbIXCturBw+Vk7/tAxytOsy3Z0BPzlf8AlIStXhs5eRv873J68RaXMh7NhcNzTGYwt1ugK6Ow1lbhbWyiRaAlgaslVZ/FeXx99+m/N1Stqv2pboONNcm+gPtkOqreXWzbzk7u39djRMYNTZJnVKh1ShVk6eP+Kv1nO8W8e8b88n/dfAzRM+Xb7LK/6azsk/lNg+a8VNZEE/45cF3ve3dyvo3gq+36uMuGtt07efkQMdt/uUIaBKmKvngfVrb9K3Nf26rX91K/ub1/PvLIc+2EEIE+V4HY3iDgvVPLGHh42vZ+nsrNXUedAnJZD9XQM4UAD0TrtFCayymBwso2BjwSZtADBA1eDiQRNxwoO6gv/rPv43iVTz06Cq2VrZPcztqg6ps9ONHoQViE5YGb2NjGhMGA1EahgekD5vLyXvqaUHsVP++iKLPNaRm5pD7WC65S9NJNqB8SUdoAtK6OXkkxJdLS1N71U8H3nxBzy1Bx1VA3tRRAGhi1MvQnu8uBx/vVc16P5/ljywn/w16vv9/C95TZ1tL0N+K7tJ1fq7Zv4sj9aCJ/0HXAYvajDhGAA5bqSpYc1Ky6iGWPbMVa4+3291xdM6+dzdF22rRzM0g57Fcch/LJn22AVqBgXT5MoL75MHg8lDXopyLzrxt5YALdDPyKH55A6ufzCFjroHGj6pUbRs92Gt3B04AnJTaHMAI4mYQdnlQrjcXjk/VLx9UkP/EQyz/5W7/OSHuFtU1mYdlBBCpIQbgzX3YPRqM92xi26Z15D2WTfqsWByV0mZMCBGe730w5ldXQ8WOQtY+sYSVr9twawzMvDcNSGLYUMAT6gulkZPHHDjqGgADw7XgbmxQJwpL0shYwEOLRz0HGusdOI7W06s1t7aEfhITwLJ0A8Xr8siefwtJSUkk3Z5B7tPJGNQJe0XJF1o65h6eb3Acc3A85A56873JhVU9S6VH+9+mntCJLtN1da6dtIQRsHRgGI4WN42n1DMC9WK7XR5HF6Zns6F4A3lL07klKYmkpNvIeCyP5DHqhH2hgvXPFFK2z0FLjIFJM1JIX5rHhh1bWHFnYDWfh3Oh+po51YgbLcNHK3+GUx6U660RV8APpA4Mw9Eqzck6aPnWgePYcU4C1JWw8pkirF82EDXSiGV2KhlPrmbLtnVkqZ64CiFEKN/fYGxuLute3sa6xeoZYH/jYw65QTvcAJRR3wA0H2bXk0r3F+rPql9bAQcNbtAOCxkCdKus/hugicNvdlz/8ieXs/yZrd0GJb0Sn8PiuUa09VZ/e67MzIUse7avOr1V8oVvqjoek/ejPOFS8+b7kLiunzBd8v0PpatzbSYmCmhyc0Q9qytHG5SAostg5xJsNyQ9OT9LxTjIidXX7mpxJgsfX+uvZu5zdVZKnl/OQ0uUtmBrt1fhRE/SwmyS/Ym0xI4MWkoxRglkG46HXx6U620EcXMDJqodbcANnNzbscwqn3z8z+kOlVP4zDKWLM5k4cKVbH3bhltnIu2BrOB1CiFECN/fYOw46OJ0mGbk0mEwlykTGKUFT5MLgJoTLtBN4JagxtHAnFw2bFxN1s0AZRxxAteavNWb7cxPbGHHjgJyJgVPD/K5Exc6Jtzsa1jvk0LuCwWsXtL+ldSnpitVMc6DpcFftDfqe1ct2oE3X8YmqhrJg37JagqeyyXNFDzdp+aEC4YmMGOeasZ96yjesY3V916O/Q/Fd64TyVU/+Zgzj4lx4HJUd1F1G8KeI5xsBcPkHMxBM8zkvrSDHRtzMF2K7YbkrQ6tt1Ea9JZxInpdwJ995b4VFLxcwAr/9eWk5q183jvkAZ0+KD+Midnehvo+ZnImG6D1JEf29KA8fO7EhZaEGYEvCACks654B9ueSfefk47b1JP1TAHrnkjDBCQ/vI6Cl9cFvKBjp+LV31BdB4wY7/8xYUw0q9YjhBCKkMHY1qkH+cOs/erJV5f9pZTXuiE+mdyNK8iYm0zy7GRSM1dQ8K/J6HFz4KNSAKzFH2D36LA8WEDuvRaM6DHfns26nyVjjI8Fb0Pqkj9V44o0kPL4OrJvN6OPN5OydDX/MkOPxnWcqgPBuxCksogPjnjQTV1CwWPpWBKUvqSy1y0hOWEUsdj8SV1NHtAmMHNpKqkPpAU8OegF7xeOfmou2XOMgBHLvStY8xNTz6rZuqDki5HbfrFayRffNu4yY4jX0tR+aEGUfNdiztzCinstGDFiuTeXgjQT2lYnX350efY/lJLXq3BG6knO2eB9C9e7bw9a0HkcVO9pbxEY3vkqoXyfC8ak8C/PZZOSqFfO/5p/ISleQ4OjClsPtxseF+5m0I6bSfbcVDLmJ7cHm/EWcpemYASM09NZsSoN02D18n3glIbYOANJP1sXdA5vS9BA3ZHgJ8Ljb2PdmoD8ee5fSBkDrn3llNCD8uC93rSJGWx5SrnejNPTyd34E0xacNqs7eckYJskWEh/ag2pUwzoBzVhA2zNGkbFmUhdtYK0RD3Em0lZ+s9Y4sHzdS3lQPJT29iwZjWbnuv6TVQhxPdTyGDs+8FJ2bO/oHSfE821SaQvzSX3sVyy703CoHFS81Yha30dvtaVkL+lHDsGkjPz2LBzC6sfTsUU7aTqt/mU+H6Bv7+erW/V4Iwxkfrwara8tJqcuWala4Zf5Sv9FnXKScmvXqb8EBhmZ5D3wk62rMkh1aTB+VEJ+cXtTyhKf/8BDo8W09xssu+eepFPgEp47V077hgTqY9uYOfODeQ9YKbpfaXLgz7x/nqef72ahqFmJV92biAvMwl9k42yX6/vpO8pb77/1orDoycpM48NOzeQl5mMATvlrxR6G7Ffhv0PZW8+z26vxhltVN7C9e+bA+tv17M14LdMuOer4sWtlNU6iZmUSs6aLcr5T4yhYV8p+Ru9pacH2w1PKbs+cuAZaiJ1aTapU5W9K3m9HLtbi2luDht27mTD0xmYm95Tuunoa77rJjrgHGYmoffYKS8uDHjS56amsoaWiQH5M0mH60AZW1/0laJwy4OTkl+VYD3mQT9Dud42PJ1B8rVgf/dlCl9XUle8+Dyl+xqISVS2ufOFPDJm6GkK2KazOJ+SSgeeuCSy1mxh50uryZlrQlNfTekWpSLz27NKm0nP+YBdEEIIr5Cdvv562pdcG93CTz9S1bddrbw9pWsAuuwhXelhfFhUd+mMWGaPUTr7bDqOdW/P3qlSekZXXpf/9qD6jTIfZV80Tmt4vdh3JyAP3I4+WmcHvc2XMPL9sux/KGHsG/TsfCVYSDYonbF2fizhbjdM8WaSJmpwBr0BGLCNHp2v3uqifHg7fT1UvIS1e8M41z0oD2Fdb+Gck4BtBo9koNDH63HWXdRZEkJcpSQYE0Jc+QKDscAhyoQQ4irwPa6mFEIIIYTofxKMCSGufK0eGpsb8bSqZwghxHefVFMKIYQQQvQjeTImhBBCCNGPJBgTQgghhOhHEowJIYQQQvQjCcaEEEIIIfqRBGNCCCGEEP1IgjEhhBBCiH4kwZgQQgghRD+SYEwIIYQQoh9JMCaEEEII0Y8uSTAWOeACCbFnuTv+7zxgcJI2up4bhjYROcDf2b8QQgghhOjr4ZAGALPjXGSN+5oRUefVs2lqjWTXcT1vfn0NrRcGqGcLIYQQQnzv9NmTscgBF3jIeJx/STgaMhADiIlsZfG4E+RN/ApNRJt6thBCCCHE906fBWP/dN0J5safImIANLdF8rpjFA/uvYGffjSFRVU38psj1/JNy0AGANOHn+bRCQ7k2ZgQQgghvu/6pJryhiGNrJp0hJiBbXzTouH5A9fxtyatOhkjozysnmznOm0zbRfA3RpJWw9DstpvY9hgu049WQghhBDiO6lPnozdcs23xAxsw9MWQdGR0fytScsAYOIQNw8YnFh0Z4gccIFTLRp+ffhams5HEDEAYge2MnTg+R59Yge2qjcvhBBCCPGdddHB2OCIVn4Q6wbg2NlB7HMNBWDe6L+z7sZDZIyt45nJdrLHf80A4IszsdScHsLp8wPD/jSej6RNXsQUQgghxFXooqsphwxsZV3iIcZFN7P/21hW/++EoGk+p88PJK82geNnBwUtH44x0edYn3iIoQPP+7chhBBCCHE1uOgnYwO4QATBj61CTePChR62DhNCCCGEuPpddDDW0jaA5lZlNbEDW9FGtnH6/EDeOTmS896+xC4A1lM6jp8dxADg5xO/ouSH/xvys81yAPOwJgAWj6uj5If/y4uJfyM2MnR3GUIIIYQQ32UXHYw1t0XyZaPy5uTY6HNM1Z0GYM+Ja1j1eQKlx+J57gsjRUeu5QIweUgj5qFnOjTM938059EMUPog00YqDfxjB7YSIY/VhBBCCHEVuuhgDOCDvw+j6XwEmog2ssef4PoYNxeAg2e0vO7QU+0aQuuFAYyM8vDwhK+JGdhG2wVoPB/pb6Tve7oWygVv7/2+xvxCCCGEEFeLziOgHvjfM7G8Xz+CC8CIKA//T6KdjLEn0WmUqsXBEa3MG/13fmn+kuu0SqP+yr/rWPzfN5L13zeQ9d838H79cNVa27W0RfAr2ziy/vsG6WNMCCGEEFeVi36b0idywAWyx39N6iilF/7OXAA+bRjKi7ZxeNraY8GHjceZG38qKK3PubYIXjx4HdWuIepZ3wPJ5Dy3AFMMHP/rcvLfUM+/DO5bQcGPxkCTjV3PbMWqni+EEEKIXuuTJ2MArRcGsM0+hn8/NI5vWgaqZ4O3qnH70dGsP3hdUCDWr2bnsG5jAQUbCyh4OgO9anbyw+uUec/lkKyad3kMJ26cAcNYA2PUO3e56MdgGGvAMC6Ozp9fCiGEEKI3+jQiugD8V72Oh6on8681P+D/2q+l9Fg8RUdG82+fJ5D13zew+3gcrd63LAPZ3dFUu4aE/OxzDaGhkwDvog2JUwKNsQYM09PIfSA44hke753X20Bkbh5btu9gx/Yt5M1VzxRCCCHE912fBmM+rRcGcKgxmj/VXcPrDj1lJ+L439MxIYMwn3dPjmDtF8aQnxcOXofdHa1e5BLQYLo9mxT15IsRqSF2sAbN4Fg08u6BEEIIIVQuSTD2naazsOBJi3pqaAkW0pfmkvtYLrmP5ZAxxxg02zg9meTh7SMODBqeTPLsZCwJAYnizaT515FL9nxzh6rSQPrENLK7TavHfHsGOd50uUvTMMer03jFm0nJzAl7++2MWGYrx5M8K9xlhBBCCKHWZw34v7PmraZ4iRktTuxHhmMcrwGPnbInV1JSB2lrislK1IK7hpIlaynzLmZZuoHcO41oVU+7PMesvLyhEGsd5GzaScqY4PkAjvcXsvxl0M/PY80iC3pN8HzPqWpKf7GesjqANFYXZ2HWguu4g5gxBgKTe+qrKV3rSwvEp5G3OgNLnGqlrS5q3nietW/Y/ZM62z6uGkpfWMvuQ96/Hy1g5xxDQB5YyFqXS5pJC3hwvJ/P8perVSsRQgghRDjkyZhfLI17q3C2AhojqY93bMzvN281uXO9gVirB9dxBy5lrHQ0Y5N51LtsS7MHjydgOY8Hj2/alBzyfIGQ20F1pRXrPgfuVtCMtJCxLD1gQYVujAFNqxvXcSdu73o1cRYy/PtqJmdlQCB22omj3o2nFYjUYX5gBSume1cWuH3AXe/Aedq7Up2ZjH9dQejng3rS1kggJoQQQvQVCcYCNRWya58LAI0phew56gQAerJnm9ECuG2UrsrkoSeW89CSZZTUKhGZxpRMxhQo+nkmmTtqUKa6qdmRSebiTFZug7R7Z2LQoDxteno56zcXUvj8cp56V3lypfnBTLICNwtwuoaSJ5bw0BPLWJK5norjSvCkSZhKKsCceVjGKtGVs7KQZdnLWP7oEjLzrUqQiR7LT5QgL2WeRdl+qxPrpmUseXQ5y7IzWV/pVLYVZ2HBfb4Nt5uw5pfKk0I8OCpfZr0EYkIIIcRFkWBMpaL4LWrcADosC1fQcKpRlSLJ38WE63/fbq/Kw0nZ72tQQhk9Y6b6FwhBz4RrlCGkGDSJjF/uYMd25bPpdm+7s0gNMUHLgHPfK+3VkVSz9c8HlEAvchTj54H+RgM6gFY7VZut3n0B9hZiPeQN3OInkowe82idMu9YFYUf+lNSvdmKrRlAgz5B1ZmH1kxyone/66rYHrgNIYQQQvSKBGNqdWW8UmHHg/J0aH6HDv8NDPfGI41nVN2f7m+ixftfjTqSCpLEsKHe/0Zq0AwO+Kjbb/m5OXlUFfrs/ZbAUDFpZKzyn3ONNARMB2hs8VZBRmkYHrB9d2OHlJxThgYlanAXnXnEJ7H4YbN6qhBCCCF6SIKxEJzFr1FVB6DBOF7dcsxBg7d92Ii4tOBZc3Qo4ZAHd+jBBLzKqPfGQB7bbhYuXBjis5ytQctoGXW9KviZrW/v+6wVyuq/Uf6vHY4h6O1JPfpYbwTZ5OZIwPa1ww3BbePi9cR6XwBtPHMkcA7gwbXXSs1pAA2G2YvJ6uwtTSGEEEKERYKxkKop3F2N0npMrYzao8pTJu3kVFbM9VYrxieT+xOzt5rQweFKb/JW33Jaho82QnwyydOh5oS3bVpCMivu9IVDetLWFLPztR0Uv5BNkm9R39zpS8m92Zs2IZ3Vd5qUNyubHXz5NmA94q02NDDziSySvYGS8b5HuHW88n+Xo5oawHrE+5RtzExWLEn2BmRG0h++FWMkgAvHvholjY/7AG+9WMgrb3rbwXX3ooMQou/ct6KfRwMR3x8WstZsobiogFz/95O4lCQY68z763nL2yBfbfebH+DwAJF6kpZuYOeOHex8KZfkMd7G85+8RpGvbdfbtTiUsdEx3LmBnS/lsmAGWIsrlLZZkXqSHtqitBnbsUVpHB8J39g/oap9kwqNgeQntijbeyEDs7eq0VX7HqUA+0sp9+6zNiGN3E072LFjJxse8L5w0Gyj4tUKAGpeK/c+4dJinJfLlh072PHaBjK8bcI8tgqK3vduV8X51iuU27xt0EypPDL/6rtY/cNgbSwgL1N9fMnkPKfMW/ewfC2KYOlPeYdXW5cTugPpgCHY8hab1HM7J8OSCcxkPeMtXxtXk3Wpep+aN4/bEvVohxpIvnuBeq64BCQY60LZr9/DHtg1hc/eraz/rRWHL1bzNfRq9eD4qIhnNwa+Ybib1/bU4PI/IfM+LasrZVVhOfbTyqT29mIenHtLWf9r1VMp3NRUVuP0BGwPD87aMra+qARY4KTs2ULKap1Km7fIgDZoLhtlhaso9QWJdWWs3VpGTb2vjwyNf4QA14Ey8leVdtE430npS+XevNFiXvAIqgrb7zz/MFhjDVjm5pIRVB3bPl6oIV6+FkWwLz1RStkwWUgK8UZ2ys0WTGMNGMbGwlGberYQnZuSQtIU373JTNLtF9duN/Vp74OAl/OUN/J9jtbzjferwX3KEThHXCLS6etFMk5PZkwMgIdvD1ZR43/bUU2PeZYJjdNKtf8NTEX468Db8/0YtIDb0XFdfvFmkiYOU6oxm45j3dve2auaPjEJ00glautynd8j/s5+vVx7t/KQP+ht74jXXVvCkmd9XQELAUzJZcszStW/a99WHnreV24AUsjbloNFB9RXsfbRfNQ/uzrVofNl8X2T/PQ2cqd734QHcFVT+NB6VK+Sha2zTs1BGWEmWe/B9pGvlwBxKUkwJkQI6mAMnFS9uIz8vXQTjOkxz08jZZz3zVb3MT6o3B0c4PoC5ZZvsX10EtMDC5gWFwXuw1T8qYyausB1tODct4vSgO5HfIxzMki9UU8UgMdJzZ9LqZBA+gpgJvel1UqbTVc1Wx9ajz8cm5PHtkct6ADnR2tZttEXioVRbjoEY8oPvGFR4Dllo6rWV0Y6m+6dF7QdX5kLSCKuUO2BvPu4A8YY0OKi+uWHWB+qSUm8mbS7U5jge/v/aAVlb/kCK+VHfeKdDykdeLttlL36Nodxc7yyGntAGerwAz3eTEpKMuYRUSHW65VgIdmg9T4IgJTMVCV9Z/ephBQybjd7OyFvwfl5OaXvd/4A4WokwZgQIfiDsTo79pFGjBrwHCnjiadKcHYWjCWks/rnGZgDfriCdyiqHatY+5b3duUbgsttp+aUAbO3o15QOva1Hk1o788NlCrq7U+1L4+F7BdySU0ITKOks79dyMpXpSPe/mZ+Ygurb9aD6ssy5d+2kTNVBzixPreMwv09KDcdgrFOymFn0zsbKs3jovq1Vaz3ly9xRfIH8h5sb7wB8zIwDVafe0VnQ921D7WXQ8HOFAzBswEHFQuXszWgDPmG76OL9XY6hF5dDTWDzcFl2+OgYuNytu5V/uxsaEHX/hJWPVf2vXkqJ23GhOiKtpF9f1VuB5rxqeQ+oG7M76Mn65H7lJuOx4ntIyvWymqlXWGkDvPCXDoMcKU1Yh6rjDnq9LU/HOrtWNftxHHcpbT9Q4v5jgx8rUPSnvEFYh6cB6qwVlZhq/coL2PMzSZXfkP1u5oKm/dLRMeEGb6XPNJInuj9VqqzUbGf3pWbXgkYKq3VjWNfwHY0Oix9th1xqaSneN/Wb7ZT+/puPj3ivTtcPzP43AUOdecdrs831J1mpIWMf81CT4syNF9AW2ZPszJcn6+vzA56M4RevBKIeU45cHrbR6MxkPKAb3yZLBZ5AzHPKRtVlVaqDij3Pd2U+Z2MgnN1kmBMiK4MhMbNu6h2AWgw3Z4d+g054PDeD6iqtWHdsYxVGwsp3Lye5b/1dpEy2EDiXPUS4KzM56EnlrPs6TLs3rducVWzdckylj/xEKW+N3qHDmM8QHwOKVOUJ2LOynyWPZNP4eZ8Vj36G+8+6kmcK2949rv9pdQeV/6ru/4WpSuKeVPxPcx0fFHqbyvWm3LTY/PuZeZYjXdYtqdY/rx3O0+XY28FBpuYuUS9kLhypDNtvBIFeY7WUgrs/szbOflgI9MChq7zD7XX6sT60hPK/SX7CWW4vlYP6MaTTBErF2dS+oX3/uKuoXRxJpmLV1LUvqogvR1Cz1m5nsxHlrMsexnlvq4r9eO9L33FeF8c8+D4sJD8zYXkP7OKl39dROHGf+/0jf6rUchgLGffRKmiFMKvgqIyb99qOgsLnmzgW9+vPD8n1te3suuTw2iTfK+eryPvRm/bHADVY3hwYv+Lt0qxroFG78gHbketv42RvyNfnxlxjPD+d3jSCv8wWju2/zPmIcr0LkdOEJeJk10271tougncMhvSpiUoXcy02qnd7at86U256Tn9+FHKttEyaeGm9nLzy9u8/Qp2N2qI6FcPTMM4GCVoOXqM5NnJJNc7lC6W0GBK8j1pChhqL2ioOydlzy5h4aJMMh9ay27v1PD1cgi9VjtVm33NJpw4G9XdRR3E6f2ha5y/hR3F2yh44RGm6Vs4/j17cSBkMCaECOZ86xXe81YL6KfPZ4w6AZD21DbWLU3FMsn36rkJy2yTUrUQUgtN+9XTumEY7v1S9XaHEvjpgy9t0Xecu20o4ZgOw7QMpvpaUh87GDDGbG/KTc/5h0oLUW7ElS9rureDbzQYb88l97Fcch9Lweg7fWMTyY4naKi9jkPdXYyu1tvFEHohhuYLVsH6F4qwHnPjaQWNVochwUzyvTlsKN5A9nR1+quXBGNChMVJyetVOFsBjRFjh2GgckiZoUMDuGt3s/bxhSz8+XoKX/c+UesrRxu863NS9Zx6CC3lo27MK/pJ3S5s3qpK/Y2p/ipKe21RwC/+S1BuAoY08/E/YW22sTtEmVm4sL2RtrjCxGeTOFY9USXSyMS79UFD7XUY6m5OOhlzLXjHjOmhLtbb5RB6YThUTuGTS8hctJBlz+ZT+me7Uva1Rm6dG9T72VVNgjEhwrW3kF37Qg+SxVxf9aGbQ/tKla4CDlXjHuerHuoje45wshWl2iAzq73B7PQcCnbsZGfxNvLuDVpC9JuAqsqhWv8oGPuKA5JcVLlpwONtba0dN5W0eFRDmgX43Oltg2Yi+alU/5epfv5qinfuZEfRBrJnBC8irgz69ETv+fRge10dRJd6qwjBOFV5yafmuPceNWYmuYu9d4j4ZHLTM0hfmseG7ev8Df493idaDBqOPgH0Nyd3bITv1ash9LqTuY7i7TvYuXMLK6aDs7aK3b9+jRrv2M7aoZ29MHX1kWBMiB6oePEtakI9svAPe6XFvGib0vbn5WLyZvX1zaSE1z5RborahDTyXlPa/ux8OkVpXHv+GLW/Vy8j+otzd63SQN7L1/ja76LKjZUqu/eLd6iZrE3BQ5oFqSyiwjuEmX5GtjL82fYdbFnsHSrtWzuffKJeSPQ/MxmTvR1QNNv59A31/Pa3Kok3kTIFrNvf8g91Z5qfx84dO9ixKdc7VrEHx0e7/G3Gyr9weEdrMZD6wk62PLGgw5jIPhczhF6nKg5yMlID6El6eqd3WMA8kkYCeLDXlquXuGpJMCZEj5TxSoX3LaYgu9n1rk0Z9kqjU9r+xGlwflTd541Qqzc+S8lep/JaemRAux+3nfKt0jP7FaWujIPHfH94sO8LCsUuutxUvPoWNb6HtZFKu0H3oXKqvNWj7ZyUrsqn/JD3l4Smvdx46qsp3bA1/JEAxOUzJQWTt0mE58inIRve+9+qRE/iXZbOh7pr9Q6193JAP4RvvEZZbSdP+9U6W29YQ+h1oq6E/N9W+cdv9g8L2OrGUVlCfnGP1/idFbLT14iICAYMGBCcUgTRJ6aR9uMJKM1iGzn2cQW7Qw45FGaP1132ltzzdOr9O/xfZZQF9cQdXhrRQwHDUHXoubqv9WDIK3GFu6hy01WP+yH4ekcnzPTiOyvsofbizSRN1OCsrCacu0jfD6HXXoa73derlARjPWYkfc2/kZHY8V2nDj0Gh9njdWe9Grf3lqz8HV46PWlPryFjut779o0/FY7381n+cnWYaYQQQghxOUg1ZQ/pl+RwX6KO9t7PrVQfUx7966ZkkOvv9C6gx2uUHogd9crru2h0WBatICs+3N6SVencDqorrVj3OXC3etMt8zbJnJ3NfG+Q5T5WHbB/Ggy3LlJefw4njRBCCCEui8hx48b9wvfHhQsXaGtrY8CAAfJkrBMxsdcQc+E8nqP/yb+u/y1Vn1Rhfec019/5Q0YPjiS67Qx/+MvfYN4jPH7raDSAs7KQvNXb2P3HP/BBy2Tm3DQKPAM431jG0B8/zi2jNd7ekvNYtW03f3rrA5pvmMNNI6Et4jyNZZVc/4g3nbuGkqf/jaKKKqo+eIe/Dv0h9/xgOJHDY4l+4z/5zJzCff8wCg0uql9ZScEfqrC+c5RTjdX89c8VvGlzQThphBBCCHFZSDVlbySkkv3ALSR6ezp2//0IGJMx6doHbdU/toUts/XQaqds0UpK1OsAQE/uS1uUt1yOlLHwqdCpgtK1evAEth6P9DZ49A3wGp/Fho1p/s4APaednPzaQe1/72lvWxZOGiGEEEJcFlJN2VPzVrDtuWxSp5q8vWUbME1VArFA/h6vu+yBuKtejQO1p/O9Pef/qNqPUVdC/pYyauqVqlPNUD2GSRZSF69m0yt5Sl9E4aQRQgghxGUhwVgP5aQkodMAp2vY/ewyFi5cyfrNpd7+V9r5e7zWDscQFNykkJ6ZiiWBrns1DuotuT2dx7a7Q8/Zymc5W72LOj8sYe2jS1i4cCErXyxi9z4nHm8btJSfhJ9GCPH/sXfncVGV+wPHP+w7IgiCLIoLSIImLrigmLlValfTTCu7pVZaZpblz9vNbouVZZnX0hat1MrMskVvqZlGaYoppkgKqAiCwCCI7Du/P2YGZg4DMyCLyvf9ep3XiznPc9Z54Hx5tiOEEM1PgrEGGYd7O/VPhcnH2HxSBSQSXdCVjtqaK63YVPWM1/gwaL52pnQPwh+/i+mTZrHk9S9YNsX02ZJj0tT5rLqFs2hM9fzZTHhhA1u//IINr88iDAh7ciVffLGVrZuWMd0TEo/sZPPHNfO4OLqEm5RHCCGEEC1D+ow10OSXv2B6TyugjJzUDPIrrXHt5IG9ZvI7bZ8xdaD0JjO1s2FXlFFWWdOsWJa6lxVPriXacwJLl80kRBvMlZVRZq6dTE9nqgnP6Sx7czIBtppsxWU6/cXKSNnzGgs/iIFblvDRvFD1S4a1/cuqJ+fLIXrNHF7DhDwNnUlZCCGEEI0ioykb6FSeAzcHB+Jma4GtszPO7RxAFcXxfB+8HKFMdZzvIuOBAuIjk7G7KZCuHg5YmFtgoQnYyjKj2bxsJZH5QH48kRftCArsSkcHC7CwwMJcM1ty9GZeeSeSAoD8k+y94MyA3t1pbwMWltr9qWdVfmXV7+p85/eTanszId3dsLWwUOczB8pUxPzvQ5Z9n2ZaHiGEEEK0CKkZaxSd2YJNmfncxBmvTZ0t2bR8/oRGeGteNlxIqsGZlU3JI4QQQojmJMGYEEIIIUQrkg78QgghhBCtSIIxIYQQQohWJMGYEEIIIUQrarI+Y65urtw9bRpdu3XF3LzuGC87K4sNn24kNSVFmSSEEEII0ebUHTU10O133E637t3qDcQAXN3ceOCfM/H28VEmCSGEEEK0OU1WM/bUM0/TqVMnSktKyMzMpLKqerfVnJyccHFRv8SxoKCA7GzNK4PqUVpSSnJSEgcPHiQ7y3h+U61Y+ZZylRBCCCFEk1q08GnlqlqaPBi7fPky6z9aR3pa7cmvXN1cuX/mTHz9fJVJRknzphBCCCFuRPW3KTax7KxsNm3cyIXkC8oko9q7uhISEqxcLYQQQghxXWvRmrHGGBYxnPETxmNhYUHkr5Fs//4HZRYhhBBCiOtWi9aMCSGEEEIIfRKMCSGEEEK0IgnGhBBCCCFakQRjQgghhBCtSDrwixbVN7Qvg4cOobNfZywsLZTJog2oKK8gKTmJgwf+4Fj0MWVynaTstG2NLTdCXA8kGBMt5smFT4KZGT/v/pn4uDjKysqUWUQbYGVlRUBgIKPHjIaqKt5Z+Y4ySy1SdkRjyo0Q1wtpphQtom9oXzAz48P3PyD25El5mLZhZWVlxJ48yYfvfwBmZuqyUQ8pO4JGlBshridSM9aC/Lt2JSAwAL/Ofni4u+Pk7IylpaUyW6OUl5eTl5uLKjOT5KRk4uPiSTx3Tpmt1cyb/xiR+yKJPXlSmSTasF7BwUTcEsGa1e8pk6pJ2RFKppQbIa4nUjPWAiJGRLB4yWIem/8Yo8eMJjAwkPaurk0WiAFYWlrS3tWVQE01/mPzH2PxksVEjIhQZm0Vnf06Ex8Xp1wt2rj4uDg6+3VWrtYjZUcomVJuhLieSM1YMwrtF8ptt99Ge1dXAC5nXybu9GmSkpJRZWSQl5fXZE0uVlZWODk54dGxI507+xHYsyftXdsDcDk7m59+/Inoo9HKzVrMipVvmfSyVNH2GCsbxtJF2yTlQtxIpGasmdw1dQoz7ruX9q6uJCcls2XzFv77zip27dzF6VOnyM7ObrJADE1/iuzsbE6fOsWunbv47zur2LJ5C8lJybR3dWXGffdy19Qpys2EEEII0cqavGastKSEzMxMKquqd3tVbGxscHd3x8zM7LqoGbO0tOTB2Q8RGBgIwE//+5HDUYeV2VrUwLCB3HbH7QDExcXxybqPKS8vV2ZrVvJfrKiLsbJhLF20TVIumohtRwZGjCAkwBtnE3vOlOelcj4ump9/T6BYmSgapcmCsftm3kefm29u8HamKi0p4fvvvifqUJQy6Zoy59GHCQwM5HL2Zb7Z+jWpqanVafb29oT0DqFr9254enri5OQEQF5eHunp6Zw7c5aYEzEUFhbq7LFpeHt7c9fUKbR3bU9cXBwfvf+hMkuzkj+coi7GyoaxdNE2SbloCp2Z8Mw8IjqZGIUp5MVuYfm6w9d2QObiTY+O9sq11QozEkjNUa5teU3WTPnj/37k7JmzVFZWKpOuWmlJCQf/OMifh/9UJl1T7po6pToQ+2zjJr1AbOSoW1m0+BnG3X4bAQEBODs7Y2ZmhpmZGc7OzgQEBDDu9ttYtPgZRo66VW+/TSE1NZXPNm7icvZlAgMDpclSCCHaugGjCWtkIAbgFDiIcOXKa82IaTzy6KN1LnePUG7QOpqsZqytC+0Xyoz77gVg3QcfVQdi7h7uTJo8Ga9OXoot6pd2MY1vt20jU5WpTLoq3t7ezH5kDgBffPZ5i3Xql/9iRV2MlQ1j6aJtauly4eQ3kLCRofT19cXN2Ra9wfDl5RTnZnHhQjTH9h7mcHKuTuI17B9PsSLCG4DUyKdZ+Z0yg2F9Zy/j3l62QCqRC99muzJDK6g5p6ZRHPs5/17XMs9HAAs/P7//aD9UVVVRWVlZXWMjTPfQ7Iews7Pjp//9yOlTp0ETiN038346dOigzG6Uk5MTgT17cvbMGQoLmq7ZMi8vj6LCQnoE9MDHx4fff/tdmaVZjBk3lt27ditXN4r/LdOZPmM6906awLix47h1UAh+7fJIjVNRoMzMSKY/PpFbBw5iUF1LHw8Ko+NRASETZzF99HAGDeyKw+GTJCp3p8NjzExm3zZCkTeECXOmc1u/mn1WrxsSinflEU5dVOwI6thOez6heBQfIz5DsUlD9J/M3MljGBrYnsQTZwzcJ+CW6SwYf2ut+xPsbYW1ZTGpKuVWmnvb3YGomPruVP2MlQ1j6aJtaplyYUvHsAk8/PAcJowIprunG452lpgr25TMzbG0c8TNswe9Bo9gzJBgHIrOcT61gJbtndtAPQczposzAHlJuzmkfnQZ5RV6K709LIE8knYdJF6ZoRXUnFPTKM+MYW90mnJ1s1EWKdEIESMiqkdN6nbWnzR5Ms7O6oLeGM7OzkyaPFm5+qodjjpcPcryWpmHzCSe41jw9gbemDeZkX0D8PH1wcfXB//gMMbdv5T3PlrGzP7KjQIZGBFOeH1LeF+6aHJ36Ttcs34Ct9bbkhvC9AkTNHkHoh6uAdCFvuH6+6xeFzGS6Q8sILR6vS5D22nPZzh9/XRWNsKEO+5kZEQ44XeMZIKnMlUjaGDtexMRzrgZC1j0wnt8sWYJk7vpbqC5twNrrl6IG4ZLKHcvfpFn7gnHW/fPeHkx2emppCbrLOlZFOtGXc7eDL3nWf6zeBp9XXTWt6Zh97Jw4VMsXPgU9w5TJl7/8lWJpGbX9F4rzlZ8R8YWxbaJqrzqzy1BgrEmMGjwIAAO/nGwet3IUbc2uGnSEK9OXs3Sh0x7rtpzv/ZNYNF/ZhLua09ZZjx7v3iNZ6dOZerUx3jpg21EXSgElwAmzF1qONhIj2LVu6sML+9/S+1hIVYEhM1UrqzReywBho5jjGcY9z8SolzbzCYzqLumA6uFP6H31H981UHd+7OebXuiiEkvxMo9lOlPL6ojmBTixmEbMo1nltzLQE9tTUs5eecOs/3DF1n0zHO8unwNX+zYzo4ft7Njx+esXf4q/37maV78cDuHz+VW14ZZeg7k3iXPcndI0zWfNZpbR7z9vPH288bDTZl4/Uv4YR0rY7KqP2fFvM3KlQ1YFNuu/yGh+nNLkGDsKvl37Yq7hweXsy9z+tQp0IyaDB+m363xQnIyGz/dUO/cYqWlpWz8dAPJScl668OHhWNvX/dokMY4feoUl7Mv4+7hgX/Xrsrka86E56cQ5mZF4dntrJj3HGu/jdY0C6qI2bOZFU89w8aTheAcwsSHRio3h4oC9kfuN7wcjKluFlTLIScL8A1mVh0B18jxgXhQRmFu3d9nLYUqVMVW+ETcz8w69tsspvXD3xZyjkWRWAEe3UZSXzhWWqx7f3ay+YMVvDR/FfvTAfdQ7qq3xlCI65tt2D9Z/M+BdLRWfy6/dIqda17gxdVbiDwFfac8xQtvLeOZeZpO4POe5ZW3XmDhlFA49StfrX6R/6z5iVOXNCGZdUcG/nMxD4RdAwGZuGZJMHaVAgIDAIg7XdPYHtI7pFafuz8O/EHiuUQ+27jJYEBWVlbGZxs/I/FcIn8c+EMvzczMjJDe9T0+G0d7ztpruGZ5zmJYL3sojmfnOxsx3KVSxfYPfiE+S8XlMheu7oqsUWWqwMKf4MkeykRgJGH+LlCcSEqelTKxHhns359CmZU/4+ZPx9Cem8PM/gFYkcO5Ays4eQHwDGBsb2UuY6JZdSoFsMLFs6XOXIgW1mUCc6eE4GQOUE728W9ZtmwdexKKgc6MW7CYe4d6a9J1mDvjPfReFi+4jc5AccIe1i9bxrfHs9S1ZObOhEyZxwTdPgiiZQyYzSsr32LFymXMGlDPulamLFKigfw6qzvzJOnUZnXtrtexBjTTXvh19iM5KZnPNm7Sm3RVHYht4kJyMv5d/Zlyd+2qB0P7vFrac9ZewzVrbCD+FlB45hCb63vLVvpGnnv0MZ59c9tVdygt+fMkKYBP0PTatUhTbiXERX0+Kco0I/I/2ERUOlgFjOPR8crUZuA5i2BfIOccv0fCxpOJgAeB4w3UHgrRpgVx7/0j8Na0TObFfsPbn+5H23Oo4z/uYkQXde1Wefphvlr5IosWPs2LK7dwOF3999y2ywgm/6OjZotcDnz6NttiNSMrLb2JuP9egjSpQuiSYOwqebi7A6DKqBnq5ulZuw3K0tKS+2beXx2QffHZ55SXl1NeXs5nGzeRnJSMX2c/Ztx3r8EXiBva59XSnrP2Gq5VE7qo/7hlnL+KAdQWDrU6pquXMEIM3drKbzh5Xl2LNFJRizS5jz9W5HD6t8acTzSrNkShwp6QO5fQ3CGRx+3qQFZ1agf7ATYcI74YXPzDGnhsD2b5+wBl5GTqN+peFzzHseDl99iw4SPeeHwc/g1NJ5SZ/1rJRxs2sPJfMwmvVWaMpUPo/UtY+f4GNryzhJlDa9cueoxZwLI1G9jw/hssuK32GRhLN3oNxtKNXoOxdOPXaCy9NXWcMoG+6tcIU37xVz7Vm8w0lPH9vLEEytP3s3b5lurpK/KSD/PV8jUcSC8HLPHuN4G+1dsVc3jdp0Re1Pzz7RrK+CnaYE2IGhKMXSUnzWjJvLyakRfamfWVrKysuO/++/H19SXxXCJbNn/J5s83Vwdi991/v8FAjHr2eTW056y9hmtbIfk1/Ss1/AmtFVyFEx4RWvtB4xnGgscXGFjmManWCEwAFes1tUjBt+n2/5vJoB5W6pqmfTqrG+LICr45kgMuodz1VHN2h/fgrt7+gIrEPTGadZs5mVwGLj0Jr6NmztpWOZpyLkvefpNxXYCcGH7ZotziWufBzKdnEt7TA3t7F/wjZrHoSd36TmPpMHLxXCb09cHF3h6fvhOY95j+KGdj6dyyhLkTQ/Fxs8feO5QJjy5AL4fnTBb9M5wAd3vs3fwJf2gRC3T/CTCWbvQajKUbvwZj6Uav0Vh6qwrnH/00QVJ5Kgc2bSdJN9mzBx0dAcpJPPStfhoASXx7KFHdJOnoQQ+9QDWJ7Zv2k6qJxzr2m8RQ3WQhJBi7etrgSbcfmLK/mC4rayvue+B+PL28OJNwhnNnz+Lt4819M+/Hyrru/kf17bOxtOdcVwB47RvH/bWCqwUsePx+ximz5sTX7rgfuZ/9kb9xTH+8RI0Nh9S1SN2HVdcieTwUrK5pitXUNDXS3g0/EFMIHgNnscBgMNgEek8n2BtIj2fXiZrVm48lUoY9PcMMPwo9Buvfz1mTRhLqa09ZZgzbP3iNvcoNrnlhdPHQ/93y6KIbYBtLh0Av/fkJrPyCmaDz2Vg6QR7o5bD1IVg3GO7fhY56p+CB3ikYSzd6DcbSjV+DsXSj12gsvTUN7omvpn99efIJflZ2h/B1Qj2EqpzyuqZ9LCzXjKJ0wMlXkZb+MzHJmmjM1peegxXpos2TYOwqaft+WVnV/KGrMuEl6RYWDbv1puyzobTn3NIvDW+oy8WlgD3t1RNF64jjsCK4iq/rHWMFKbWntNBM3bBdJ1DRt41DZwp1apE8mBDkD6Rw8kttTVMjpW/n/R/jKbPwIGzazGbpzB8yKkC9X/sAHnp7JSu1yxD10ay69GO6ciMgJ14RsO7ZxvrlzzJj3ktsPKLMfT2I4rxKf9BMTqru92csHeLS9AtWWdp5vVnHjaVzSoVejrIMzu/Q+XzkPBl6p5BDaqzOR2PpRq/BWLrxazCWbvQajaW3poMHOJWt/tGy61CmKTt1x2RxRZ2KvZvhUZG2bvao/629Qrbyz8OAexjaVfNPb/bf6MyCJARIMHb18nLV/QZ0mxF1myyVtKMmU1NS6da9G926dyM1Rf3eSEOjLLXq22djVb+oXHMN16r951WUAT7dlfN+7WWzXmB1jhLNcPSmsv230+Roa5F6Tye0C3D+JN8o/3NuBNWWVew8X4ZVl3EsmKZMvVrhjA/ShHjOHtUT5KoXF6wAbAPo+4BiMyA/VRGwfrCZnUcaP8N+61Ox8a2N7D+bQ2FxISnHtrP2Td16TWPpsHf5WrYfS6GwuJCcs/vZ+M7GBqWz7zXW/hBNSm4ZhVmJ7P90BXo50jey4tP9JGYVUpabQvQPa1kR2YB0o9dgLN34NRhLN3qNxtJb1Sm++S1B00fMmZDxsxmoG3MV/8EZFYAlnYf+k6HKiVxdwnlgaGd1MKZK4A/dN2fbDmTW+BDUf22LSfjtG9STIAlRQ4Kxq6TKVL870qNjTafM9HTDT2rlqMl7ZkznnhnT6xxlqauufV4N7Tlrr+GateUoicVAlzAW1dOk5zFtED2bdjo22Pc753LUtUgzR6prmhJPrlfMS9ZYKjZu+p2UCisCbh+GY5Ey/SrcMoyuLkDqXh6bOpWpymVDDIWAf/CsZqmVu+ak72TV/83hgfsfYOGrBqZHMZZONBtfXcgD9z/AnP9bxc5av47G0iF602ssnDWDBx59llW7a5cg1e5VPPvoA8yYtZDXNtU+A2PpRq/BWLrRazCWbvwajaW3puLIT/k5QRNFOQcxef4kelQHZBn8tPeUemSlYw8mLfkXj0wZQd/AUCKmPMq/lkyihyNALqf2/kT1cC7bHkyafxdBmm655ef3sSFSN1ITQk2CsauknaC1s870EOfOnNXJoVbXqMm6RlkqGdrn1dKes3KS2WvPNr7cn0IZHoTNXap4JY+ax9AFLLkzQF3j06T2syNWBbYBTBjkAcXxHNqgzHMVTqxl02EV2Pvj34SzYo8c3BUXIPHEN4YDxx3HOFsI+AYafmOBEG1OMZEff88p7UwUncKZ9UxNLVhx1Do+jUxV9wuzdqPH0Anc++i9TBjaA1drgHJSIz9lfZQm2HIJ54FnZjO0k6Z5MvcU2z7YozNCU4gaEoxdpfg49YxWgT17Vq+LORFTq4/X119trXPUpHaUpY+vD4nnEvlm6zd621ZVVRFzQtkJ4eppz1l7DdeymA82sTNePcP+9GUbWPnCIhY8NItZjy9gyesf8c78cHyKYth/so7etW6hNX2mlMvLc9HvyqwvZq/mxd0WUHb+KNuUGa5S9NvfEF1XX7dGmUB4oAtUJBL3o8FQDNjO/rgck16PJESbUXyY9Z9sJ0EbkLmGMGnJv5h1aw9sgaTv3mbZxl85lZ5LeaVmm8py8tJPEblxGSu/SwJs6XHrbP61ZBIhrtpALIHtn6zjsERiog4SjF2lxHPnyFSpaO/anp5B6un8CgsL2f+7fn+MIUOH4t/Vv85Rk1bWVtz/wEw6d+nCkKH6Q232/76fwsI6goxG6hkURHvX9mSqVCSeO6dMvgZFs/G5Z1gfmUIh9vgEhxF+2zjGRYQT2s0FsqPZ/OpL1Hklti6KflM6i5877ZX5dZ3YTPR5gBxi9jZ1KAawl9e+VzcbNokpmubaCydZb6ApSWvvwXPkYPz1SEK0Ked/5YPXtnCs+nVGbgSNf5RXXn6WWVNG0CX1Z9Yvf5H/e/ppFi18mkVPL+bF5ev4ObUzEVNm88zLy3hkfJCmtgzKL0Xz1WvvE3le7yjXhtTc6po674i3WLHStOXeXtqhp+U0Ze+Ktsxs6NCh1VU4lZWVlJeXY25u3ixTKdyoIkZEMOHOiSQnJfPJ+o+r1z/86CNX/bLwtItpfPj+B8rVV+3BWQ/h19mP7d//QOSvej2Bm8WKlW+xaOHTytWN5EHI4AC8/UPwLorhZEI8USfrqgES1zpjZcNYumibmr9cONN3ymzuGuyNrbLaorIcvd4klpZY1spTTOrBb1j3dXT1LP6t6h9PsSJCPSQ9NfJpVn4H0JkJz8wjQtuU2kB5sVtYrjc5biszdI22HenRWd1pLzcpgYziOtYZ2rYFKYuPaITIXyO5nJ2NX2c/BoYNrF7/7bZt5F7FSMXc3Fy+3db0NTEDwwbi19mPy9nZLRKINT0VMQf3s/OLtaz/dr8EYkKIZpDLsa/f5t8vf05kbAZ52mZJAHNLLK11Ft0naWUuGbG/8vnLz7HyWgnE6pTE9tVv89XOw5w6l0pqsmlLUuxhIre9f20FYnUpziAhLoGEOE3QVde6ViY1Y00ktF8oM+67F4B1H3xEamoqAO4e7kyaPLnBNWRpF9P4dts2MlVNO9LR29ub2Y/MAeCLzz4n+mjtMVXNofn/ixXXK2Nlw1i6aJtavlzY0jFkEAP79aR7e2fsXN1wtgZKr5CVXUzu5QROHz1MVEzGtRmgtHLNT4uY+BQrblFfY0bU+3x3TJmhHn0n8UiYeoaB1H1Ps/IHZYbmJcFYE7pr6hQGDxnM5ezLfLZxE9nZmlkEgZGjbiV8WLjR+1pVVcX+3/ezd88vyqSr5urqyn0z76e9a3sO/nGQb7Z+rczSbFr+D+cNaPxSNjwQopkJvG4p+6aycI1y7bXLWNkwli7aJikXDdQWgrHAe/nXo6FoXjHaSFkce/9VPo9Trm9eEow1sTmPPkxgYCCXsy/zzdavq2vIAOzt7QnpHULX7t3w9PSsmXQ1L4/09HTOnTlLzImYJu+sj6ZG7K6pU2jv2p64uDg+ev9DZZZmJX84m0DvCcwa3hVH5XoFVewqNjf2vZmtwFjZMJYu2iYpFw00YDavzAjC8PsDTJFK5MK39d+6cM2xZeBDi7k7pPHvW86L2cLyj1u++VWCsSZmaWnJg7MfIjAwEICf/vcjh6MOK7O1qIFhA7ntjtsBiIuL45N1Hxucy6w5yR9OURdjZcNYumibpFw01NV11r8+gjE1J78eeNop15qgKIOE5Mb3874aFn5+fv/RfqiqqqKyshIzMzMJxhqpsrKS6CNHcXJ2xtfXlx4BPejatSvFxcVcunRJmb1Z9QwKYsKdE+k/QD1t/cE/DvLZxk1UVur2RG0ZY8aNZfeu3crVQhgtG8bSRdsk5aKhrhAfHUNOuRXm5lCal0felYYsGZw5FMMF5W6vQaVXssnOasRypUS5qxYjNWPNKLRfKLfdfhvtXdUt2JezLxN3+jRJScmoMjLIy8ur932UDWFlZYWTkxMeHTvSubMfgT170t5VPXvW5exsfvrxpxbrrG+I/Bcr6mKsbBhLF22TlAtxI5FgrAVEjIhg0OBBuHu07FsAM1UqDh08dE1MXyF/OEVdjJUNY+mibZJyIW4kMs9YC4j8NZLlry3nvdXv8fPun4mLi+NydnaT9tsqLy/ncnY2cXFx/Lz7Z95b/R7LX1t+TQRiABXlFVhZ1X7zgGjbrKysqCivUK7WI2VHKJlSboS4nkjNmGgR8+Y/RuS+SGJPnlQmiTasV3AwEbdEsGb1e8qkalJ2hJIp5UaI64nUjIkWcfDAH4weM1pqOEQ1e3t7Ro8ZzcEDfyiT9GjLjr29sRnWRFtgZWVlUrkR4noioylFi0hPS8fd3Z2xt40jLy+fnMuXW2VUp2h9VlZWBN10E3dPuxuqqvju2/pnn0xPS2fQoEH0G9Bfyk4bpi0398yYztkzZ66ZLhhCNAVpphQtqm9oXwYPHUJnv85YWFook0UbUFFeQVJyEgcP/MGxaNPfVyJlp21rbLkR4nogwZgQQgghRCuSPmNCCCGEEK1IgjEhhBBCiFYkwZgQQgghRCuSYEwIIYQQohVJMCaEEEII0YokGBNCCCGEaEUSjAkhhBBCtCIJxoQQQgghWpEEY0IIIYQQrUiCMSGEEEKIViTBmBBCCCFEK5J3U4oWJS97Fo194bOUnbatseVGiOuBBGOixTy58EkwM+Pn3T8THxdHWVmZMotoA6ysrAgIDGT0mNFQVcU7K99RZqlFyo5oTLkR4nohzZSiRfQN7QtmZnz4/gfEnjwpD9M2rKysjNiTJ/nw/Q/AzExdNuohZUfQiHIjxPVEgjHRIgYPHcLPu3+msLBQmSTaqMLCQn7e/TODhw5RJumRsiN0mVpuhLieSDNlC7G2tadrcDg+3W7G3bsH7dy8sLF1xMy8aeLhqspKSorzuZKVRmZqAiln/+Lcyf2UFl8bD7Dlb77Bv//1nNRqCD1WVla88uoyFj/zrDKpmpQdoWRKuRHietI0kYCok5tXV0ZNe5a5y35k3L3/JnjQeDr6BmJr79xkgRiAmbk5tvbOdPQNJHjQeMbd+2/mLvuRUdOexc2rqzJ7i7OwtJCHqailrKzMaGd8KTtCyZRyI8T1pOmiAVHLsInzuP/ZTwkeNL5JAy9TmZmbEzxoPPc/+ynDJs5TJgshhBDiGiDNlM3AzasrY6YvoaNvoDKpVWVciGP35tfISjunTGp2K1a+xaKFTytXC2G0bBhLF22TlIum4ejoyNBh4XTt2hVrG2tlskG5V3KJi4vj0B8HqaysVCaLRmj56pobnHfX3kx9/L/XXCAG0NE3kKmP/xfvrr2VSUIIIdoYR0dHZj08m9FjRtOtezd8fX1NWnoF92LS5Ek88OA/MW+FVp+GaN++PYGBgXUu7du3V27SKiz8/Pz+o/1QVVVFZWUlZmZmUjPWCG5eXZn06Aps7Z2VSU0mv7CEyCNn+eHXWL7aeYyvdv7Ft7/E8MO+k/z65xlOJarILyyho5sT1laWys2xtLKhW0g4iacOUZR/WZncbMaMG8vuXbuVq4UwWjaMpYu2ScrF1QsbPIh+/fs1KqAyMzPDwcGBpMTzXL7ccs+Shho9dgyTp9xFv/79DC5VVVXEx8UpN2tx0kzZhKY/9VGz1oh9/fMJfoyMpbKq+iurk7mZGbdH9GLKaMO1YBkX4tj89hzl6mYjTQqiLsbKhrF00Ta1dLno3qMHAwb2x9/fH0cnJ6yta5r0SktLyc/LIzExkT8PH+FMQoLetteqCXdOJGJEBACRv0ay/fsflFkMenDWQ/QK7kVRURGfbdhE3DUQzGjPqanEnozlk/UfK1c3GwnGmsiwifPod8s9ytVNIlV1hY++Psj51GxlklFdvF2ZM2Uw3h7tlEkc3fclv/+wRrm6WTTlH86BAwcSERFBx44dASgqKiIqKopffvmFoqIiZXZGjBhBcHCwcjWpqamcOXOGmJgYvfV15dd14cIFvv/+ezp16sSYMWNwcHAgOjqagwcPKrNW5wHYvXs3Fy9erF7n6Oiol7e4uJiEhAROnjxJVlaWXpouHx8fxo0bR48ePbCwsKCiooKEhAR27txJSkqKMjt33nkn3t7e/Pnnn/z555/KZABmzJiBo6Mjv/76K2jug+4Dpy4nT56s3qYxjJUNY+mibWqJcmFubk7YoDBuHT2Kdu3amfRcrKqq4sqVK/zy8x6iDkVd032qJBirW0sHY9JM2QTcvLoy7r5/K1c3iVTVFVZ8spf0S3nKJJPk5BVx7FQKIQGdcHaw1Uvr5B9MwonIFmmubIomhYCAAJ5++mnGjx+Pj48PLi4uuLi40KFDB26++WZuueUWSkpKOHv2rN52t912G7fccgtdunTRW0JCQhg+fDjh4eFkZWVx8eLFevPrLmZmZkRGRtKlSxfuuusugoKC8PHxITY2ltzcXL3ja/P4+vpy8uRJVCqV3na6++3evTsDBw5k7NixuLu7ExsbS3l5efW+7OzsmD17NrNnzyYgIID27dvj4uJC+/bt6d69O7feeqvB7aZNm0a/fv3IyMggNja2er2uhx56iJ49e5KQkICTkxP/+Mc/6N69e61rVy75+fkcOXJEuTuTGSsbxtJF29Tc5aJTp07MfmQOA8MGYmdnV/1MLC0tJTMzk8uXL5Obm0tubi5FRUVYW1tjYWGBmZkZtra2BN0URHBIMEnnk8jLa9zf76Y0NHwok6fcxaDBg7C0sOBC8gUCewbSpUsXAJLOJ5ncXNc3tC8eHh6Ul5dz4viJev9xbCnePt44Ozljb28PQNalLLKysqq/I2NLeVm53rbnzp0jIT5ecZTmI8FYExh6xxw8fAKUq5vE2xt+bXQgplVcUs65lCxGDOiuTMLCwpJzsQeUq5vc1f7h9PHx4bHHHqNr165kZ2ezY8cOVq9ezYYNG0hLS6Ndu3Z4e3tz0003UVhYqBeQDRgwAH9/fw4ePMj333/P4cOHOXLkCDk5OdjZ2eHl5UW3bt2IiYkhNzfXYH7lcuzYMVQqFR07dmTgQPUfaycnJ2xtbWvVPGnzABw+fFhvu5KSEjZv3syBAwc4fPgwZ86coby8HFdXV7p37469vT3HjtW8FPnRRx8lIiKC8vJyoqKiWLNmDR988AHR0dFYWlrSqVMnunfvjpubG4cPH67eLiIiAnd3d06fPl1nMDZu3Ljq4508eZKkpCSOHDlSfc2+vr5YWVnxzTff8Ouvv+rdi6vpM2KsbBhLF21Tc5aLAQMHMuPeGbi5uWFmZkZFRQXnE8/z/XffsfnzzRw6eIiM9HQSzyby999/s/PHn9jz8x4uXryIs3M7nNs5Y25ujpOTE7379KGgoICLqanKw7Som/v2pXef3rRr1w6VSkV8XNwNFYwlxMfT3rV99fX8efhPNn66gaiDh0xalNvu+GG74gjNq+G99oQea1t7eg28Xbm6SXz984lGNU0CONrpNy+dT83m659P6K0D6DXwdqxt1f8NXMvuvfde/Pz8SEpK4pVXXmHr1q3VfwD279/P0qVL2bdvHzY2NkyaNImQkBDlLigqKiIyMpLIyEj27dvHhx9+yJtvvklKSgoeHh5ERKir6w3lVy7Kps2CggKKi4sZMGAAQ4cO1UurT2VlJRcuXKje77fffsurr77Kt99+S0VFBTfffDOdOnUCYNKkSQwePJji4mI2bdrEO++8Ux10nj17lvfee49PPvmE/Px8hgwZwt133604mumysrI4cOCA3jVXVFRQVVVFZmam3nplTaQQ17Nhw4czafI/cHB0oKqqioyMDDZ88ilr3n2P+Lh4pkydwiuvvsJj8x9nzqMP89j8x3nl1VeYMnUK8XHxrHn3PTZ88ikZGRlUVVXh4OjApMn/YNjw4cpDCVFNgrGr1DU4vFkmdM0vLOHHSMM1GMYEdnHj+7f+wZAQ9UNc68fIWPILS/TWmZmb0zU4XG/dtSY8PJygoCByc3P56quvDPaJAtiwYQMJCQm4uroyaNAgZbJBKSkpJCQkYGFhgYuLizLZZCUlJfz111/Y29szbtw47OzslFka5MyZM+Tl5WFnZ0enTp2ws7Nj8ODBWFhYsHfvXnbvNlwjsG/fPn777TfMzc3p37//VZ+HEG1JSO8QRo0ZhbWNDZWVlZw4foJVb7/D37F/4+joyOxH5hA2eBDWNjZ621nb2BA2eBCzH5mDo6Mjf8f+zaq33+HE8RNUVlZibWPDqDGjCOld+59E0byGRQxn+Yo3WL7iDYZFqANiQ+taW9NHEW2MT7eblauaxMHj500aNakU2NmV9xePwtnBhtuG6r8GqbKqioPHz+utoxmvoakEBgZiZ2dHcnKyXtObUlFREdHR0VRUVNC9e+0m2eZ24MABUlNT6datGxMmTFAmN4i2q0BlZSWlpaXcfPPNuLu7c+XKFaKjo5XZ9Rw8eJDs7Gzc3d25+eZr+7sV4lrh6ubK7XfcgYODukbs1N+n+HzTZ5SWlgIwasxo/P39Abh48SLvr3mfRQuf5v0171f3N/X392fUmNGg6Vv2+abPOPX3KXUNmYMDt99xB65urjpHFUJNgrGr5O7dQ7mqSZw8k65cZVR3n/asWTwKR3tr9v+Vyn8+qt0XzNB+m+samoqXlxdoRj8ac+HCBQoLC3FxcTHYVKlkZ2eHv78/FRUVtTrZ2tnZERERUWsZNGiQwRqnoqIi9uzZQ0VFBREREQQENL4fYXBwME5OTuTk5JCQkIC3tze2trbk5OTUaiJVio+PJzs7G1tbW/z8/JTJbVKvXr146aWXWLduHfPnz8fNza1B6T4+PixevJh169axZMmSWt+tsXSA+++/nzVr1vDWW28xcuRIZTLjx49n1apVvPfee/zjH/9QJhtNN3YNxtKNXYOxdEy4RmPprWnkrSPp4N4BNDXmW7d8VT0S0tPLk17BvTAzM+Niaiofrv2gevqKMwkJfLj2Ay6mpmJmZkav4F54enmCphvC1i01tfkd3Dsw8tZr67rFtUGCsavUzk0dKDS15IsN6yvW3ac9H/xrNM4ONuz/K5VF/91HRUXtmjVD+22ua2gq5ubmVFZWkp+fr0yqpaSkhIqKCuVqMBBcTZ8+nWXLltGlSxeysrI4dOiQXv7Bgwfz+OOP11oefPDBOmvefvzxR06ePEmHDh244447lMm12NjYMGrUKL39L1++nIkTJ1JaWsqePXsoKirCwkL9UuSCggLlLgzS/jcv1N/7/fffT1BQEO3atWPYsGHcf//9Jqej6bPYr18/2rVrR2hoKPfdd1+D0idNmsS4ceNwd3fHz8+PGTNm0Ldv3+r0oUOHctddd9GpUyc8PDy46667GDt2rMnpxq7BWDomXIOxdGPXaCy9NXn7+BB0002YmZlRkF/A7p279f7edOzYEXt7eyoqKog5EVPrb1F+fj4xJ2KoqKjA3t6+etodbdrunbspyC/AzMyMoJtuwtvHR297ISQYu0o2tvrzRDWVK3nFep97de3AuufG4uKk31cBoEundiYFYhjYL814DdcaZXA1efJkfHx8UKlUbNmyhXjFMOb4+PhaHfcjIyM5ePBgvaOHdu/ezZUrV7j55pur5xeri4ODA0OGDNELErt06cLFixfZsGFDnX3DhOm6d++u98oTMzMzOnfubHI6gKenp94I844dO+rVvJqSrjtnm4ODg17Nko+PD7a2NVPP2Nra0rVrTTcDY+nGrsFYOiZeg7H0+q7RWHpr6t6je/WcfxmqDOJOn9ZLd27XDktL9RtNSur4R0e73tLSEud2+vM6xp0+TYYqAzSvIOrew/A/c6LtkmDsOjF9bBB9Ajx4f8kYvYDM36sdHy0Zg7ODDZHRF+oNxK5XJSUlWFhY0E7xB84QR0dHrK2tqaiooLCwUC9NGVzt2rWLt99+m8cff5zffvtNLy+apop333231vLpp59W9xEx5NixYxw4cAAbGxtGjhxZ76tGcnJy+OSTT3j33Xf56aefKCsr49KlS7z//vvs3bu3Ol9xcTGVlZUm3QM0D7rKykqKi2sH323NmTNn9KbeqKqq0vv+jKUDpKenU6XTh/Py5ct6zcWmpOvWVhYVFZGYmFj9OSUlRe+7Kikp0RuoYizd2DUYS8fEazCWXt81GktvTTEnTpCdpW418PX1ZcjQIXrplzIvUVZWpp6uQjFRs5aToyPm5ubq3+HMS3ppQ4YOwdfXF4DsrGxiTtQe2S7atrqfEsIkJcXGm84ao52T/gStL354gD9iLtLN24V1z43F1dkO345OfPivMbg427L/r1QWvxtpNBBT7pdmvIamkpKSQmVlJd27dzfYV0tXjx49sLOz49KlS7WmXFAGV+vWravVNNlUvvrqK86fP0+XLl3qHdmpO7XF5s2bOX36NB06dGDcuHF6+ZKTkykqKsLNzY3w8PpHv/bv3x83NzeKiopITk4GzXHQNPm2NUVFRWzatIn4+Hjy8/OJjo5m8+bNJqcDfP7550RHR5Ofn098fDybNm1qUPq3337Lzp07uXz5MiqVim3btukNRjlw4ADffPMNKpWKy5cvs2vXLrZvr5nnyFi6sWswlo4J12As3dg1GktvTdlZ2Rw9epTy8nKsrKwYHjFcrykxOSmJy5cvY2ZmRmi/fnTuol+r2LlLZ0L79cPMzIzLly+TnJRUnebt48PwiOFYWVlRXl7O0aNHqwM/IbTa3l/mJnYlK025qkn4ddIfcVNWUcmilfs4dDKNzl7t+Oi5MXz03FhcnG0bVCOm3C/NeA1N5c8//yQ7Oxtvb+9aQYouHx8f+vfvT1VVVZ0Tm7YUbWf+kpISBgwYoEw2qKioiF9++YXCwkJCQ0O55ZZbqtOOHTvGmTNnsLe3NzqP2bBhw3B2diY5Obl6wliVSoWFhQWenuqOxUohISHY29tTWlpqcr+060lsbCzPPfccDz74IK+//nqt6VGMpaekpPD666/z4IMP8txzz9UqX8bSATZt2sTDDz/MY489xo4dO5TJ7Nixg8cee4yHH364VqBjSrqxazCWbuwajKVjwjUaS29Ne/f8QuK5cwC4urkxZeqU6pGPBQUFHPnzCGVlZbi0d2H2w3OYcvcUQnqHMOXuKcx+eA4u7V0oKyvjyJ9Hqn+HXN1cNftRD5a4kHyBvXt+0TmqEGoSjF2lzNTmeSFscPfaD82yikoWvbOPvxJU+Hk649bOzuQaMS1D+22ua2gq8fHxHD16FEtLS+644w6D/bB8fHx4+OGH8fT0JCkpiR9+MO0da81pz549/PXXXzg7O+PqWjsINuTAgQP8+eef2NnZMXbsWL2awN9//52CggJuvvlm5s2bZ7CWcNasWQwYMICCggJ++aXmj/7ff/9NYWFhnf3YIiIicHFx4dKlS1f1aiMhrleVlZXs2P4/sjX9QX39fJk1Z3Z1LVjkvl+JOhRFRUUFdnZ2DBo8mAce/CeDBg/Gzs6OiooKog5FEblP/Z7Wzl06M2vObHz9tM2TWXz37XfX9LsqReuRYOwqpZz9S7mqSQzu0wVzA6+kKimrYP4bv/BXgspoZ30lczMzBvdRv+5BV3NdQ1P64osvOHbsGE5OTjz00EMsX76cuXPnMmfOHP71r3+xbNkygoKCSE9PZ8uWLRQZeGF4Q/Xu3ZsVK1YYXB5++GFldoP+97//kZmZqVxdrx9++IHU1FQ6d+7MpEmTqtcfOHCAH3/8kbKyMkaMGMHbb7/NwoUL+ec//8nChQtZs2YNY8eOpby8nB9//JEDB2qmNtm/fz9///03dnZ23HffffzrX/9izpw5PPbYY6xcuZLhw4dTXFys109NiLYmNSWFr7d+TU5ODmgGKcx+eA5jbxuHubk53237lq1bviI1JaV61HZFRQWpmqkwvtv2Lebm5oy9bRyzH55TPaoyJyeHr7d+TWodE1YLIcHYVTp3cj9VzfCfjqO9DbdHGH4DfXFpOQtW/NKgQAzg9oheONrrj8asqqzk3Mn9euuuRUVFRbz++uvs2LGD/Px8/P39GTlyJGPGjKFv375YWFhw9OhR3njjDb13OV6NDh060LlzZ4NLXc19SvHx8fz22296L+02JiUlhd9//53KykqGDRumN2Jt69atrFu3jpSUFFxdXRkyZAh33HEHQ4YMwc3NjZSUFNatW8fWrVv19gmwevVq9u/fj7m5OX379mXMmDGMGDECb29vcnJy+Pzzz2X0pmjz4uPi+WT9J6SlpVFVVYWdnR2jx4zm+ReeZ9LkSaSlpbPyrZUsXvQsixY+zeJFz7LyrZWkpaUzafIknn/heUaPGY2dnR1VVVWkpaXxyfpPiI9ruZdOmyonJ6c6qIwYEcGKlW+ZtPQKVj+bKisrKSsrU+xVNIbZ0KFDq5/mlZWVlJeXY25urjeEWdRv1LRnCR40Xrm6Sfxnzc5Gv59SVxdvV/4zr3Z/q5OHdrBnyxvK1U1uxcq3WLTwaeXqRgsJCaF79+60a9eOhIQETp8+Xe90EzciHx8fgoKC6NKlC0lJSZw9e7bWoAVD3Nzc6NmzJz169ODKlSucOXPG6ESyzclY2TCWLtqm5i4X1tbWTLxzIv0G9MfKykovrby8XC8IsbKyqp76QqusrIyjfx7hh+9/uCbm/Ztw50QiRqjfvxv5ayTbv/8BR0dHZj08u3qkZ0NUVVXxd+zfbPjk02um6bWua/TUTByenpZGfn6+wXWGtm1JFn5+fv/RfqiqqqKyshIzzatYhGmuZKfRZ2jtGbGbQo/O7hw7lUJxiek1K0rtne2Yf+9wnB1qj6Tc9cWrFOXXDHlvLmPGjWX3rqardVGpVJw+fZq//vqLCxcuNEmz5PUmNzeXc+fOER0dzdmzZ/WmLqhPUVERFy5c4K+//uL06dOoVCpllhZlrGwYSxdtU3OXi4qKCv7++29Oxf6Nc7t2uLRrh4Um4DI3N8fKyqp60R2lXFpSwunTcXzx2edERan7mF0LAnsG0qWLuptK0vkk4uPiKC0tJTbmJGXl5VClHqiQm5trdEm5kML+/fv53/Yd10wgRj3XmJ2dTXZ2dnVQbGidoW1bktSMNZFhE+fR75Z7lKubRKrqCh99fbBRNWRdvF2ZM2Uw3h6156c6uu9Lfv9hjXJ1s2ju/2LF9ctY2TCWLtqmli4X5ubmBIcEE9I7BHd3D9q5qCeCLS8v50rOFTIzVcSciOFkzMlrKkDRau2an5Zwx4TxjLhlBGZmZvx5+DB/RZveH/rm0JsZMHAgVVVV/LrvV/63vWVH+0ow1oSmP/URHX0DlaubzNc/n+DHyFiTXiBubmbG7RG9mDK6tzIJgIwLcWx+e45ydbNp6T+c4vphrGwYSxdtk5SLhmkLwdiAgQOZdNckvTc9NFRpaSnffvMtf7bwHHgSjDUhN6+uTH38v9jaOyuTmkx+YQkHj5/n5Jl0ki9mcyWvmMqqKszNzGjnZItfJ1eCu3syuE+XWp31tYoLc9n67hNkpann1GkJ8odT1MVY2TCWLtomKRcNMyxiOOMnjK9+z21DFRUV8dmGTcS1cPNdQ5ibmzPznzO5qVevRk1wXVlZyd+xsWz8dGOL125KMNbEvLv2ZsKsV5s1ILsaxYW5bF//L1LPtezrOOQPp6iLsbJhLF20TVIuGuZqOutznQRjWr5+vtjb2StXG1VYVMiF5AvK1S1COvA3sbzLGSSeOoSnXxCO7Took1tVxoU4tn/8HBkX9F+C2xKau7OtuH4ZKxvG0kXbJOWiYRrbWV+7XM6+zN9/x5J7JVe562tO7pVcsrKyGry05rVJzVgzas5O/Q3Vkp31DZH/YkVdjJUNY+mibZJyIW4kDW9UFSb7/Yc1bHrjn5w8tKNZJoY1pqqykpOHdrDpjX+2aiAmhBBCiLpJMNbMstLOsWfLG6x97nZ2fv4KJw/tIONCHMWFuU0aoFVVVlJcmEvGhThOHtrBzs9fYe1zt7Nnyxst2lG/LhXlFbUmThTCysqKivL652GSsiOUTCk3QlxPpJlStIh58x8jcl8ksSdPKpNEG9YrOJiIWyJYs/o9ZVI1KTtCyZRyI8T1RGrGRIs4eOAPRo8Zjb19w0e4iBuTvb09o8eM5uCBP5RJeqTsCF2mlhshricSjIkWcSz6GFRV8fCjj9ArOFiandowKysregUH8/Cjj0BVlbps1EPKjqAR5UaI64k0U4oW1Te0L4OHDqGzX2csLBs3+aC4vlWUV5CUnMTBA3806IEqZadta2y5EeJ6IMGYEEIIIUQrkmZKIYQQQohWJMGY0DO8sJDhhYXK1UIIIYRoJhKMCQAsgddUKp7MyuLJrCxeUamkcAghhBAtQJ63AoD+RUUElpRUf76ppIR+RUV6eYQQQgjR9KQDvwBgSm4uM65c0Vt32M6OUzY2euuaS6aFBX/IPFJCCCHaIAnGBAD3XrnCXbmt98Z6gD/s7FjRoYNytRBCCHFDk2ZKAYBvWZlyVYsbUlREj9JS5WohhBDihibBmADgWpnTXOpjhRBCtDUSjAkAcixaf0bzKDs74q2tlauFEEKIG5r0GRMAPJ2VxVDF/GJ77O252ELvAbxkYcF+BwflaiGEEOKGJ8GYAGBFRgZdFf215nXqRPo1UGMmhBBC3MikmVJgV1lZKxADyJWAXAghhGh2EowJ/MrLlas4b21NaFER4QUFuFZWKpOFEEII0UQkGBP4GKgV8ykr46nsbJ7KzubdtDTuyM9XZhFCCCFEE5BgTNDZQM2YZVV1V0JsKyuZdfkyb2ZkECavSBJCCCGalARjAn8DNWOGdCstZfGlSzydlUV3E7cRQgghRP0kGGvjbKqq6KXzgnBTDC0s5I2MDGZeuUJ76U8mhBBCXBUJxto414oK5apqJ2xtlav0/CM3l9Wa/mSWykQhhBBCmESCsTbOp553Uh6zteWpjh2JtLdXJlWz1/Qne//iRQnKhBBCiEaQYKyNyzUyqet5a2tWubnxaocOxNnYKJOruVZUMOvyZdZcvMhteXmY6wwAEEIIIUTdLPz8/P6j/VBVVUVlZSVmZmYyA38bkWVhQbvKSoMd8o/b2lYHYBetrPjFwYEcCwsCysqwqSPYsq+qol9xMSMKC6k0M+OstTWGcwohhBACCcYEwFE7O5wqK+mhCMh0gzGts9bW7HFwoMLMjJvq6fjvUFlJqCYoKzcz44y8AFwIIYQwSJopBWhe1G2qfHNzvmjXjse9vNju5KRM1uNRXs7Dly/zdnp6rReRCyGEEEKCMXEVLlpa8omLC/M9PdlhJCjrUlbG01lZPJ+ZSUhxsTJZCCGEaLMkGBNXLdXKio9dXHjC05Mdjo7KZD19i4t5MTOTx7Oz6VLPSE4hhBCirZBgTDSZFCsrPm7fnie9vPifkaBsZEEBb6enc39OjryIXAghRJsmwZhocsmWlqxv356Fnp7sdXBQJuuZlJfHmosXuTMvTwqjEEKINkmef6LZJFlZ8a6rKy+4u3Osntn8rauqeCAnh9VpaYwoKFAmCyGEEDc0CcZEs4uxteVld3fecnPjvJWVMrmaV3k5T2Rn85JKRX/p5C+EEKKNkGBMtJgD9vY85enJRy4u5JnXXfSCS0r4V2YmT2Vl1Zr7TAghhLjR1P1EFKKZ/OTkxGNeXnzVrp0ySU94YSHLMzJ4KCcHj/JyZbIQQghxQ5BgTACQYmCG/KR6mhSvVr65OV86O/O4pye7jIy8HJ+Xx/tpaUzOza3zNUxCCCHE9Upeh9TWDXuFaXc9iE2nIopOxuGrqYE6YG/Pt87OelndJ65lwu334GH1LeeT9ZIaLc/CgqN2dpywtcWpqgrveuYe611SwoiCAorNzTlnIHgUQgghrkcSjDWjfvP38I+77sAp82vOZyhTm1I/+jz0BiNGTMbfs5T40wnKDHULfZDRfTpjUxjLu6fPccrWll8dHPjOwIz6HYc9Rv8eHlSpPiH2tDL16lyytGS/vT2JVlZ4VFTgVlGhzAKaF5EPKCri5uJi8szNSW3G2jshhBCiJUgzZTOysLXGxtoJ2+auxPGcRJ+Qznj5daZ72Hi6K9Mb4ISNDScULwdvSYft7fm/jh1539WV7HrelxlYWsr/ZWWx+NIlgup5Yfn1rt/8PcxdvoVb+ihThBBC3CgkGLsBeI3rhav2g20Peg7VT78e7XZw4DEvLz430sk/rKiIZSoVj1y+TKd6mjivVy0W0AshhGg10kzZjLyHPoivUylZMZ9z5qIytancweA7R+Nhm0VsVCpuPh60t7fjcNSfyoyAF67DHqNv/1vxDeqLZVYSOb6jGdTFhbLMP4mOPqmTtwe+Yx8nuE8Evj16UJwRjfVN99LTw5r8pLqbKW38RuHq2ZGKrApsw2YzIGwMvkGDsC8/R1Z2vjI7Fh53EHDrdAJ7RdScU4E6X0X7IVwK6MOhju4UX8kkoJ5pLrqXlnJ7fj4Obp04V15CiaHya9sP/1vncFOfCPU5oSLrUrYyV73npMflDgLH1J1P916009x3e/NIsjK12w+me8RD9DR0Prb9cO3WC69eEXRyKCUvKYNs665YVZyjRKZgE0KIG4rZ0KFDq4enVVZWUl5ejrm5uQRjTWDgM78xpFM+cV/czk+GYqOmMGAlc2b0wyH7IJ+/mc3I1+7AqzyBfc/P4rjuQ9v2Hm55ZjZ9XHWrWEpJu5iHVyc3CmLX8NG6L9WruzzJXY9Mxld30vzKLNJUNnh5OpIWOZwt3+mk6VBfcxpxMdYEhrjppJSSFvk8W747qPnshevkN7h3WGf0GyNLyTy4hi+/2kaF5yvMXDwc19JYflo8l8rSUibkeRBeGKO3hVKuhQVfOTvzo+4ozS6LmTb3DrwUNUxXYtaw8eMvUfdQM+GctPnGv8K0W3ug36BbSvaxdXy+Ub2/6nsR60RgL/W5aO+dzdCVzJjcj3aKuukrcV+y7f01XNF+r/rJ9d57IYQQ1yepGWsKLoNx79wThw5d9RZPRa2GXrpre0qy07jaiRq6T3yG4A4WXDn9CQeO7sGy51T8Xd2wrNjJqTPaWhov+jz2CoO8rKE4idg9X/Nz9HksnLrQ1ccFQKdm7A5umT+HACeoyI7lyK4vOHQ6H+uOAXR2twOot2ZMXRvoRIeOlaRFfcnuvccpat+NTu3scPLsxKVf/kc2YBH2MvfeEYRNeRqxP6zn5193k5DliGcXXzp07oFjyhbOJZrjOjACT0d7KrM/JyrDgoMRr9H9H2NxyM7ANjNVeXgAbKqqCC0uZkBxMTnm5qRa+TFwzr+4yQWy/1zH1o2biE4sxMU/CE+/IFyy1TWXJp1TpibfxGDsyCft+A4ivztAqpUbnTzdcPKq2V/1vfAoJe34b5xKSCEtZS8Z1ku5Z2Y4rhb5XNi3kR27dhBzrgz3bl3p4BlMB7OdnIq3xKzqCpXtutLetpTM4/uITThHYrxOzZoQQogbggRjTWHcazw85U5694/QWzo5AFjTPkB/fe/+EfQOcid93y5ylPtqkJkMuWsArpZpxG58g+T8fC61D6d/dzdc7O35848D6mDP8ylGje+BQ3kah9fOIPLocQpTDnHu0DnMbx6Nt6NOMDb0KW7r54Fl/lG2vz6fmLOnyE2KJP5wKe0HDaCDtSnBGGQffJ4vtvxAbno0yYfs6DiyD+1tKsg7/jXJ+V4MmPEE3Zzgws+z2fHLnxRmnSP3zC7OtB9HP18PHGyKiI7+AbOAqfT0cMSq/AJ/HT+H+6hH6RnamXMDZlDlbYf5X4ewq6xUngYA7SsqCC8qwqvcBrPQwTh7wLmfVxN7Lp6S9EOc+fssZ2J2cTImjSpMPaes6nyZvz/P5s1fk511nIy/viXVbRw92kFhThpx8Un69+Lzz7lwai8ZF8F/ymJCPa25cnwlX2z5msKscxSmRBJbPpC+PT1wtbfnz90fkHoqknYDH8TXqYiUvQ+z7xcJxIQQ4kYkHfibQlYSacm1l2xNM2GBqnZa2oV0DPRCapixEfjbApcSiE5Xr6r4NYaL5UCnUMI8NfkCvXAGuBRL1Hmd7TnIH3FJuitw8HbFBii4EEWibjNn8ZecTjL1jPPJTNQ2R6o/q7t7OeHoCxCBmzNAKQ5BbzBt4abqZXw3JyoAh/adAUhMSKIEaNexHxb0o6evG+TH8VtMGum3PMJPMz3ZGHQ/lZZ193AfXniOGf8ZS+etnxN832fMWbKJ8f9cTPeOBWQnHNU0PZp6Ttp8aST+qHuNkPbFNNY+P4nvd/yms1Z5L7zwclM3WVp63aN3nGmhmi/M2RM/nS2EEELc2CQYawq//5stK++vtZzOBsgnZU/ttC0fLufqKjm86NOzh7pvU4fhzFn5G0+u/I0nX5uMr6U6vfvwfuqs7R3VfZsqSzSBR908nDXzixmubGoinXF2BLCmnacXrrqLizXlpaWUFGumq4g8ysVSwCOAYNtRdOoAJWmxxB0/zxXAvfscfr59LofX/k7CwADFcfR13rKSvgtG4xd/nO597uC2h1Yy/4WV6oDW5HPS5iulvFEd6SNop+nKZuuiOI6nE5SWUlJcirz8SQgh2g4Jxq5XttPo7qP+saK0lBK9Rb3eNfBu3AHSLlEAYO1Uq0O4hbl+jVJydh4AltYGXlHUZKUlidx8gDSOvjCKtYsNLKvf0eRdx/mLgKUXne4LxgtQnV0HccdIzwcbr14M7+xFaXsLdjoX8VTHjkTa2+sfTodjRhK93lmA3bwJXI4+Ai79GHH3qAackzafNTbamkctl3446A56MOhLsnIB8jmzzcAxFo9i7bLFXFBuJoQQ4obVZI9X0bIsxvVT14Cp9vCx8mG++EsulAOuAdzcC4hKILsc6BDK8DCvmp3Y3sOYPjqfgYpYdY2TjX8EA7voJHRZzMAeBgK0RvmSi5dQ195NHKyX4jVzC3Of38Qtw2rWnUxKAhzpHtgZSOJCJMCXnEsrBcd+dPcA8s9zPg7OW1uzys2N1zt0qPeVSQNUJ7nz1an4f/4mLnZ+DTinmnw9x9+jM+rSi8AZy5nz2m9M+0cvna1rS8zIAhzp0m+23qhNi3HrmfviFsZPHKWzFsAR++qJ5IQQQtxoJBi7ToUFqftUZSb8T13rpWcNJxNLATd8BtwBrCH6VD7gSODd65n56HKG/HMtM5+bR6D+6ych7itOJJeCZWeGzP2WabNfYdjsTTw49w68LBV5r8LxPQe5ArgOXs6cxSsZNnkpo+d/y5S+Xtg4Q67OG50qDiWpR2BaApeSiNU0D8adTaJCs74kLZbEmk04bGfHoo4d2eCiHi1aF99v19D/1dcZmZ9v8jlp8zn0msdDS9YyevJyxj+3ntt6WEN5Aqd3xioPoyfzq5+5UA42PWYy74X1jJ68lGGzt/Dw2B7YODtRkranJu9ldT8935Hfcu/ib7lzos6OhBBC3BAkGLtOdfdA3Yl8/1FlEgBxsQnqju+B6tcjJX78Mj/HZYG5I66BgxnYpxeu1mkcPZ6m2PIoR9eu4Wh6KVi74dVrOP16daZdYSxHz5nagd8EsYv54uuDZJeCg2c/+g0bRa+ubliUpnH862c5qhmQAED6r1zQzIV65cKvNcFnZDwqzY+qs+u0a/V87+TEvE6d2O3aV5lUza6whMcvX2bprw9xbO0nxs9J99w9etFr2GC6d3CE4iSObvy3/vxuhhSv4bsPt5GYDxYuPeg1bBT9enlhU5lP4q6X+VlnTrrEzzYSmwNYu+Hu6Yat/MYKIcQNRyZ9bUbutyylR/tc0v54h0Td4KIVWXgMx7W9NVBAbtxB6nuro43fKJztgMrLOqMOm5oXDj16YW/e3MeBm4uLmVpsRlCeNoQzbIeTJ7v6RFBqa2/knK7+3KvvsZHvw8JjOK62aWQmN+Al8EIIIa4LEoyJNue2/Hzuv3IF2zrmJwMoNjfns3bt9GfxF0IIIZqBTPoq2pwz1tbsc3TEHOp836WlZhb/3iUlZFtYkG7ZhB3mhBBCCB0SjIk2qcjMjGO2tsTY2uJaUYFnueGZvdwrKogoLMS1ooIUKyvyzaXTlhBCiKYlwZho0zItLYl0cCDTwoIeZWXYVRl+W2i3sjLuyM+nxNyc0zb6rwcXQgghroYEY0IAidbW/OLoSBVwU0ld3eihT3ExYcXFXDY3J9XKSpkshBBCNJgEY0JolJmZEWNry1E7O5wqK/Gpo+nSpfoF5OWkWVlxxUJ36lYhhBCiYSQYE0Ih28KCA/b2pFpZ0aWsDKc6Rl12LitjXH4+VWZmxFlbUym/M0IIIRpBgjEh6pBsZcVuJyfKzcwIqafpMrikhPDCQnLNzUmu5xVMQgghhCESjAlRj0rgbxsbDtrbY1dZSZeyMmUWAJwqKxlcVESXsjIuWlqSI02XQgghTCTBmBAmuGJhQZS9PYnW1nQqL6d9heG59n3KyxlbUECJmZmMuhRCCGESCcaEaIBUKyt2OzpSbG7OzcV1v4SyT0kJISUlpFtakikTxgohhKiHBGNCNEKcjQ2Rjo5YVVbSrY6mS/eKCkYWFGAOnLa1xfAwACGEEG2dBGNCNFK+uTlH7OyIs7HBvaIC9zqaLnuVlBBWVMQlCwsuytxkQgghFCQYE+IqpVtastfBgRIzM/rUMeqyXWUlwwoLcais5IyNDaXy+yWEEEJDgjEhmshpGxv+tLPDtbKSTnVMGBtQWkpEYSEqS0tSpJZMCCGEBGNCNK3LFhb8bm/PFQsL+paUYOi3yK6qiqGFhdhUVRErfcmEEKLNk2BMiGZwxtqa3+3tcaxnbrKepaX0Ky4m1cpKRlwKIUQbJsGYEM0k39ycKHt70q2s6Flaim1VlTIL7TUjLmVeMiGEaLskGBOimSVZWRFpb49dVRXdSkuVyaCZl6xbWRlJ1tZcMTdXJgshhLiBSTAmRAso1kyDkWFpSb/iYgyFW53KyxmTn88lCwsS5R2XQgjRZkgwJkQLOm9tzQE7OzpUVuJtoC+ZGTCwuJgOlZXE2dhQIr+HQghxw5NgTIgWlmdhwX57+3pfqdS1tJRBRUVkWFlxUTr3CyHEDU2CMSFaSZyNDSdsbfEpK6ODgdn7HTUTxQLE2toqk4UQQtwgJBgTohVdsrRkr6MjVlVVBNXRuT+4pITA0lKSrKzIsbBQJgshhLjOSTAmxDXghK0tidbW3FRSgp2BKTA8y8sZW1BAlqUl56RzvxBC3FAkGBPiGpFqZcXv9va0q2ei2AFFRbhVVHBa3m8phBA3DAnGhLiGFJubE2VnR66FBf3q6txfVkbXsjLOWluTK82WQghx3TM03ZEQopX95OjI4o4d+buOWfn7FBezKCsLFwMd/4UQQlxfJBgT4hqVYG3Nvz08+MbZWZkEgF9ZGQ/m5ChXCyGEuM5IMCbENe7zdu143c2NbANNksMKCw2+81IIIcT1Q4KxtqDLk9y2cBPjJ0+m9uO8fhZhr3DnwvWMvmWwMunaMewVpi3cxPiJo5QpN4zD9va86O5OmoEJYKUjvxBCXN/Mhg4dWv1vdWVlJeXl5Zibm0sH/qth2w//MfcxsE8gro7WQCkFKXGcPvwZR6OO0tK9fPxn/8idvRyBJH5feD9HlRnqMfCZ3xjSCcg9yjcvLOSCMoPSgKXcGd4DWy7y19rFxBnog959xib6dVSsLLrIhcRjHI/8kgKdbdR58znz41yOxuluoOMfm3gyojMFsWv4aN2XytQbSmhxMU9kZ+Os6Sv2lbMzX7Zrp8wmhBDiOiI1Y02ty5Pc9cJK7rylH16ujthYW2Nj7Yhr134MuWclDy94kpZ+dCbHxJFdWsqVhIOcViYaEZeQREFpPtlxUcYDMSBwwHD8/Trj5TeY/uP7KZMBcPXujJefYgkczMBx85jz4haGhXjVyuthuNtUmxNta8uCjh1Z5u7OMnd3CcSEEOIGIMFYU7K9h9senIyvLVRkx/LHlwtZvXA477y4mB37ErhSCTZdJjN55h3KLZtVRdRCNi4exSdr1lCgTDTiynf389Hi29n4hQk1TrZPEuxfMyGpe487cNDLoCufxB0v8fn7L/H5++/w077fOHOpFKy96Df5SWrCMaF0xcKCo7a2HJVXJAkhxA1B5hlrQl4zXiaiiz1c+o0NrzzNudQ0qgCKU8iO+4GYkoHc3NMDBwc7Mn7dhe44OAuPOwi4dTqBvSLwDeqLZVYSOQX5NRls++HarRd2FkUUWUUQOEad19X2Ihnp2eqm0VvncFOfCHz9vMhOPUlZuf62jg4WFF7JBrxw6DEIF7f2lGQ70iFiDjf3uxXfoL6Qdohc3aZFl8G4d+6JrUURRbrnY4DFqDmMDHCj6vxvnKIzHm7tsDy/hfNZ+vm8hz6Ir1MpqiMvc/zEOQqzTpEVt5f436HDiFBcHdtjlv455zJq8mbFfM6Zi/r7qdZzMoO6uFCW+SfR0Sc1K3vQLjAU5w5dsao4R0kx6usOuYfe4ePxD1LeJ809cdfNX8PGbxSunrppPXCvvm8RuDrnc0n7fQshhBANIH3Gmsw93PbyPAIdSznz9Sh2HFCmow4Q/OBKcoLOOi9cJ7/BvcM6KzrXl5J5cA1ffrVN3cdswErmzOgH546S69cPr+p+3KWkHdxBQa/JdNdtysv+jS0v/5s0nW0dLv6Pd95cXnOuJBCX3ZlAP53X61RmEfvVPH6OSlN/Nrk/lhcDn9nCkE5wYedwon3V/dSuHF/OJ5/+Ty+nuh9aPnFf3M5Pf+ol1UpTfjao1jkOps+C57mliyMV2QfZ8eZiEhnFkLn/x0DdawUoTePoZ0/ye0wa3R/dw/hAa0ri1rH2/Y06meZx15v34GuZRey6Sfx89h5GL55HLxedLEBF8v/YuHI5V/RXN5xtP1w7tzc62KI0Yw9XZGYLIYS47kkzZVOxDcDVEShNIMFgIAaQoAjEwCLsWe4a1hmL8jRiv3uHz99/iW92HSW73Br3wfczspdedhy69sP14h52vP8O+05lUYE1XoMn091Wvf033x0ksxxwHc6Qsfrb1uLYg8BOeZzZt47Pv/ofZ7JLwdyNXiPn1NO8WAfPOfToBJQncGYXJB6M4QrQrvNw05scbZ/E1wMgj3xTOqgZ5IX/bHUgRs5RdSBW7IX/zKcY6GdNheogP21UN43ui82iQqdZ9MzvR7kC2HSLoI9OC6DFuH50sgRUx/gjFvzvm0kvFyg4t41vVi7kozXrOKoqxcLvDsb9w+SrrVvxRdoNfYx7H11a53LP7f3Qq1oVQghx3ZJgrKmEtMceoLSYUt31mmkX9Je19AsE8KLf8H44ABd+eZKfI7eRGbeHCzsX8s2faYAbXQbfo7s3yDnIdytf4kzcNo5/+D+SSwFKSdyp3v5C5GL+SFA3J1ra6W9aWymJO+ax44eNZB5czo6PDpEN4OKFjzKrEQ7hvXAHShIPchwg9jdScgCXEG4eoMwN4IjPKP378uCLk/G1hIqL0USnK/Obxn/2evXI0dyjfL98IYnFgO1MBgY5QnkCv61cTNyxPWTGbeP4ulf465LOOcZ+RZwKsOxBz3HaoKofYX17YAFcOPYRBYCzgyMAuYn/40LyUQoSNvL72uf5/P3FfPedpkbxqqSR+PE8vo9RtO9qNFkNnBBCiGuCBGNNJbcUbRctPW4GRg5Wjw6MwM0ZoBSHoDf0ApPx3ZyoABzad9bfX2G2uukRgHxKSwHSSNunEwRU1vxYvzTSInW2S8+jBMDasYEjPkcxMMgLKOVi7DrNuv9xMikLcKRLv5mK/GoOHvr3pJ01VFw6yE8fLW/wQAMA224zGd/LEcji+GeaQAygjyfOAOWu9JyrGwA+ibrV0hFnb4CjRB1LoALwCtLUDnpOorsHUBzL8Z3qe3XytCbPreuZ+/wm7pz9Cn2CIDfuoPr+NQnDAZkEYkIIceORYKypxCWRWw44dsLHU2f9r2s0IwZf4vP3NxKXq5NGZ5wdAaxp5+mFq+7iYk15aSklxU33eG82vUbTxRXAGv9//MaTK9XLtD5uANj4DyZQuQ2lZB7fw+HfNcvOdXy+charly3mTCOb3yxsHbGoBHAjePLSmoDSq4M6sLJ20r/Hnl4425ZSUlpKsSaSrtgZqQ7iPPoypBd4jeuFK3AlbgdnNLur2DWLjV//RmJ2PpYunfHvNZxb7l7O3OWbGNhFe9CmoB+QSSAmhBA3JgnGmsy3XFABeNF9jM7UFTkHyYzbo14se+CpN19WErn5AGkcfWEUaxcbWFa/o7vBNclrQIA68ClXBza6S0UlYN2L4Fr910rJjn2JP7Zpll0byVT0p2uw3Fh+fn8Ncflg4TmccRM185ylXlLXtKl+40Pl/dUs3+/Q7mQjsWfzATd8BiymX6AbkEbcTv1BCFcO/JvvX76d1U8PZ/VrazicnA/WnQm7c55evqunCch2fCmBmBBC3KAkGGsyaRw9HEsJ0K7vY9ypfH2Qy0xuu3uwovnvSy5eQh3ATdTP7zVzC3Of38Qtw/RWX4Pu4eZubkApZ76rHeR8eVxdq9Op55NGRwderYILkcQmfMnu/QnqgQ3D5qk74v+ZQHY54NmLIXo1V3dwy3M/8uDiV+ius7Z68EHIHXS3hYrzB4mq7sM2in7zvmXO8k3V+6pQfckf++MoACycFc3KTSKNxF/WSCAmhBA3KAnGmlBF5EvsjM0HHPGfuJy5z2v6Ji3+lrnPzybQOavWVATH9xzkCuA6eDlzFq9k2OSljJ7/LVP6emHjDLlXWVnU7IZG0MURKE7gtIFRpJm7Y8kGLHxCCG6hOUordq3hj+RSsOzBkIdmY8Ea/ojJUg+YmP8j02a/wpDJy7nzhYX06eCIY2meTj88nY785qgHORx5R+cVVnvILrXBwbozAx/Zwm33LGXIPSuZNlE9EKMg7ZjunoQQQgijJBhrUmkkrpvFlt+TKKgEG1dN53RPN2zK04j9+lliCxWbxC7mi68Pkl0KDp796DdsFL26umFRmsbxr5/laCNHFbYML3qF9sAGKEk6WN2nSk/6tyReUo9QDK7j9UhN7yhHv/2NzHKw6TGZMWFepG18lh3H06gwd8Sr13AGDhuMv4s1FZcOsmOtcsBATUd+co5yVBFkJn72Nr+fywdbLwLDRjEwrB9ejpqpLj6rby42IYQQojaZ9LXZ9KBdYGesASovk51g7AXhXjj06IW9uan5RaPoTKhacTmWbJXhqSi8/vkt0/q4GZy0tprLYNw7qmdkkwlYhRBCNJYEY0JoWNh6UVGcBl0WM+2xO/CyTOPw8mn8cU3XTgohhLjeSTAmhIb/7B8Z38MaLK2xMIeCmDV89LE0OwohhGhe0mdMCI38/DzKgfLyfNKOf8lXEogJIYRoAVIzJoQQQgjRiqRmTAghhBCiFUkwJoQQQgjRiiQYE0IIIYRoRRKMCSGEEEK0IgnGhBBCCCFakQRjQgghhBCtSIIxIYQQQohWJMGYEEIIIUQrkmBMCCGEEKIVSTAmhBBCCNGKJBgTQgghhGhFEowJIYQQQrQiCcaEEEIIIVqRBGNCCCGEEK1IgjEhhBBCiFYkwZgQQgghRCuSYEwIIYQQohVJMCaEEEII0YokGBNCCCGEaEUSjAkhhBBCtCIJxoQQQgghWpEEY0IIIYQQrUiCMSGEEEKIViTBmBBCCCFEK5JgTAghhBCiFUkwJoQQQgjRiiQYE0IIIYRoRRKMCSGEEEK0IgnGhBBCCCFakQRjQgghhBCtSIIxIYQQQohWJMGYEM0tYi7L3l7G3AhlwrXBY8wClr2+iAmeypRrVThzX17JskfClQnXIA/GPb6MN56agIcySQghNCQY0xq/gDfeXsmyeSOVKUJcHSd3fHx9cHdSJlwb7ro9nIBuYYycrExpqMksenslK/WWN1j61CwmBDdlKNIedz8ffDzbKxOuQXcxNiIA/8EjuUuZJIQQGhKMacyMCMff14eAvmFIOCZMctsS3tv0BW/MUSZcX775cT/xp/eza5sypaE64u3rg0+njnR01yye/oQMHsfMF97hvcUNrx0at/g9vtj0BrOUCdeNb9gVGU985C6+USYJIYSGBGMAnrMI9oXCnBxw6Un4eGUGIQywsMLR1gorK2XC9UW1exXPPb+KnenKlMYpPLWZGffPUC8zpjL1/9ayPxU8+k9k1i3K3PWzsnXEytYKa2XCdUPFznef47l3d6JSJgkhhIaFn5/ff7QfqqqqqKysxMzMDDMzM/2cNzCPu6dzb4AZMR//TnFoMF3tKvhm3yllNsCDkIn3MH30cAYNHERodweyYhLJUeYKnsA9d9/G8IGDGNS3Kw45J0m8rMjUbSTT757IrYM1eUpUJKoKGphH/3yCvcvIOZ1W63zU/AmNCKabuwXJF5U51Gl+TsWkavbvETySiZN0jq28hm6hhPf2o11JKqr8etZ7hhDWzx/XklQchs3igXHDCfYs5FhcHY8mzxAm3D2d24YOYtDAYLwrczhV63xNvW5/Rs6YzsSIoep8noWkxqmovoPdQgnv7YVFUgXB02Yy6ZahdHWI4uR5dbLe9xjsTVnuKdI098C/fzjBvj3pF+hGRXoaudZ+dLBMrk7XExDBpJtduXz8OyLjtSs9CBncl4BuHbBI0p677vmG0tW5BNU5zfl2CyW8dzfDx1Dec6PlRqH6PmjOQ+e++Fbf52A8ClOJr28/DGDc3f7YqY7zXc2FwuVEoooCGDPQn/ZmaWxP72DCtVjj0y+YzgH9CHSrQKXKxbyL9l4FEjGpD66Xj/NdnE1NeQn2oDAtXr88gvpej5rIXbffytC6fm8bc831fCf+/cMJ7tyO4hQVHv3DCe6k+L3TK+ehdHXI4uR5/RLsf9ss7rqpXL/MCiFuSGZDhw6t0n6orKykvLwcc3PzNhSMeTD3nfcY6RDN2jmv4fPmViZ4xrPt/ufYrJvNcxyLnptJmKcVVJRRhhVWFkBODBuff4nt6ep9jXvqBWYO9sAKKCtDXWtSkUPMF8/x0g/qACT0/mXMvSMAFwsoKy7DytYKKCNl3woWrolWn9XEpSybEaLIU0jiT6t49uNoIJS5by9ipK8VlGnOxwrKsqLZ/J/XNOejK4QFq5cS7hLP5vufQ7dFyuOhN3jvNn8Sd0zl2Q2Ka9Aeu6KQxN3aYwPzVrL1FldiNjzASzt0dqZcP34pGx7oRsbZUvy7uQBQeHIjD7y4XWcjjf5zWfnUSHys1MfFygorizJyjmzmueXb1TULut+DznXrfw9A/5kse2QCAS7qL6LMXP19lV3Yy4qn1hKN9lytSTzbHv9u6uqtlH1TWbhG5x5UlKm/R1srve9x7jtbGemtc+7V2+qvg5p7cLb6Xmn3356MyDW89u5+VMbOt/cC3ns+nPanNzPjeb1vj1lvvsc430S23/MsO42WGwPmrWTrLbB36kLWVn+2Jv60IwE97fX2E//tMzz3RR2BNHNZuXUkrga/X520b7uacC2J9N06Eh+dVEjRnOMElm6YSbe0GLL9QvAx1/l9zI1h43M65UD5e1upKS+Fiexc/Szrj2jyNeaaey/ivefDsD62ljmv7tVJUJ9fiGonjz2znrve2cpI9jL1ybUAeIxZxAv/DMNDr5xD4eltrHh+MzFQfb98KCTm4wd46Sed3QshbjjSTNl7OsHekJMYxV5g45F4ymwD6PuAbiYPJjwynTDPMhJ3r+Kxe2Yw457HWLUjnkKXEKY/PRMPwGPio0wf7EHZ+Z2smj+VGTOm8tg724kvcCHknkXM9AT6L2DWHQE4ZEax/v+mMuP+GUz9v/VEpYPPLbNY1B8gnFkTQnApiWezNs/89URn2eMfMZ4JAA/cw0hfK1SRq3hshro56LWfEsEtlJGTDfXMiWHzqRSw9affNN31HtzV2x+K4zm0ATymLWDmYA9I3c9a3fPLtML/trksaWAzk5o9/t2sifn6JR6b+hj/+SxKmQGAmdNG4mOuYv87j6mbuO55jZ3nwaX/SO7yRO97iN+hvW6d72H+dE2fpFAWPDCBAIccorc8y9QZM5hxz7OsP6wC35HMXazbK9ADf78M9q95lqlTn+W/u3S+x/jtrHpS3dxW/T1OXcB0T9j52SpW/RBPIaA6uIpV765i0y6d3dYjdN4SZg5uz+WDG9WBWPX5qoj6uPb5znoqFE5s5mQqWHXpx3TdnXneRXAXKEs4xEZTyo3JPAjodLZ6P4+9s5+UMnsCbpnVqD6VHg/54wNkZ0aZeC072fTuKrbHFwIqot5dxap3N7FTJ7t9txBc4zbzrPb3MTKFMucQJj6kPUMPps+fSZgnpESuVeebMZVnP45CZeXPuEeWKK6lgdd8Yhfx6eDir+hnOr4v3ewh8ZTmHwg94cycHIZHRSI7teX8SfV12vecwD1TtPlSSM0spCw3hYRj+nsQQtx42nwwFjIqAA9yOHdA85/tlpMkFoN/8KyazsaedzEy2J6y+J2s+Gi/5g+siv0bVvFLfA6Xix0IwIO7RoZgXxzPzrfWs1/zn7nqwEZW/RZPTk4ZDoEw4Y5QPCxURH20gp1nNfs/u5MV22LIwYPAW8MBR6zMgaLLnNfmSd/J+ndXsOLN99kOoKlNK7gSX/0HP/rjtax49zX+u6b2IwBAte0kiRVWBPTXuTZtMHrqF7bhwV2DA7Aqjmf766vYq3t+H/1OSoULIaP0Hp8myzm2gZe2xKBCReJZw+fnYAGUFXC5Oj2a9e+vYNXy/7I2Xed7OP09z23Q/x5+Ow9W3foyDmD8eEI9QXVkLa99najZVyI73/yG6NQc8ivVNXRaiXteY9W+RCCRxLM13+P3z23U/x4PJIJtAH3HQuKR/ey/XAJAafF+9kfuJ1p7v+oROm8li27xoeDEZl58W9OPSHu+h99nxU/65xuTAx5BYwlHxTcnNMd/qCbYDrknGB9yiNm7zbRyY7Iy4ne/xDbNflQHVrE3rhBcPAhUZlWwah/MgscXVC+LXljJm2P8ITeGvdtUYNK1JBIduZ/LpQClFETuZ39kNNq7A0BONBte3KZZp2L/u7s4nQsuHpoz9LyLgQFWlMVv57V391Zvm/jTCt7fnwIuIdyq949JQ685hr1nVeDSlTCdf1Imh/XEviKRkx8bKuep7N21k51bN7H+gCY9fT8bV0ejwgqfoHGafNtZMe8BZsx6js21armFEDeaNh6MhTM+yANyzvF7pHbdZo6eLwPfwJp5lwa64wqkxG9W/KerYuNzc3js+bXsJwz39kB6XK0/nqoNzzFn3nOsjfSgawd7qHAk4J+KKQAmdMUBsLZtD+xkf3wOuIWxZP17vPHCIubOGIdPURRRJzVn8P0xEsus8L/jHT56ZxlLHp/F5MGOpCgfWLrS13MoQf/a1MGoipM79kI918CJbzifCVaePdQBT4MUcuGkbjOOYd8cT6TM1p8Jb37EypeXsOChyYTZpbD/iOaKNN8D7sMU0ycsIdRV3aHeAfDo0hF7ckg5qmyW28uKJ+ew8E3dpjEVqT/qfquae4AHwxTTNCzp2xEAKwed7A3gGLSMBbf4YJUVxYaXa2pN1OcLjt0eUlzXBLraAtZWtAdUHx8ivhj8g7SjEkMYG+AB6SfZsQ/Tyo3JLqOq/p1oGCvvUMIjwquXsGAfSI9m+9qa5kPj12Jcmeo8+qVqJ5l5Oh/r/L2FmG3n1cFPgG5pbvg1x3wZTWKFC10Ha+vGJtOvi5Wmds+QRKK/Xc/6WCvGzZirDlgfmky4D5QCmF/no0GEEI3StoOxW4bR1QVwCmHepi/4QrNM6WYFFv6E3hOizufTHnsKyc9S7kCXD+3toTBf2SNZVxjtnIGyUvUfXj35ZFxIISVdvf3e5c+xakc0KWWO+ASGMXLSLJa8/gXvPTVO/fBK38izz69nf8JlrN38CY0Yx/SnlvLeR8uY2V+57xrbok5TWH1tI9XBaHo8e09g5BpUlFYAltCox0WlckVtqg3P8tzH+4nPs6Zjt1DCb5vOohfe46OXZxKK9ntQ98VTKr2SQsqFVDKAMDdHIJ8ckx6spZTqBZ7qe0Bp7W+IsmxSLqSQqnyym8Qe/4EBkFMIbqFMfKCmRkh9vmWUGriu/MwUUpIzUX8j2zh0phC6hDK9N3DLeALdQXV2r6afkQnlxmTK+2K6wpMbmTp1qt7ywJOvsVHbPwtMuhZjykpr9dTXV9/vbbrmd1Av+GnENadvJ+4CuASGq5uBp/TD37aMxON1zxMS+tAbbFi2hFkThxEWFkbYqOksWByu6B8nhGhL2nQwNnJwV1woI+diBhmZOkt6Cjll4NFtJCEAyZcpxJ72ig7b+lK4XAj27er7k7qdzMtA8Tm+eWohCw0sz32wX5NXxf4Nr7Hw0QfUfc9e3EhUOngMvotZ2pncz+5k1fOP8cD9M5g69VnW/hRPoUsAE6bN1Dmmwo79nM7RXNstYXR1gcRj2k7D9V1DCA7WQEEhmsGGzSLxp1U8N099zVP/by074wtx6TmBex7Qfg+QcaT2fVMvK9gGbM/MBlxxv025d1Oo7wHZUQb2r15WfK3cxjSF53eyas4GonOs8B+l7ntG9fkWcO772sda+NRCFj6/Fm2p2P7baXLwIGBkiLr8ViQS/aVu+GJCublGGL+Wq1Tf721vB6yB/LyrLc0q1p9MBPtu9B0P0/v6Y1V4mkN1lRHPudx/mz/2mfur+yOqvydts7sQoi1qw8HYZG4NcoH0KP6rfPg9tZAf4grBM4CxvYEd58moAJ+guergrFoIC1Z/wRdvzyWA7ZxXAZ0CmNtbLxMhT77HF1+sZG5PiEnLAZeuDFN2hL9lAW+8vZSZQ9HMZP4eK5+p6RasOrmdFZGJlOGCR08If2QZK9csUw8KACCRvR+vIzodcO1ST1PiXnbEqsAzmIfuDMGlOJ5D1X1btNcQzAJl7Zqm5iInJVoduFUAWGGj12TnwSx/Q4GcKdSvuHnvZfVgCADO7mX9x9GogI5+46q/B73+fAB4MPP5lSx7cgIBALEqcrCn20Bll/XJLNvwBR89X99U85p74BvMLMXrgTweWMrKlxcwIUB/vWkKORu5nmj28tr3MRTaBjDuEc35xarIwYWuQ5XdxEey4PWVLH1A57U/+3ZwMh08ej3EhCAXyhIOsb66Nsd4ubmm1HstTUD7exu8QF2zqmPkbYF4kEPKsSYI/jYcI77Ynm79FhHsZ0VO3P66++f1VzedquI2V/dHBKCXB7XeJ+AZQkg35UohxI2o7QZjU/rhb1t3s4j2v/bA8SOBjew8lgPeI3ni5VmMDPbAI3gks154gjBPKy6nRBEPbPwxmhwLH0bOX8asUSF4eIYw8qGlPDHQA6ucVKJOw/4Nv5NY5kLoP1eyYFIo/ngQMmoWy+4Lx9/TEc4C5GDl5IHPwAdY9tBI/AH//pNZFOGPFSrOH4T4Yis6ugcw7rlF6lfNeIYw8qHZhHpC2cWTeqPOlGK+PEkKLvh4W2k67tfYuCUKlYUH4XPfYO6kUPzxJ3TSAlb+MxSXshSid2h66RxU97kJGKO51m6hTH7mBW7torOzBomnwLYjHj3H8cIzEwjxVM91NuuhUDwoI+XUzprvocutLHtB/T2gOe643j542BQQDxC5nt/Pl2EfPJ33nplMaDf1/Vvw9p0E2IMqXlvPZJj6e/Tn1v8sVV8b/oROWsQLY0Pw8bSnQDuFVnIh+UDHoEVMjpjOZFMnC97xPjvjy7APnqgenao5X5e+D7DycfX5egSPZNayBwjv1hFH9VVpaEbFuvjgY6vt7K5lvNxcW+q7FrXzeflAR4KfmUz4jMkNHBG6kS8Pq8AznAWvz2Vyf391eXl8JQ/0d6HsQrTJ/dPqp+5nah8cRoCtzmAgQzQBokffBcy6xV89x9+kRbxwZ4Ci+X8mb7y9lKWvf8Sia6xGUwjR9NpsMDazv/oBFb/XUCgG7IviXE7NsPW9y9ey/aQKh57jmPvCe7z3wlzGBTtw+dhmVryt6Si+7zXW/hCDyiGAcY8s5b3VS5l7WwgOOdFsfmuFem6r9I2seG8nifgQPmMJb2x9j6WPjCPATkXUpyvYmA6wl9c+2E5MphUBt83lja1beWPxdMLcy0j8aT2rToBqwwo2RqZQ5h7GzBfeY+vqpcy9LQCrzGg2v1f7oaYn/RvNxKY5nDuoeHAcWcGLm6JR2fkzcsYS3tj6BktmhONDCvs/fY21JzT5Tqxi/b4Uypw11/r6Eqb3hZgGdxTXUrHxrY3sv1CGx8CZLF29VX2PA6xQHdnMKk2zz97lr7L52GUcgtXfw9bXlzB9oAcFp7ezdrn2WnT3NZ0lr6vvX3gnSNy9hlVbjJzjvtd4dUs0l51D1Ne29Q2WzAjDoyCe7R+8VtNp/MRGdp7IwcozjOmPT2aYn/5u6qZi8+qd6qB86iJCUbHxrTXsPAs+Eerzrb72gxtZsUExbGTbSRIrgJxzROkFE8bLzbWm7mtRi/liJzE5VngMnM6CScMa3K8q+u0X2XhEhVW3kUxf/Ia6vET4wIX9bHxjrcF/xBpj2/FEygAy49hRb1/FjXy5O5FChwDGzXuDrVvfYMm0EAr2RRlupqwopcxQF04hxA1FJn1tqG6hhPvYA1CYUtd0Bv6ERnhjD1CQWjMaUI8HIYMDaGcNlF4h/mCMwT/G/v3D8XYAKCTV0EhJzxDCAtupJ2jNijdx1JxmAlj289L8VXU8kEw7v5r7Ucf5NYJHcBgBbuqpO67ERRFjqOnKpO/BxH3VyZTvUXMuHmV136MGMOl8NRPAcuAlHnvH8LdntNxcK0y4FjTX41Fsavk2oFG/J81I53zqLL+e/viTSKKhMiCEuKFIMNYWjV/KhgdCyPjpMZ41OBeSuJZNeGEDM4My2Pnks03bx6oV3EjXIoRQq6qqDiuuey0VC0kw1oaMe3wZY7va4+jpg0uB+vVP9fRuEdeUcSxYNpYuDo509Hah4Mha5lQ3y15vbqRrEUJwgwVgdWnOuKjN9hlri/Kxx9W9I9aqaP2+T+I6kA8OrnR0sybjmG7/uOvRjXQtQrRtVVVVtQIx7bobYTF0Xc1BasaEEEII0WC6gYn2Z93Y4XqOI1r62qRmTAghhBANogxWzMzMai3XM+W1mJmZ1brmpiTBmBBCCCFMZigouRECsLroXpuha28KEowJIYQQwiSGghFz87YRSmiv09A9uFpt4w4KIYQQosm0tUBMy1BA1hTa1l0UQgghRKNoA5AqzajCG7VZ0hht/zHd+3G1JBgTQgghRIPcyH3EjGmOa5dgTAghhBD1UtaKCf17cbX3ROYZE0IIIUS9lMFYY+IEp6ARjAhqj6UywYDyrNP8GnmKPGVCo9jSMSycId5OyoQa5Zc5/duvnMpRJhhWVVVFZWWlXi1ZQ++HLgnGhBBCCFEvbTBWWVlJZWUlVlZWyiz1sh37BP8Z19mkQEyr/PxP/GfVHoqVCQ0UNPNfzOrrplxdW3kSO5//L3tMPGBZWRnm5ubVnfqvJm6SYEwIIYQQ9braYKzv7GXc28uSjOOHOJOrTFVqR/ewEDqWnuLz59dxTJncQBOeeYuITrkk/H4ClTJRw7nHIEI8yzn1xXOs/1OZapgEY0IIIYRoMU0TjGFisBPKrJfvJYimDMZSiVz4NtuViRoNOz+1pgzGpAO/EEIIIZpYKLNefosVL8+mrzJJx4Rn3mLFyqeYoEy4KhNYuPItVjzTtHttThKMCSGEEEK0IgnGhBBCCNFyPAcybvIkJk2eRCd7ACc6aT6PC+uozN0mSDAmhBBCCKOudi6tar69CR8WztBh4fRwAXCmh+ZzeIi3Mvc1rynuiwRjQgghhGg5MdvZ8P77fPD++xxWAWRwWPN5ww9/K3O3CRKMCSGEEKLlFGeQEJdAQlwCReUA5RRpPieoTJzk6wYjwZgQQgghWkV2aiqpyRlkKxPaGAnGhBBCCNEqDnzxNitXfs4BZUIbI5O+KrRr1w4nJyfs7OywtGzIixuEEEKIG4/2fZTa9zGWl5dz/vx5ZTYF/Ylbyx96gQdC7MlTZZBrtCXSEudOHXEqjGHDC58So0w2agILV47A++KvLHpzOxMWvkWEXzHZyVkUKbNqWLp0pKNzITGfvsiG48pUw7p06QJAaWkp+fn55OYafbVAnSQY02jXrh0eHh4UFRWRl5dHUVERpaWlymxCCCFEm6MMyIzPwK+YRb/LBBY+MgJvW2W+OlQWk/r7h6z8LkmZYgL9YMw27J8svjsEJyNtgcXJv/LRyu2YesSysjJsbW2xt7fH2dkZOzs7MjMzuXLlijKrURKMAZ6entjZ2ZGRkUFhYaEyWQghhGjTrjoYUyY3K/1grLloX4dkZmaGmZkZ9vb2eHp6UlRURHp6ujJ7vdp8MObp6YmlpSUpKSnKJCGEEEI06t2UmmDMOoOYqARMbcArSv2DnVEZytUNpAnGchI4EGNkX6Wp/LHjMEZyGVTXuyl9fHwoLy9vUEDWpoOxdu3a4erqSmJiojJJCCGEEBoND8ZsGbXgRcZ1aVjf6+LYz/n3umjl6gYK4t7nZ9PXVbnegPzG19zVFYwB+Pv7k52dbXKTZZsOxnr06EFqaqo0TQohhBD1aHgwBuBMUEQEPd1MD8iapmYMsO3IwIgheDsoExSaoWYMwN7eHm9vbxISEnS2qFubDca0oyaleVIIIYSonzIYs7S0vOHjhPpUVVVVx0uGgjE0zZV5eXkm1Y4ZGVtw43JyciIvL0+5WgghhBBGVFZWKle1KaZcf15eHk5OTsrVBrXZYMzOzo6iorpmHBFCCCGEIWZmZiYFIzeyysrKWjVhSkVFRdjZ2SlXG9RmgzFLS0uZR0wIIYRoBG1zZVtk6rWXlpaaPHl8mw3GhBBCCGEabS2Q7rxa5eXlymxtQnl5efU9qKu/WENJMCaEEEKIBtE2Vba1FqbS0lKTmigbSoIxIYQQQhilWztmbm6OhYUF5eXllJSUKLPekEpKSigvL8fCwqLeUZSNIcGYEEIIIRrMzMwMCwsLKioqKCwspKKiQpnlhqB7fRYWFk0SfCm12XnGgoKCOHXqlHK1aTzm8PrEIGrGSBTz9w//4gOVXi5hoqkvrGOsb83n5J9n89IW3RzXuGlLWTfar+bzhV3MfnGrbg6Thc1fzZw+meya/RKN24MhoTwy7T5ucoKUA0/xZiOLvWhBg59g9azeZF5vvwuiTdCdc6yqqoqKigoqKiooKyvD0tISKyur6tqj6zGW0L5/U3tN5eXl1dekDcZMrRUzNdaQYKyhtIHYxV9Z8OMPylRxNa77B1AYT6ycQ+8cCcaa0p23v81Ip1Ns3PIRR5WJN6qr/F2Y+sI6xrqc4KOF/yVKmShEE1AGZNpJUCsqKqpHG2rXX290O+drm2O1k9w2JBCjAbGGNFM2UL8e/thRzN9HJBAT14toPtjyFAvWXX+BmBDi2qTsP2Zubo6lpSU2NjZYW1tjY2ODjY0Ntra2192iPXftdVhaWupdp+71NxUJxholi3RpkhRCCNGGaQMSbW2RtmlS21R5vS/aIEzZ5NrUgRjSTNnwaoJ+Q19lZlAWe9e9xffKRG1zUN6vLDjSUa9fWdGpz/i/A/pvor/z9rcZ2anms16eoKdZNRT2/pDL4IlB2JHK3nX78NQ0N9GYZlJN0wK9jO4AABQZSURBVEft+YCLOLF+Pv89qPlYK1+ywWYzZV+vouMfMX+1olGk1r60FMfEtKYZdTNezd4MHtNkmmbF6rdVGDgnprJ03VhqeoQZyqNlrJlSebza51/TTHmKIJ3jKvOZpqZ5Us1A30aTy9lEnpk9Ag48xV7XV5kZZKvZQaqB3wV1Xp/qz7WPW/N7pHMsgDxNc2StfpkK2nzK9XVSf4/udd5v/fJtUtmuVTaUvyc65WG3l97vQe39KfelVqv/pLJ/ou4x6/xd08hTNFvWyl9f2RaibsqmSO1n5frrSV2Bl/KzMabGGhKMmSLoaVYN9VaurVH9wNJ9+Ok8gDTb1/TX0TysdB8oyr5oQU+zaqgbRXlZHNySQMDsEbjlFUPK12xlSj0BYR00f3ipfghoAwP9B4g22NF9CKgfTLp/qDUPDt0/7to/7Lqd1008ZjUjwZj6PGo/fGr23wDa89V7QIXxxAthRL2o+ax58Ok+OA3dnxr1B2NTX1iN126dh11d++8KRU52NfdBk8/wMU2kKU/KoMj0cqYTYFWXW015R6cca8q67j8W6sDLVq+/mnYdev+EaI6h+EejafqMmR6M1Spnhr47A2Wv9na6wbfO74/y+zSwL4O/C4OfYPWYNOZXl626f5+M9xmbytKVXuzSSa/9ey5Ew13PAVhdriYeMjXWkGZKU5x6iwXr1H1uNp4q1tQGqD8vWPeUgRqqVPau03nonUogBXBzDQWg39BB+JDKXt2Hi+ojtp4qhk6DeMRDu9IWUvZVB1x2JLK1unbNDc/qfMaF9e+BHcn8Vv0giuK/h5IBP4KmaXNNZWwfO4qOf6T30N/64i6SsaP3mKkAhM0fjh/Jen/IOfhfPjteBL7DeWKwepVpxzTR4CcY7gvJP+s8dDTHtOszFvWZmW7qGGUghvr8tIEYYTwxyA8u7NJ7eEet/owTeeA36AnCqteaZuuLiofcll2cyAO7rmH6+9INxHTyNeaYpmlAOdOrkYrmgxOp4ORPfw/U/4z09oaLv+rVAh898DV/54FP7zn0q9kToB5QUJP3B+IvAk4da+VrOWF4uQAXTukFN8rvTlt+PtMpG+rfEz+Gz1d+S8nsmq2z/ZZTJAPuHup8hvZl0MH/6gRiXN3vE1t5SRGobd19giLs6NFfef5CmE7byV13uZ4oz72lzl+CseaQl0uK3oofeHOd9qETSn8fWwN54GhCIkXY4uymXVNMYoJO02ZeRqNrBvxc62q8KCInWfPjtCD8gEyV8qGwlVMXABcvwggjrKsd5OWg3Uwr6kgCRdjhomlFMemYJtIGdqcUNUNRqkzAHS9NAGiaqQT5QtG5qDpqDYDBYfRwgqJs5YlGEXWuCJxcajUpNVwUaTnKdUCt69Tka5JjGmJ6OStKOaqflpVbU2Y9+uHvBEVXlCU7miMpxeDkrNN0CVBMbpbeCr7/8SkWXFUN2NXS3Gvfsax7oa4Qv67yk0xOHti5Kr6lWr8rW3lp9mxNkF/XvkyUnEORcl1jHUwjU7lOiCagDG6u5aW1SDDWWup54Glr0JqS+r9e3f/aNTU/eQlE6TVJ1BMo6QYDOWl1Pjy0//Gbfkzj1IGdH2PXrWOd7qLXf8ZEg71wNxh01lZ3noYGgJpmJsX56/ZLqk9ydpM9cptdVrZ+38gaDavNbS1bX5zNR8eL1AGZ5ntarVvbpSk/dn3m6JfFdfr9AZvD1BcU5b++PmJGhM1frTj/2n3WhBAtQ4Kxa1DdD7Or4OeCnd4DZA69UTbTGVHrP3zDqgOYpjimhjoYSWbX7NnMrrU0sI+LpgZAGzQ2TiZpDTmmbp86nXPfdUGZ0TA/V8O1kdeX62cUctTq+Zrv6CN1U3KfOTUBmab8FB3/yEBZnG2wv2BTqOnTpXOs9ScaVTOm2/ex5tx3XeflS4jrlwRjLU7TZNOpB3cqUtRzmKUSb7yvXwPV9H/Se2gog6Itp0g22GdEtylF00znG1Srn5Z+U6KJxzRRg5sjpy1VB4ArDfWz0jQnKftq6ToYRUIe+PWsdZXqZlpFnyJj1PemiBO7G7KVlvr+11cb2Vr0yqzqKIl54OM/UZFL0zR/McH0ASc6Uq4YauJsCprvsl5R/HehJiCrbn6sozmyUQzva+oYZY2Xpgxc+M2kfzySs+trStd2NTjBrsYOCBFCNCkJxlrB0QOHSMGbkbfrPLQ85jA1yJaiUzUdqZuOph+Mi1fdwQcAW9l1vAi7PvdVd8IHmPrCWPx0OuJHrf6NZPwYq9unZvAT3NfHjqLjuzRBiqnHNNGWXZzIs6P3rKW1gkBDpvbUPIacehBWK4DTdHx26s0cvX5BYTzxgjZ40+TxHctSnc7RYfPvo7dTw4MqdTCpG+iG8cRK05optfd/VzPVuDRarTKr6dDfaQTPBNVk6zd0Cjc5NX6i5KPZWYA3g4deTfO9MgCvPc2I2lSWKgP4Wv0Ha8qGXvNlo2j/udEZ+DJ/NWNdihQ1Xurz1/snaNrSOpsp1eXN0GACdPog6vxuDH6C1dJMKUSrkaktGsikecZ0h/rXSTkXk+JVNXpTEegP+VefA7WnKaiX4XmMMDCXkXIur1rzE4HB/Sn3YyiPVk3euvPUnrep9vxPUMe7ILXzMRk8dw1T5lmqb14nDYPnpKF7T5T3Nfnn2ZzqqZiCoNY51XX/TVDfXF3aUZEml7Pa5RVlmdWqNRVM7bnIGlqGdafCAOWoTlPpl7Wi4x8xXzWWdaPR/04NfAe1y7bhfPplSBPwmdA0r1eGLuxi9ouwdN1Y0Duu4ncl7wQffQX3KafA0FCWN/1ypAxGk6vntlNO/yGEaDxTYw0JxtoEA/OCaVOabW6h1jimaD7qYMzNwOTFQgghDDM11pBmyrZAM/rL0PD5Zhul1xrHFEIIIa5DEoy1BZrRX7U6rGv6eTVmqgmjWuOYQgghxHVIminbDMN9s2q/I68ptcYxRfOQZkohhGgoU2MNCcaEEEIIIZqBqbGGNFMKIYQQQrQiCcaEEEIIIVqRBGNCCCGEEK1IgrEb1NQX6noVkDDovmV89tkyZijXA+DBkukjWDM9hMnKJHENMfI9hS/iw88+ZFG4MkEIIVqXBGOitsFPsHrdOr3XAF1XGnr+4Yv4cFxnCo/9jy+UaW1FB3/emj6CJT2UCTeQ/Sv4Ndmemx+tK+gWQojWIcGYaONmsOzRm7FP/pGH3/pDmaiRz5UCoCCfC8qkNmJA/zDWTB/Bmon+DNBNuKaCOOPf0xf/WsNfeZ25fe0ihigThRCilUgwJtq0IU+PoDNJ/PivNlsn1gBFFNGZUddE4NVYf7Di878odLqZfz4t4ZgQ4tog84yZSP3S3UzFS6LVk5oqX6xb6wW9hl5kbSCfcjLUmmPuwkv3pb4G3veo3BcYyFfrxcb674es74XX1HpZsvJFw7XPvyHqPLbi3imvU/eYde5Do/bLnmew7LPbcT+2pp5aMRN18Oet0Z117m0WezbHsE0vk7qG6cEeOuefcJinjxTq5QGgRwhr+rvVfM74m3l7tW/UtmfexIEE5//NvBMOesc1tL/6jjl55AhGddTJrHDhyK+8lqDdB5xMKCTYOaPmXDTXfUmTT3edbjk7+XMUay5pPvYIYU1/2PNzPkNHd8aOLPZsPovnxIEEOyivtf7zb6wZr37G7X5J/Hjfc223aVoI0exMjTUkGDORqcGYOh96QU7Y/NWMVc3XCwTUgUNyzf40gRK19qV+CNUEHZpZ7XWClFr70q5z0Q3GprJ0pRe7dIKzOl/YrTmXzFrBS42pL6zGa7fOdtOWsm60X6MCslrnX8e+auUzcM/QWV/f+YOm0/44d/56/2FW7FcmNoAm+EA3SOjgz1u9C3haJ6hQBz46QZqh7XQCJG0gBOoAZgkxms+aYMwB/UBHE8DpbmfqMQ0GVTqqg7GfE2k32p8r2mPW2s6DJRMd2PNDIn9qtlWfg/I87SkqKOTADxkETr+JDgVFcDGGLwnhwR6F1edr8vk31H3L+GxcZ5J23sdznykThRCiaZgaa0gzZRPzc6393sWo1fqBGIOfYLgvJP+sE9gd/C+fHS/Crs9YpupkRVOjUxNsbOXUBcDFSz1S0tC+DNrKS4ratK27T1CEHT36N3zM5dYXFQHcll2cyDPwLkqjphLkC0XHd9Wcv6F9GbrOeu6ZKWbc1Bny4vnjagIxgPaO2FHE2fM6wcGlRL1AjA7+DO0IF47o1JZdSuTLhCLsenSrGf1XnU8RFCVoAzFdWezZrFPjlJDBBaBDO3v1Z1OP2SAq9iRAty6aY9Si4jWdQAxg24kkirBTbGMHF89Wn5cdKr6sDq7s8ezQXOev8VksSYB7R2mqFEK0PgnGmlhydhE49WZOPdNKhPXvgR3JnFLU2kSpMgF3vAbrri0iJ1n3M2x9cTazNYFVXfsyieZl3k0jirQc5ToTDPbCXblOKyetOnis6zoN3zNTDKFTe+DyRa6ygRIu51OEHcGjw5jXQZmoNqCLB3ZkEacIqP68UlgTfNSTz6BaHdVVvLb51+oao7r2pTxmQ/15JJFLDQmGLhWgjRdrKILX/AK9AI5mPH+181zOA3u3LsoEIYRocRKMNbGo1fOZ/XOyOiBbt45169ax7gX9ehs/VzvAj7HadO0yWvlK7aYVNn+1/vEMvMTbZJrpI3T3V19/rTod/C+/XUC/dmvaWHo7QfLpmrq+1rpnJrmUyNOb/+YCdgSPHqEedaiY68rX2Q5wY9R0bbpm0e0Xps1XK8hqHFOP2XAq4jLcGNrfcO1Y9cjL6uUmGlM0mu/8hRDi2iLBWHPY8hKzZ89m9uzZ7LoA+I7VC8iSs4uAZHZp8ugviua/JqLtf5b8s+6xdqGodDONdiDAhV16576rURFEGF4u6Adao/1qdbZvjXvWMOpaqXmbf2XekSxNEFETkF3ILdI0K2ry6C01TY0XcovAwbFRwYuSqcdsjG0nkqBTR/1pLnQ62184onusvxsVXDbn+QshxLVEgrGrMS3IaM3S1hc1QYq2j9dVNa3VZnBfmv5VNcII62oHeSfYZUpzpqb50t3DcEOrusmwiBO76++lVq26Fm1p7b5d08bS26mIE+v1Ayxlx3uD11kXI+ev9gcXLwPtOzX9fFMJMZqArKYpzdSmNXU+NwKbYPoIU48JNU2J1f3NjLmUwVk8GNBed6U9AzrZQUESe0xpZjWiQeffYF1o7wSFWeeVCUII0eIkGDOROhjwI0g7q7tmxJ++MJ5YqQw41B3Udfs/qTuo29F7ljJvI2w5RTJ29B6j3dNUls7qDXlFOpk0/bmcehCmDWYGP8HqOpspk8nJUzQd6lDfC92O/2E8sbLuZkp18Ib+/dNKzqEIO1wMn0iNBt2z+s9f64u/k8ApgCFX+XqcAf3Dak16OtnXDSgkvbpz/VlOFtgRPLqOV/VoJZzlZAH49lf0P+sRUusYRpl6TKieMNX0jvGFrDlVSHCQh9669HzAwYMB2nPv4M9bjWymbNj5N9B9vegMZGZcdY9BIYS4ajK1RQPoz3GVXD3/V49zulMraKae0NlOOUWDlsF5sXSmrDA0TYZBevOHqaeqiOq/mjldE3SmtlDOC5bMrtmnCDIwT5pa7evQbTpUzveV/PNsTvVUTqehUX1++tNvaCn3VU05T5oJ96xG/eev1nTzjNWar6sgiU8UowoxlI/a82phKJ+hecYwfAylWvtCuT8tD5YoAqda84zpzhemk79m9KfutBtomhk101dop6PQTG2h3pdmH5rzMXQc08/fdDNe/Yzb2//Fmrkrrn4AhxBC1MHUWEOCMdGqdPuyKaf/qO6XVivQajpDnv6QeX25+rnGxPUjfBEfPnozmTLHmBCimZkaa0gzpWhV6lGStaesaNppN+r2x1u/koQ9N98r7ypsG4aw6N6bsc/7i/9JICaEuEZIMCZalXqUZO2+ZFNfUDcx6k5v0Ty+4Ln31e8qnPfqDGWiuMHMeHUeNzsl8aM0TwohriHSTClaneE+Y4b7lzWb+5bx2TjkXYU3svBFfPhoAPHSJC2EaCGmxhoSjAkhhBBCNANTYw1pphRCCCGEaEUSjAkhhBBCtCIJxoQQQgghWlGbDcbKy8uxtrZWrhZCCCGEuGrW1taUl5crVxvUZoOxoqIi7OyUI/iEEEIIIa6enZ0dRUW6ryasW5sNxvLy8nByqn43kBBCCCFEk3FyciIvL0+52qA2G4xduXIFOzs77O3tlUlCCCGEEI1mb2+PnZ0dV65cUSYZ1GaDMQCVSkXHjso3EAshhBBCNF7Hjv/f3h2zJhKEcRz+F8JiMCoWuUIb/f6fyU4QCzFGWAxTpEmuGLg7HQ622Ocp39fK6sfMsvsrp9OpHv/RqGPs/f09fd9ns9nUKwCAp202m/R9//CpWMYeY0lyPB5TSsl2u3VlCQA0eXl5yXa7TSklx+OxXv/VaD+HVFssFnl7e0vf9/n4+Ejf9/n8/Kx/BgCQfL++Yjqd5vX1NdPpNKfT6akTsR9irLJYLH7/qZPJpF4DACTf7yz9OcRpibAfYgwAYECjf2YMAGBITsa+dV2X1WqV2WzmehIAeEgpJbfbLefzOff7vV4/xMlYkvl8nt1ul+VyKcQAgIdNJpMsl8vsdrvM5/N6/ZDRx1jXdVmv1/UYAOAp6/U6XdfV438afYytVqt6BADQpKUrRh9js9msHgEANGnpitHHmGfEAID/paUrRh9jpZR6BADQpKUrRh9jt9utHgEANGnpitHH2Pl8rkcAAE1aumL0MXa/33M4HOoxAMBTDodD04tfRx9jSXK9XrPf73O5XJruegGAcSql5HK5ZL/f53q91uuH+BwSAMCAnIwBAAxIjAEADEiMAQAMSIwBAAxIjAEADOgLw06Je7mHt0cAAAAASUVORK5CYII=)
"""

from google.colab import userdata

import gradio as gr
import requests
import json
from google.colab import userdata

OPENROUTER_API_KEY = userdata.get('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')

import gradio as gr
import requests
import json
from google.colab import userdata



def chat_with_model(user_input, chat_history):
    # Prepare messages for API
    messages = []
    for role, content in chat_history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_input})

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "openai/gpt-oss-20b:free",
            "messages": messages
        })
    )

    if response.status_code == 200:
        data = response.json()
        reply = data['choices'][0]['message']['content']
    else:
        reply = f"Error {response.status_code}: {response.content}"

    chat_history.append(("user", user_input))
    chat_history.append(("assistant", reply))

    return chat_history, chat_history

with gr.Blocks() as demo:
    chat = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question...")
    state = gr.State([])  # stores the conversation as list of tuples (role, content)

    msg.submit(chat_with_model, inputs=[msg, state], outputs=[chat, state])

demo.launch()

"""OPENROUTER_API_KEY = userdata.get('OPENROUTER_API_KEY')  # or set your key here as a string"""

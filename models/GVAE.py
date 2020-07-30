import tensorflow as tf
from tensorflow.linalg import set_diag, diag_part
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.backend import batch_dot
import tensorflow_probability as tfp
import numpy as np
import time
from max_pooling import MPGM
from losses import *
from utils import *


class VanillaGVAE(Model):
    def __init__(self, n: int, ea: int, na: int, h_dim: int=512, z_dim: int=2):
        """
        Graph Variational Auto Encoder
        Args:
            n : Number of nodes
            na : Number of node attributes
            ea : Number of edge attributes
            h_dim : Hidden dimension
            z_dim : latent dimension
        """
        super().__init__()
        self.n = n
        self.na = na
        self.ea = ea

        self.encoder = tf.keras.Sequential(
            [
                Input(shape=[n*n + n*na + n*n*ea]),
                Dense(units=h_dim, activation='relu'),
                Dense(units=h_dim*2, activation='relu'),
                Dense(units=z_dim*2, ),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Input(shape=[z_dim]),
                Dense(units=h_dim*2, activation='relu'),
                Dense(units=h_dim, activation='relu'),
                Dense(units=(n*n + n*na + n*n*ea), activation='relu'),
            ]
        )
        
    def encode(self, args_in):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        Args:
            A: Adjacency matrix of size n*n
            E: Edge attribute matrix of size n*n*ea
            F: Node attribute matrix of size n*na
        """
        (A, E, F) = args_in
        a = tf.reshape(A, (-1, self.n*self.n))
        e = tf.reshape(E, (-1, self.n*self.n*self.ea))
        f = tf.reshape(F, (-1, self.n*self.na))
        x = tf.concat([a, e, f], axis=1)
        mean, logstd = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logstd
        
    def decode(self, z):
        logits = self.decoder(z)
        logits = tf.cast(logits, dtype=tf.float64)
        delimit_a = self.n*self.n
        delimit_e = self.n*self.n + self.n*self.n*self.ea

        a, e, f = logits[:,:delimit_a], logits[:,delimit_a:delimit_e], logits[:, delimit_e:]
        A = Reshape(target_shape=[self.n, self.n])(a)
        E = Reshape(target_shape=[self.n, self.n, self.ea])(e)
        F = Reshape(target_shape=[self.n, self.na])(f)
        return A, E, F
        
    def reparameterize(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        eps = tf.cast(tf.random.normal(shape=mean.shape), dtype=tf.float64)
        return eps * tf.exp(logstd) + mean



if __name__ == "__main__":

    #Dear TensorFlow,
    #What I always wanted to tell you:
    tf.keras.backend.set_floatx('float64')

    n = 5
    d_e = 3
    d_n = 2
    np.random.seed(seed=11)
    epochs = 111
    batch_size = 64

    train_set = mk_random_graph_ds(n, d_e, d_n, 400, batch_size=batch_size)
    test_set = mk_random_graph_ds(n, d_e, d_n, 100, batch_size=batch_size)

    model = VanillaGVAE(n, d_e, d_n, h_dim=1024)
    optimizer = tf.optimizers.Adam(learning_rate=1e-5)
   
    for epoch in range(epochs):
        # loss.backward
        start_time = time.time()
        for target in train_set:
            with tf.GradientTape() as tape:
                mean, logstd = model.encode(target)
                z = model.reparameterize(mean, logstd)
                prediction = model.decode(z)
                # Penalize the latent space?
                log_pz = log_normal_pdf(z, 0., 0.)
                log_qz_x = log_normal_pdf(z, mean, 2*logstd)
                log_px = mpgm_loss(target, prediction)
                loss = - tf.reduce_mean(log_px + log_pz + log_qz_x)
                print(loss.numpy())
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        end_time = time.time()
        mean_loss = tf.keras.metrics.Mean()
        for test_x in test_set:
            mean, logstd = model.encode(target)
            z = model.reparameterize(mean, logstd)
            prediction = model.decode(z)
            loss = - tf.reduce_mean(mpgm_loss(target, prediction))
            mean_loss(loss)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, mean_loss.result(), end_time - start_time))
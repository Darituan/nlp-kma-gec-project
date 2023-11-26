import tensorflow as tf
import numpy as np


def get_angles(pos, k, d):
    i = k // 2
    angles = pos / np.power(10000, (2 * i) / d)
    return angles


def positional_encoding(positions, d):
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def feed_forward_network(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])
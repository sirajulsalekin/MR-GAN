import tensorflow as tf
import numpy as np


def conv2d(x, kernel_shape, step_size):
    w = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))  # tf.truncated_normal_initializer(mean=0.0, stddev=0.001))
    return tf.nn.conv2d(x, w, strides=step_size, padding='VALID')


def deconv(in_put, kernel_shape, input_shape, deconv_shape, step_size):
    w = tf.get_variable("weights", kernel_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))  # tf.truncated_normal_initializer(mean=0.0, stddev=0.001))
    c_d = tf.to_float(tf.reshape(in_put, input_shape))
    deconv_shape = tf.pack(deconv_shape)
    deconv_out = tf.nn.conv2d_transpose(c_d, w, output_shape=deconv_shape, strides=step_size, padding='VALID')
    return deconv_out


def cal_marginal(raw_marginals, eps=1e-7):
    marg = np.clip(raw_marginals.mean(axis=0), eps, 1. - eps)
    return np.log(marg / (1. - marg))


def marginal_initializer(raw_marginal):
    return tf.constant_initializer(cal_marginal(raw_marginal), tf.float32)


def batch_norm(h, training=True):
    return tf.contrib.layers.batch_norm(h, is_training=training, updates_collections=None, decay=0.9, epsilon=1e-5, scale=True)


def conv_maxout(x, num_pieces=2):
    splited = tf.split(split_dim=3, num_split=num_pieces, value=x)
    h = tf.stack(splited, axis=-1)
    h = tf.reduce_max(h, axis=4)
    return h


def lrelu(x, slope=0.1):
    return tf.maximum(x, slope * x)


def add_nontied_bias(x, initializer=None):

    with tf.variable_scope('add_nontied_bias'):
        if initializer is not None:
            bias = tf.get_variable('bias', shape=x.get_shape().as_list()[1:], trainable=True, initializer=initializer)
        else:
            bias = tf.get_variable('bias', shape=x.get_shape().as_list()[1:], trainable=True, initializer=tf.constant_initializer(0.0))

        output = x + bias

    return output

import tensorflow as tf
import numpy as np


def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    std = gain / np.sqrt(fan_in) # He init
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w


def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[4]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, 1, 1, 1, -1])

def conv3d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, x.shape[4].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    return apply_bias(tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC'))

def conv3d(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, x.shape[4].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC')

def maxpool3d(x, k=2):
    ksize = [1, k, k, k, 1]
    return tf.nn.max_pool3d(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NDHWC')

def upscale3d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale3D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3], 1, s[4]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3] * factor, s[4]])
        return x

def conv_lr_bias(name, x, fmaps):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(conv3d_bias(x, fmaps, 3), alpha=0.1)

def conv_r_bias(name, x, fmaps):
    with tf.variable_scope(name):
        return tf.nn.relu(conv3d_bias(x, fmaps, 3))

def conv_r(name, x, fmaps):
    with tf.variable_scope(name):
        return tf.nn.relu(conv3d(x, fmaps, 3))

def final_conv(name, x, fmaps, gain):
    with tf.variable_scope(name):
        return apply_bias(conv3d(x, fmaps, 1, gain))
'''
def output_block_layer(inputs):
    w = tf.Variable(tf.truncated_normal([1, 1, 1, 64, 1], stddev=0.01), name='end_con3d')
    x = tf.nn.conv3d(input=inputs,
                        filter=w,
                        strides=[1,1,1,1,1],
                        padding='SAME')
    b = tf.get_variable('bias', shape=[x.shape[4]], initializer=tf.initializers.zeros())
    output = tf.Variable(tf.ones(shape=x.shape), name='output')
    output = x + b
    print('output -----> ',output.get_shape())
    return output
'''
def output_block_layer(inputs):
	w = tf.Variable(tf.truncated_normal([1, 1, 1, 64, 1], stddev=0.01), name='end_con3d')
	output = tf.nn.conv3d(input=inputs,
                        filter=w,
			            strides=[1,1,1,1,1],
			            padding='SAME',
						name='output')
	return output

def group_norm(x, name, G=8, eps=1e-5, scope='group_norm') :
    with tf.variable_scope(scope+name, reuse=tf.AUTO_REUSE) :
        N, H, W, S, C = x.get_shape().as_list()
        G = min(G, C)
        # [N, H, W, G, C // G]
        x = tf.reshape(x, [N, H, W, S, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 3, 5], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma_'+name, [1, 1, 1, 1, C], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta_'+name, [1, 1, 1, 1, C], initializer=tf.constant_initializer(0))
        x = tf.reshape(x, [N, H, W, S, C]) * gamma + beta
    return x
####################################################################################################################
def autoencoder(x, width=256, height=256, length=256, **_kwargs):
    x.set_shape([1, height, width, length, 1])

    skips = [x]

    n = x
    n = conv_r('enc_conv0', n, 32)
    n = group_norm(n, 'gn_enc_conv0')
    n = conv_r('enc_conv0b', n, 64)
    n = group_norm(n, 'gn_enc_conv0b')
    skips.append(n)

    n = maxpool3d(n)
    n = conv_r('enc_conv1', n, 64)
    n = group_norm(n, 'gn_enc_conv1')
    n = conv_r('enc_conv1b', n, 128)
    n = group_norm(n, 'gn_enc_conv1b')
    skips.append(n)

    n = maxpool3d(n)
    n = conv_r('enc_conv2', n, 128)
    n = group_norm(n, 'gn_enc_conv2')
    n = conv_r('enc_conv2b', n, 256)
    n = group_norm(n, 'gn_enc_conv2b')
    skips.append(n)

    n = maxpool3d(n)
    n = conv_r('enc_conv3', n, 256)
    n = group_norm(n, 'gn_enc_conv3')
    n = conv_r('enc_conv3b', n, 512)
    n = group_norm(n, 'gn_enc_conv3b')
    #-----------------------------------------------
    n = upscale3d(n)
    # print('upscale1 -----> ',str(n.get_shape()))
    n = tf.concat([n, skips.pop()], axis=-1)
    # print('upscale1 -----> ',str(n.get_shape()))
    n = conv_r('dec_conv4', n, 256)
    n = group_norm(n, 'gn_dec_conv4')
    n = conv_r('dec_conv4b', n, 256)
    n = group_norm(n, 'gn_dec_conv4b')

    n = upscale3d(n)
    # print('upscale2 -----> ',str(n.get_shape()))
    n = tf.concat([n, skips.pop()], axis=-1)
    # print('upscale2 -----> ',str(n.get_shape()))
    n = conv_r('dec_conv3', n, 128)
    n = group_norm(n, 'gn_dec_conv3')
    n = conv_r('dec_conv3b', n, 128)
    n = group_norm(n, 'gn_dec_conv3b')

    n = upscale3d(n)
    # print('upscale3 -----> ',str(n.get_shape()))
    n = tf.concat([n, skips.pop()], axis=-1)
    # print('upscale3 -----> ',str(n.get_shape()))
    n = conv_r('dec_conv2', n, 64)
    n = group_norm(n, 'gn_dec_conv2')
    n = conv_r('dec_conv2b', n, 64)
    n = group_norm(n, 'gn_dec_conv2b')

    #output = final_conv('final_conv', n, 1, gain=1.0)
    output = output_block_layer(n)
    return output

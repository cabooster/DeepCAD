import tensorflow as tf
import numpy as np
from basic_ops import Deconv3D, Conv3D, GN_ReLU, Pool3d, BN_ReLU, ReLU, Deconv3D_keras


"""This script defines the network.
"""


class Network(object):

	def __init__(self, training):
		# configure
		self.num_filters = 32
		self.block_sizes = 3
		self.block_strides = 1
		self.training = training

	def __call__(self, inputs):
		return self._build_network(inputs)
	################################################################################
	# Composite blocks building the network
	################################################################################
	def _build_network(self, inputs):
		print('inputs -----> input ',inputs.get_shape())
		inputs = Conv3D(inputs=inputs,
					    filters=self.num_filters,
                        name='con3d_layer1')
		# inputs = tf.identity(inputs, 'initial_conv')

		skip_inputs = []
		for i in range(1,self.block_sizes+1):
			print('i -----> ',i)
			num_filters = self.num_filters * (2**i)
			inputs, skip_inputs_tensor = self._encoding_block_layer(inputs=inputs, 
                                                filters=num_filters,
						                        block_fn=self._double_conv_block,
						                        name='encode_block_layer{}'.format(i))
			skip_inputs.append(skip_inputs_tensor)

		num_filters = self.num_filters*(2**(self.block_sizes+1))
		inputs = Conv3D(inputs=inputs, filters=num_filters, name='bottom_layer')

		for i in range(1,self.block_sizes+1):
			print('i -----> ',i)
			num_filters = self.num_filters * (2**(self.block_sizes+1-i))
			inputs = self._decoding_block_layer(inputs=inputs, 
                                                skip_inputs=skip_inputs[self.block_sizes-i],
						                        filters=num_filters, 
                                                block_fn=self._double_conv_block,
						                        name='decode_block_layer{}'.format(self.block_sizes-i))
		inputs = self._output_block_layer(inputs=inputs)
		return inputs


	################################################################################
	# Composite blocks building the network
	################################################################################
	def _output_block_layer(self, inputs):
		w = tf.Variable(tf.truncated_normal([3, 3, 3, 64, 1], stddev=0.05), name='end_con3d')
		output = tf.nn.conv3d(input=inputs,
                            filter=w,
			                strides=[1,1,1,1,1],
			                padding='SAME',
							name='output')
		print('output -----> ',output.get_shape())
		return output


	def _encoding_block_layer(self, inputs, filters, block_fn, name):
		inputs = block_fn(inputs, filters, name)
		skip_inputs_tensor = inputs
		print(name,' skip_inputs_tensor -----> ',skip_inputs_tensor.get_shape())
		inputs = Pool3d(inputs, name)
		return inputs, skip_inputs_tensor


	def _decoding_block_layer(self, inputs, skip_inputs, filters, block_fn, name):
		inputs = Deconv3D(inputs=inputs, filters=filters*2, name=name+'_deconv3d')
		# inputs = Deconv3D_keras(inputs=inputs)
		inputs = tf.concat([inputs ,skip_inputs], axis=4)
		print(name,'concat.get_shape -----> ',str(inputs.get_shape()))
		inputs = block_fn(inputs, filters, name)
		return inputs


	################################################################################
	# Basic blocks building the network
	################################################################################
	def _double_conv_block(self, inputs, filters, name):
		inputs = Conv3D(inputs=inputs,
					    filters=filters,
                        name = name+'_conv3d1')
		inputs = ReLU(inputs, name+'_ReLU1')
		inputs = Conv3D(inputs=inputs,
					    filters=filters,
                        name = name+'_conv3d2')
		inputs = ReLU(inputs, name+'_ReLU2')
		return inputs




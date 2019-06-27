#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7' 

import evaluate
import numpy as np
import progressbar
import shutil
import tempfile
import tensorflow as tf
import traceback

from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from six.moves import zip, range
from tensorflow.contrib.lite.python import tflite_convert
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from util.audio import audiofile_to_input_vector
from util.config import Config, initialize_globals
from util.coordinator import TrainingCoordinator
from util.feeding import DataSet, ModelFeeder
from util.flags import create_flags, FLAGS
from util.logging import log_info, log_error, log_debug, log_warn
from util.preprocess import preprocess
from util.text import Alphabet


# Graph Creation
# ==============

def variable_on_worker_level(name, shape, initializer,trainable=True):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_worker_level()``
    used to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    if len(FLAGS.ps_hosts) == 0:
        device = Config.worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=Config.worker_device, cluster=Config.cluster)

    with tf.device(device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer,trainable=trainable)
    return var

def batch_norm(inputs, is_training = True, layer_num = 0):
    idx = str(layer_num)
    scale = variable_on_worker_level('scale_' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=1.0, dtype=tf.float32))
    beta = variable_on_worker_level('beta' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=0.0, dtype=tf.float32))
    #batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
    moving_mean = variable_on_worker_level('movingmean' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=0.0, dtype=tf.float32), False)
    moving_var = variable_on_worker_level('movingvar' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=1.0, dtype=tf.float32), False)
   

    mean, var = tf.nn.moments(inputs,[0, 1, 2])
    #print (mean.get_shape())
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999)
    update_moving_var = moving_averages.assign_moving_average(moving_var, var, 0.999)
    tf.add_to_collection('lstm_update_ops', update_moving_mean)
    tf.add_to_collection('lstm_update_ops', update_moving_var)
    
    mean, var = control_flow_ops.cond(tf.convert_to_tensor(is_training, dtype=tf.bool, name = 'is_3d_training'),  lambda: (mean, var), lambda: (moving_mean, moving_var))

    return tf.nn.batch_normalization(inputs, mean, var, beta, scale, 0.001)

     
def batch_norm_2d(inputs, is_training = True, layer_num = 0):
    idx = str(layer_num)
    scale = variable_on_worker_level('scale_' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=1.0, dtype=tf.float32))
    beta = variable_on_worker_level('beta' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=0.0, dtype=tf.float32))
    #batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
    moving_mean = variable_on_worker_level('movingmean' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=0.0, dtype=tf.float32), False)
    moving_var = variable_on_worker_level('movingvar' + idx, [inputs.get_shape()[-1]], tf.constant_initializer(value=1.0, dtype=tf.float32), False)

    mean, var = tf.nn.moments(inputs, [0, 1])
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.999)
    update_moving_var = moving_averages.assign_moving_average(moving_var, var, 0.999)
    tf.add_to_collection('lstm_update_ops', update_moving_mean)
    tf.add_to_collection('lstm_update_ops', update_moving_var)

    mean, var = control_flow_ops.cond(tf.convert_to_tensor(is_training, dtype=tf.bool, name = 'is_2d_training'),  lambda: (mean, var), lambda: (moving_mean, moving_var))

    return tf.nn.batch_normalization(inputs, mean, var, beta, scale, 0.001)

def feed_forward(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forward2', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=tf.AUTO_REUSE)
    return ff


def linear(x, num_hiddens=None, reuse=False):
    if num_hiddens is None:
        num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope(tf.get_variable_scope()):
        linear_layer = tf.layers.dense(x, num_hiddens,reuse=tf.AUTO_REUSE)
    return linear_layer


def dropout(x, is_training, rate=0.2):
    return tf.layers.dropout(x, rate, training=tf.convert_to_tensor(is_training))


def residual(x_in, x_out, reuse=False):
    with tf.variable_scope('residual', reuse=reuse):
        res_con = x_in + x_out
    return res_con



def stacked_multihead_attention(x, num_blocks, num_heads, use_residual, is_training, reuse=False):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=tf.AUTO_REUSE):
                x = batch_norm_2d(x,is_training,i+3)
                x, attentions = multihead_attention(x, x, x, use_residual, is_training, num_heads=num_heads, reuse=reuse)
                x = feed_forward(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
            #if i % 5 == 0 and i != 0: 
                #x = tf.nn.pool(x,[1,2,2],[1,1,1],padding='valid')
    return x, attentions


def multihead_attention(queries, keys, values, use_residual, is_training, num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attention', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forward(Q_K_V_, num_units, reuse=reuse)

        if use_residual:
            output = residual(output, queries, reuse=reuse)

    return output, attentions


def scaled_dot_product_attention(queries, keys, values, model_size=None, reuse=False):
    if model_size is None:
        model_size = tf.to_float(queries.get_shape().as_list()[-1])

    with tf.variable_scope('scaled_dot_product_attention', reuse=reuse):
        keys_T = tf.transpose(keys, [0, 2, 1])
        Q_K = tf.matmul(queries, keys_T) / tf.sqrt(model_size)
        attentions = tf.nn.softmax(Q_K)
        scaled_dprod_att = tf.matmul(attentions, values)
    return scaled_dprod_att, attentions


def CNN(batch_x, dropout, is_training=True):
    
    batch_size = tf.shape(batch_x)[0]

    batch_x = tf.reshape(batch_x,[batch_size, -1, 143,1])
    batch_x_shape = tf.shape(batch_x)
    

    # 1st layer
    b1 = variable_on_worker_level('b1', [1], tf.zeros_initializer())
    h1 = variable_on_worker_level('h1', [3, 3, 1, 1], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.nn.relu(tf.nn.conv2d(batch_x, h1, strides = [1, 1, 1, 1], padding = 'SAME') + b1)
    layer_1 = batch_norm(layer_1, is_training, 1)
    layer_1 = tf.minimum(layer_1, 24)
    layer_1 = tf.nn.dropout(layer_1, 1-dropout[0])

    # 2nd layer
    b2 = variable_on_worker_level('b2', [1], tf.zeros_initializer())
    h2 = variable_on_worker_level('h2', [5, 11, 1, 1], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, h2, strides = [1, 1, 1, 1], padding = 'SAME') + b2)
    layer_2 = batch_norm(layer_2, is_training, 2)    
    layer_2 = tf.minimum(layer_2, 24)
    layer_2 = tf.nn.dropout(layer_2, 1-dropout[0])

    # 3rd layer
    b3 = variable_on_worker_level('b3', [1], tf.zeros_initializer())
    h3 = variable_on_worker_level('h3', [11, 41, 1, 1], tf.contrib.layers.xavier_initializer())
    layer_3 = tf.nn.relu(tf.nn.conv2d(layer_2, h3, strides = [1, 3, 2, 1], padding = 'SAME') + b3)
    layer_3 = batch_norm(layer_3, is_training, 3)    
    layer_3 = tf.minimum(layer_3, 24)
    layer_3 = tf.nn.dropout(layer_3, 1-dropout[0])

    # 4th layer
    b4 = variable_on_worker_level('b4', [1], tf.zeros_initializer())
    h4 = variable_on_worker_level('h4', [11, 41, 1, 1], tf.contrib.layers.xavier_initializer())
    layer_4 = tf.nn.relu(tf.nn.conv2d(layer_3, h4, strides = [1, 1, 2, 1], padding = 'SAME') + b4)
    layer_4 = batch_norm(layer_4, is_training, 4)    
    layer_4 = tf.minimum(layer_4, 24)
    layer_4 = tf.nn.dropout(layer_4, 1-dropout[0])


    layer_4 = tf.reshape(layer_4, (batch_x_shape[0], -1, 36 * 1))
    layer_4 = tf.transpose(layer_4, [1, 0, 2])
    layer_4 = tf.reshape(layer_4, (-1, 36 * 1))
    

    b5 = variable_on_worker_level('b5', [Config.n_hidden_6], tf.zeros_initializer())
    h5 = variable_on_worker_level('h5', [36*1, Config.n_hidden_6], tf.contrib.layers.xavier_initializer())
    layer_5 = tf.add(tf.matmul(layer_4, h5), b5)
    layer_5 = tf.reshape(layer_5, [-1, batch_x_shape[0], Config.n_hidden_6], name="logits")
    return layer_5
    # Output shape: [n_steps, batch_size, Config.n_hidden_6]

def SAN(batch_x, batch_size=None, n_steps=-1, is_training=True):
    
    #encoder
    
    if batch_size is None:
        batch_size = tf.shape(batch_x)[0]

    batch_x = tf.reshape(batch_x,[batch_size, n_steps, 143,1])
    batch_x_shape = tf.shape(batch_x)
    
    b1 = variable_on_worker_level('b1', [32], tf.zeros_initializer())
    h1 = variable_on_worker_level('h1', [11, 41, 1, 32], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.nn.relu(tf.nn.conv2d(batch_x, h1, strides = [1, 3, 2, 1], padding = 'SAME') + b1)
    layer_1 = batch_norm(layer_1, is_training, 1)
    layer_1 = tf.minimum(layer_1, 24)

    b2 = variable_on_worker_level('b2', [32], tf.zeros_initializer())
    h2 = variable_on_worker_level('h2', [7, 21, 32, 32], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, h2, strides = [1, 1, 2, 1], padding = 'SAME') + b2)
    layer_2 = batch_norm(layer_2, is_training, 2)
    layer_2 = tf.minimum(layer_2, 24)

    '''
    layer_2 = tf.reshape(layer_2, (batch_x_shape[0], -1, 36 * 32))
    layer_2 = tf.transpose(layer_2, [1, 0, 2])
    layer_2 = tf.reshape(layer_2, (-1, 36 * 32))

    b3 = variable_on_worker_level('b3', [1024], tf.zeros_initializer())
    h3 = variable_on_worker_level('h3', [36 * 32, 1024], tf.contrib.layers.xavier_initializer())
    layer_3 = tf.add(tf.matmul(layer_2, h3), b3)
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], 1024])
    #layer_3 = tf.minimum(layer_3, 24)
    layer_3 = batch_norm_2d(layer_3, is_training, 3)
    #print (layer_3.shape)

    gru_fw_cell_1 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    gru_bw_cell_1 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    rnn_outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell_1, cell_bw = gru_bw_cell_1, inputs = layer_3, dtype = tf.float32, time_major = True)
    rnn_outputs_1 = tf.concat(rnn_outputs_1, 2)
    #rnn_outputs_1 = tf.nn.relu(rnn_outputs_1)
    b4 = variable_on_worker_level('b4', [1024], tf.zeros_initializer())
    h4 = variable_on_worker_level('h4', [1024*2, 1024], tf.contrib.layers.xavier_initializer())
    rnn_outputs_1 = tf.reshape(rnn_outputs_1, (-1, 1024*2))
    layer_4 = tf.add(tf.matmul(rnn_outputs_1, h4), b4)
    layer_4 = tf.reshape(layer_4, [-1, batch_x_shape[0], 1024])    
    layer_4 = batch_norm_2d(layer_4, is_training, 4)


    gru_fw_cell_2 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    gru_bw_cell_2 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    rnn_outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell_2, cell_bw = gru_bw_cell_2, inputs = layer_4, dtype = tf.float32, time_major = True)
    rnn_outputs_2 = tf.concat(rnn_outputs_2, 2)
    #rnn_outputs_2 = tf.nn.relu(rnn_outputs_2)
    b5 = variable_on_worker_level('b5', [1024], tf.zeros_initializer())
    h5 = variable_on_worker_level('h5', [1024*2, 1024], tf.contrib.layers.xavier_initializer())
    rnn_outputs_2 = tf.reshape(rnn_outputs_2, (-1, 1024*2))
    layer_5 = tf.add(tf.matmul(rnn_outputs_2, h5), b5)
    layer_5 = tf.reshape(layer_5, [-1, batch_x_shape[0], 1024])    
    layer_5 = batch_norm_2d(layer_5, is_training, 5)
    '''

    # print(layer_2) [,,36,32]
    layer_2 = tf.reshape(layer_2, (batch_x_shape[0], -1, 36 * 32)) #[batch,n_steps,num_hiddens(15808)]
    #layer_6 = tf.reshape(layer_5, (batch_x_shape[0], -1, 1024)) #[batch,n_steps,num_hiddens(15808)]
    encoder_out, _ = stacked_multihead_attention(layer_2,
                                              num_blocks=5,
                                              num_heads=8,
                                              use_residual=True,
                                              is_training=True, #?
                                              reuse=True)
    
    encoder_out = tf.transpose(encoder_out,[1,0,2]) #[n_step, batch, 1152]
    #print(encoder_out)
    # [?,?,1152]

    '''
    embeddings = variable_on_worker_level('embedding', [2752,1024], tf.contrib.layers.xavier_initializer())
    sos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='SOS') * 2
    sos_step_embedded = tf.nn.embedding_lookup(embeddings, sos_time_slice)
    pad_step_embedded = tf.zeros([batch_size, 1152+1024],dtype=tf.float32)

    def initial_fn():
        initial_elements_finished = (0 >= tf.shape(encoder_out)[0])  # all False at the initial step
        #print(sos_step_embedded)
        initial_input = tf.concat((sos_step_embedded, encoder_out[0]), 1)
        # print(initial_input) 
        return initial_elements_finished, initial_input

    def sample_fn(time, outputs, state):
        # 选择logit最大的下标作为sample
        #print(outputs) #[?,1024]   
        prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
        #print(prediction_id) [?,]
        return prediction_id

    def next_inputs_fn(time, outputs, state, sample_ids):
        # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
        #print(sample_ids)
        pred_embedding = tf.nn.embedding_lookup(embeddings, sample_ids)
        # 输入是h_i+o_{i-1}+c_i
        #print(pred_embedding) [?,1024]
        
        next_input = tf.concat((pred_embedding, encoder_out[time - 1]), 1)
        #print(next_input) 
        elements_finished = (time >= tf.shape(encoder_out)[0])  # this operation produces boolean tensor of [batch_size]
        all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
        next_state = state
        return elements_finished, next_inputs, next_state

    # 自定义helper
    my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)


    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            memory = tf.transpose(encoder_out, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention( \
                num_units=64, memory=memory, \
                    memory_sequence_length=encoder_out.get_shape().as_list()[0])
            cell = tf.contrib.rnn.LSTMCell(num_units=1024)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=64)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, 2752, reuse=reuse)
            # 使用自定义helper的decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size))
            # 获取decode结果
            final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode( \
                decoder=decoder, output_time_major=True)

            return final_outputs, final_state

	
    outputs, _ = decode(my_helper, 'decode')
    rnn_output, _ = outputs

    #rnn_output = tf.reshape(rnn_output,[-1,1024])
    #b6 = variable_on_worker_level('b6', [2752], tf.zeros_initializer())
    #h6 = variable_on_worker_level('h6', [1024, 2752], tf.contrib.layers.xavier_initializer())
    #layer_6 = tf.add(tf.matmul(rnn_output, h6), b6)
    #layer_6 = tf.reshape(layer_6, [-1, batch_size, 2752], name="raw_logits")
    #rnn_output = tf.nn.softmax(rnn_output)

    return rnn_output
    '''

    encoder_out = tf.layers.dense(encoder_out, units=Config.n_hidden_6,reuse=tf.AUTO_REUSE)
    return encoder_out


def BiRNN(batch_x, dropout, is_training=True):
    r'''
    That done, we will define the learned variables, the weights and biases,
    within the method ``BiRNN()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables.
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    In particular, the first variable ``h1`` holds the learned weight matrix that
    converts an input vector of dimension ``n_input + 2*n_input*n_context``
    to a vector of dimension ``n_hidden_1``.
    Similarly, the second variable ``h2`` holds the weight matrix converting
    an input vector of dimension ``n_hidden_1`` to one of dimension ``n_hidden_2``.
    The variables ``h3``, ``h5``, and ``h6`` are similar.
    Likewise, the biases, ``b1``, ``b2``..., hold the biases for the various layers.
    '''
    
    batch_size = tf.shape(batch_x)[0]

    batch_x = tf.reshape(batch_x,[batch_size, -1, 143,1])
    batch_x_shape = tf.shape(batch_x)
    

    # 1st layer
    b1 = variable_on_worker_level('b1', [32], tf.zeros_initializer())
    h1 = variable_on_worker_level('h1', [11, 41, 1, 32], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.nn.relu(tf.nn.conv2d(batch_x, h1, strides = [1, 3, 2, 1], padding = 'SAME') + b1)
    layer_1 = batch_norm(layer_1, is_training, 1)
    #layer_1 = tf.nn.relu((tf.nn.conv2d(batch_x, h1, strides = [1, 3, 2, 1], padding = 'SAME') + b1))
    layer_1 = tf.minimum(layer_1, 24)
    #layer_1 = tf.nn.dropout(layer_1, 1-dropout[0])

    # 2nd layer
    b2 = variable_on_worker_level('b2', [32], tf.zeros_initializer())
    h2 = variable_on_worker_level('h2', [11, 41, 32, 32], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, h2, strides = [1, 1, 2, 1], padding = 'SAME') + b2)
    layer_2 = batch_norm(layer_2, is_training, 2)    
    #layer_2 = tf.nn.relu((tf.nn.conv2d(layer_1, h2, strides = [1, 1, 2, 1], padding = 'SAME') + b2))
    layer_2 = tf.minimum(layer_2, 24)
    #layer_2 = tf.nn.dropout(layer_2, 1-dropout[0])


    layer_2 = tf.reshape(layer_2, (batch_x_shape[0], -1, 36 * 32))  #?
    layer_2 = tf.transpose(layer_2, [1, 0, 2])
    layer_2 = tf.reshape(layer_2, (-1, 36 * 32)) #?
    
    # 3rd layer
    b3 = variable_on_worker_level('b3', [1024], tf.zeros_initializer())
    h3 = variable_on_worker_level('h3', [36 * 32, 1024], tf.contrib.layers.xavier_initializer())
    layer_3 = tf.add(tf.matmul(layer_2, h3), b3)
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], 1024])
    layer_3 = tf.minimum(layer_3, 24)
    #layer_3 = tf.contrib.layers.layer_norm(layer_3,scope='ln_3')
    #layer_3 = batch_norm_2d(layer_3, is_training, 3)
    #print (layer_3.shape)
    #layer_3 = tf.nn.dropout(layer_3, 1-dropout[0])

    gru_fw_cell_1 = tf.contrib.rnn.LSTFMBlockCell(1024, reuse = tf.AUTO_REUSE)
    gru_bw_cell_1 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    rnn_outputs_1, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell_1, cell_bw = gru_bw_cell_1, inputs = layer_3, dtype = tf.float32, time_major = True)
    rnn_outputs_1 = tf.concat(rnn_outputs_1, 2)
    #rnn_outputs_1 = tf.nn.relu(rnn_outputs_1)
    b4 = variable_on_worker_level('b4', [1024], tf.zeros_initializer())
    h4 = variable_on_worker_level('h4', [1024*2, 1024], tf.contrib.layers.xavier_initializer())
    rnn_outputs_1 = tf.reshape(rnn_outputs_1, (-1, 1024*2))
    layer_4 = tf.add(tf.matmul(rnn_outputs_1, h4), b4)
    layer_4 = tf.reshape(layer_4, [-1, batch_x_shape[0], 1024])    
    layer_4 = tf.contrib.layers.layer_norm(layer_4,scope='ln_4')
    #layer_4 = batch_norm_2d(layer_4, is_training, 4)
    #layer_4 = tf.nn.dropout(layer_4, 0.9)

    gru_fw_cell_2 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    gru_bw_cell_2 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    rnn_outputs_2, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell_2, cell_bw = gru_bw_cell_2, inputs = layer_4, dtype = tf.float32, time_major = True)
    rnn_outputs_2 = tf.concat(rnn_outputs_2, 2)
    #rnn_outputs_2 = tf.nn.relu(rnn_outputs_2)
    b5 = variable_on_worker_level('b5', [1024], tf.zeros_initializer())
    h5 = variable_on_worker_level('h5', [1024*2, 1024], tf.contrib.layers.xavier_initializer())
    rnn_outputs_2 = tf.reshape(rnn_outputs_2, (-1, 1024*2))
    layer_5 = tf.add(tf.matmul(rnn_outputs_2, h5), b5)
    layer_5 = tf.reshape(layer_5, [-1, batch_x_shape[0], 1024])
    layer_5 = tf.contrib.layers.layer_norm(layer_5,scope='ln_5')  
    #layer_5 = batch_norm_2d(layer_5, is_training, 5)
    #layer_5 = tf.nn.dropout(layer_5, 0.9)

    gru_fw_cell_3 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    gru_bw_cell_3 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    rnn_outputs_3, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell_3, cell_bw = gru_bw_cell_3, inputs = layer_5, dtype = tf.float32, time_major = True)
    rnn_outputs_3 = tf.concat(rnn_outputs_3, 2)
    #rnn_outputs_2 = tf.nn.relu(rnn_outputs_2)
    b6 = variable_on_worker_level('b6', [1024], tf.zeros_initializer())
    h6 = variable_on_worker_level('h6', [1024*2, 1024], tf.contrib.layers.xavier_initializer())
    rnn_outputs_3 = tf.reshape(rnn_outputs_3, (-1, 1024*2))
    layer_6 = tf.add(tf.matmul(rnn_outputs_3, h6), b6)
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], 1024])
    layer_6 = tf.contrib.layers.layer_norm(layer_6,scope='ln_6')  
    #layer_5 = batch_norm_2d(layer_5, is_training, 5)
    #layer_5 = tf.nn.dropout(layer_5, 0.9)

    gru_fw_cell_4 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    gru_bw_cell_4 = tf.contrib.rnn.LSTMBlockCell(1024, reuse = tf.AUTO_REUSE)
    rnn_outputs_4, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = gru_fw_cell_4, cell_bw = gru_bw_cell_4, inputs = layer_6, dtype = tf.float32, time_major = True)
    rnn_outputs_4 = tf.concat(rnn_outputs_4, 2)
    b7 = variable_on_worker_level('b7', [Config.n_hidden_6], tf.zeros_initializer())
    h7 = variable_on_worker_level('h7', [1024*2, Config.n_hidden_6], tf.contrib.layers.xavier_initializer())
    rnn_outputs_4 = tf.reshape(rnn_outputs_4, (-1, 1024*2))
    layer_7 = tf.add(tf.matmul(rnn_outputs_4, h7), b7)
    layer_7 = tf.reshape(layer_7, [-1, batch_x_shape[0], Config.n_hidden_6], name="logits")
    # Output shape: [n_steps, batch_size, Config.n_hidden_6]
    return layer_7

    '''
    san_input = tf.reshape(layer_5, (batch_x_shape[0], -1, 1024)) #[batch,n_steps,num_hiddens(15808)]

    encoder_out, _ = stacked_multihead_attention(san_input,
                                              num_blocks=4,
                                              num_heads=8,
                                              use_residual=True,
                                              is_training=True,
                                              reuse=True)
    
    encoder_out = tf.transpose(encoder_out,[1,0,2])
    encoder_out = tf.layers.dense(encoder_out, units=Config.Config.,reuse=tf.AUTO_REUSE)
    return encoder_out
    '''

# Accuracy and Loss
# =================

# In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.

def calculate_mean_edit_distance_and_loss(model_feeder, tower, dropout, reuse):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y = model_feeder.next_batch(tower)

    # Calculate the logits of the batch using
    logits = CNN(batch_x,dropout)
    
    
    #print(logits)
    #print(batch_y)
    # Compute the CTC loss using TensorFlow's `ctc_loss`
    total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=tf.to_int32(tf.ceil(batch_seq_len/3)),ignore_longer_outputs_than_inputs=True)
    #total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)
    #print(total_loss)
    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Finally we return the average loss
    return avg_loss


# Adam Optimization
# =================

# In contrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                       beta1=FLAGS.beta1,
                                       beta2=FLAGS.beta2,
                                       epsilon=FLAGS.epsilon)
    return optimizer


# Towers
# ======

# In order to properly make use of multiple GPU's, one must introduce new abstractions,
# not present when using a single GPU, that facilitate the multi-GPU use case.
# In particular, one must introduce a means to isolate the inference and gradient
# calculations on the various GPU's.
# The abstraction we intoduce for this purpose is called a 'tower'.
# A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`,
# is a means to isolate the operations within a tower.
# For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`.

def get_tower_results(model_feeder, optimizer, dropout_rates):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate and return the optimization gradients
    and the average loss across towers.
    '''
    # To calculate the mean of the losses
    tower_avg_losses = []

    # Tower gradients to return
    tower_gradients = []

    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(Config.available_devices)):
            # Execute operations of tower i on device i
            if len(FLAGS.ps_hosts) == 0:
                device = Config.available_devices[i]
            else:
                device = tf.train.replica_device_setter(worker_device=Config.available_devices[i], cluster=Config.cluster)
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    # Calculate the avg_loss and mean_edit_distance and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    avg_loss = calculate_mean_edit_distance_and_loss(model_feeder, i, dropout_rates, reuse=i>0)

                    # Allow for variables to be re-used by the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)


    avg_loss_across_towers = tf.reduce_mean(tower_avg_losses, 0)

    tf.summary.scalar(name='step_loss', tensor=avg_loss_across_towers, collections=['step_summaries'])

    # Return gradients and the average loss
    return tower_gradients, avg_loss_across_towers


def average_gradients(tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a synchronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(Config.cpu_device):
        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []

            # Loop over the gradients for the current variable
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                #print(g)
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads



# Logging
# =======

def log_variable(variable, gradient=None):
    r'''
    We introduce a function for logging a tensor variable's current state.
    It logs scalar values for the mean, standard deviation, minimum and maximum.
    Furthermore it logs a histogram of its state and (if given) of an optimization gradient.
    '''
    name = variable.name
    mean = tf.reduce_mean(variable)
    tf.summary.scalar(name='%s/mean'   % name, tensor=mean)
    tf.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(tf.square(variable - mean))))
    tf.summary.scalar(name='%s/max'    % name, tensor=tf.reduce_max(variable))
    tf.summary.scalar(name='%s/min'    % name, tensor=tf.reduce_min(variable))
    tf.summary.histogram(name=name, values=variable)
    if gradient is not None:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is not None:
            tf.summary.histogram(name='%s/gradients' % name, values=grad_values)


def log_grads_and_vars(grads_and_vars):
    r'''
    Let's also introduce a helper function for logging collections of gradient/variable tuples.
    '''
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)


# Helpers
# =======

def send_token_to_ps(session, kill=False):
    # Sending our token (the task_index as a debug opportunity) to each parameter server.
    # kill switch tokens are negative and decremented by 1 to deal with task_index 0
    token = -FLAGS.task_index-1 if kill else FLAGS.task_index
    kind = 'kill switch' if kill else 'stop'
    for index, enqueue in enumerate(Config.done_enqueues):
        log_debug('Sending %s token to ps %d...' % (kind, index))
        session.run(enqueue, feed_dict={ Config.token_placeholder: token })
        log_debug('Sent %s token to ps %d.' % (kind, index))


def train(server=None):
    r'''
    Trains the network on a given server of a cluster.
    If no server provided, it performs single process training.
    '''

    # Initializing and starting the training coordinator
    coord = TrainingCoordinator(Config.is_chief)
    coord.start()

    # Create a variable to hold the global_step.
    # It will automagically get incremented by the optimizer.
    global_step = tf.Variable(0, trainable=False, name='global_step')

    dropout_rates = [tf.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]

    # Reading training set
    train_data = preprocess(FLAGS.train_files.split(','),
                            FLAGS.train_batch_size,
                            Config.n_input,
                            Config.n_context,
                            Config.alphabet,
                            hdf5_cache_path=FLAGS.train_cached_features_path)

    train_set = DataSet(train_data,
                        FLAGS.train_batch_size,
                        limit=FLAGS.limit_train,
                        next_index=lambda i: coord.get_next_index('train'))

    # Reading validation set
    dev_data = preprocess(FLAGS.dev_files.split(','),
                          FLAGS.dev_batch_size,
                          Config.n_input,
                          Config.n_context,
                          Config.alphabet,
                          hdf5_cache_path=FLAGS.dev_cached_features_path)

    dev_set = DataSet(dev_data,
                      FLAGS.dev_batch_size,
                      limit=FLAGS.limit_dev,
                      next_index=lambda i: coord.get_next_index('dev'))

    # Combining all sets to a multi set model feeder
    model_feeder = ModelFeeder(train_set,
                               dev_set,
                               Config.n_input,
                               Config.n_context,
                               Config.alphabet,
                               tower_feeder_count=len(Config.available_devices))

    # Create the optimizer
    optimizer = create_optimizer()

    # Synchronous distributed training is facilitated by a special proxy-optimizer
    if not server is None:
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                   replicas_to_aggregate=FLAGS.replicas_to_agg,
                                                   total_num_replicas=FLAGS.replicas)

    # Get the data_set specific graph end-points
    gradients, loss = get_tower_results(model_feeder, optimizer, dropout_rates)
    

    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)
    #print(avg_tower_gradients)

    # Add summaries of all variables and gradients to log
    log_grads_and_vars(avg_tower_gradients)

    # Op to merge all summaries for the summary hook
    merge_all_summaries_op = tf.summary.merge_all()

    # These are saved on every step
    step_summaries_op = tf.summary.merge_all('step_summaries')

    step_summary_writers = {
        'train': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120),
        'dev': tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120)
    }

    # Apply gradients to modify the model
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)


    if FLAGS.early_stop is True and not FLAGS.validation_step > 0:
        log_warn('Parameter --validation_step needs to be >0 for early stopping to work')

    class CoordHook(tf.train.SessionRunHook):
        r'''
        Embedded coordination hook-class that will use variables of the
        surrounding Python context.
        '''
        def after_create_session(self, session, coord):
            log_debug('Starting queue runners...')
            model_feeder.start_queue_threads(session, coord)
            log_debug('Queue runners started.')

        def end(self, session):
            # Closing the data_set queues
            log_debug('Closing queues...')
            model_feeder.close_queues(session)
            log_debug('Queues closed.')

            # Telling the ps that we are done
            send_token_to_ps(session)

    # Collecting the hooks
    hooks = [CoordHook()]

    # Hook to handle initialization and queues for sync replicas.
    if not server is None:
        hooks.append(optimizer.make_session_run_hook(Config.is_chief))

    # Hook to save TensorBoard summaries
    if FLAGS.summary_secs > 0:
        hooks.append(tf.train.SummarySaverHook(save_secs=FLAGS.summary_secs, output_dir=FLAGS.summary_dir, summary_op=merge_all_summaries_op))

    # Hook wih number of checkpoint files to save in checkpoint_dir
    if FLAGS.train and FLAGS.max_to_keep > 0:
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.checkpoint_dir, save_secs=FLAGS.checkpoint_secs, saver=saver))

    no_dropout_feed_dict = {
        dropout_rates[0]: 0.,
        dropout_rates[1]: 0.,
        dropout_rates[2]: 0.,
        dropout_rates[3]: 0.,
        dropout_rates[4]: 0.,
        dropout_rates[5]: 0.,
    }

    # Progress Bar
    def update_progressbar(set_name):
        if not hasattr(update_progressbar, 'current_set_name'):
            update_progressbar.current_set_name = None

        if (update_progressbar.current_set_name != set_name or
            update_progressbar.current_job_index == update_progressbar.total_jobs):

            # finish prev pbar if it exists
            if hasattr(update_progressbar, 'pbar') and update_progressbar.pbar:
                update_progressbar.pbar.finish()

            update_progressbar.total_jobs = None
            update_progressbar.current_job_index = 0

            current_epoch = coord._epoch-1

            if set_name == "train":
                log_info('Training epoch %i...' % current_epoch)
                update_progressbar.total_jobs = coord._num_jobs_train
            else:
                log_info('Validating epoch %i...' % current_epoch)
                update_progressbar.total_jobs = coord._num_jobs_dev

            # recreate pbar
            update_progressbar.pbar = progressbar.ProgressBar(max_value=update_progressbar.total_jobs,
                                                              redirect_stdout=True).start()

            update_progressbar.current_set_name = set_name

        if update_progressbar.pbar:
            update_progressbar.pbar.update(update_progressbar.current_job_index+1, force=True)

        update_progressbar.current_job_index += 1

    # Initialize update_progressbar()'s child fields to safe values
    update_progressbar.pbar = None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    try:
        with tf.train.MonitoredTrainingSession(master='' if server is None else server.target,
                                               is_chief=Config.is_chief,
                                               hooks=hooks,
                                               checkpoint_dir=FLAGS.checkpoint_dir,
                                               save_checkpoint_secs=None, # already taken care of by a hook
                                               log_step_count_steps=0, # disable logging of steps/s to avoid TF warning in validation sets
                                               config=Config.session_config) as session:
            tf.get_default_graph().finalize()

            try:
                if Config.is_chief:
                    # Retrieving global_step from the (potentially restored) model
                    model_feeder.set_data_set(no_dropout_feed_dict, model_feeder.train)
                    step = session.run(global_step, feed_dict=no_dropout_feed_dict)
                    coord.start_coordination(model_feeder, step)

                # Get the first job
                job = coord.get_job()

                while job and not session.should_stop():
                    log_debug('Computing %s...' % job)

                    is_train = job.set_name == 'train'

                    # The feed_dict (mainly for switching between queues)
                    if is_train:
                        feed_dict = {
                            dropout_rates[0]: FLAGS.dropout_rate,
                            dropout_rates[1]: FLAGS.dropout_rate2,
                            dropout_rates[2]: FLAGS.dropout_rate3,
                            dropout_rates[3]: FLAGS.dropout_rate4,
                            dropout_rates[4]: FLAGS.dropout_rate5,
                            dropout_rates[5]: FLAGS.dropout_rate6,
                        }
                    else:
                        feed_dict = no_dropout_feed_dict

                    # Sets the current data_set for the respective placeholder in feed_dict
                    model_feeder.set_data_set(feed_dict, getattr(model_feeder, job.set_name))

                    # Initialize loss aggregator
                    total_loss = 0.0

                    # Setting the training operation in case of training requested
                    train_op = apply_gradient_op if is_train else []

                    # So far the only extra parameter is the feed_dict
                    extra_params = { 'feed_dict': feed_dict }

                    step_summary_writer = step_summary_writers.get(job.set_name)

                    # Loop over the batches
                    for job_step in range(job.steps):
                        if session.should_stop():
                            break

                        log_debug('Starting batch...')
                        # Compute the batch
                        _, current_step, batch_loss, step_summary = session.run([train_op, global_step, loss, step_summaries_op], **extra_params)

                        # Log step summaries
                        step_summary_writer.add_summary(step_summary, current_step)

                        # Uncomment the next line for debugging race conditions / distributed TF
                        log_debug('Finished batch step %d.' % current_step)

                        # Add batch to loss
                        total_loss += batch_loss

                    # Gathering job results
                    job.loss = total_loss / job.steps

                    # Display progressbar
                    if FLAGS.show_progressbar:
                        update_progressbar(job.set_name)

                    # Send the current job to coordinator and receive the next one
                    log_debug('Sending %s...' % job)
                    job = coord.next_job(job)

                if update_progressbar.pbar:
                    update_progressbar.pbar.finish()

            except Exception as e:
                log_error(str(e))
                traceback.print_exc()
                # Calling all hook's end() methods to end blocking calls
                for hook in hooks:
                    hook.end(session)
                # Only chief has a SyncReplicasOptimizer queue runner that needs to be stopped for unblocking process exit.
                # A rather graceful way to do this is by stopping the ps.
                # Only one party can send it w/o failing.
                if Config.is_chief:
                    send_token_to_ps(session, kill=True)
                sys.exit(1)

        log_debug('Session closed.')

    except tf.errors.InvalidArgumentError as e:
        log_error(str(e))
        log_error('The checkpoint in {0} does not match the shapes of the model.'
                  ' Did you change alphabet.txt or the --n_hidden parameter'
                  ' between train runs using the same checkpoint dir? Try moving'
                  ' or removing the contents of {0}.'.format(FLAGS.checkpoint_dir))
        sys.exit(1)

    # Stopping the coordinator
    coord.stop()


def test():
    # Reading test set
    test_data = preprocess(FLAGS.test_files.split(','),
                           FLAGS.test_batch_size,
                           Config.n_input,
                           Config.n_context,
                           Config.alphabet,
                           hdf5_cache_path=FLAGS.test_cached_features_path)

    graph = create_inference_graph(batch_size=FLAGS.test_batch_size)
    evaluate.evaluate(test_data, graph)

'''
def create_inference_graph(batch_size=1, n_steps=16, tflite=False):
    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    input_tensor = tf.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2*Config.n_context+1, Config.n_input], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')

    if not tflite:
        previous_state_c = variable_on_worker_level('previous_state_c', [batch_size, 2*Config.n_cell_dim], initializer=None)
        previous_state_h = variable_on_worker_level('previous_state_h', [batch_size, 2*Config.n_cell_dim], initializer=None)
    else:
        previous_state_c = tf.placeholder(tf.float32, [batch_size, 2*Config.n_cell_dim], name='previous_state_c')
        previous_state_h = tf.placeholder(tf.float32, [batch_size, 2*Config.n_cell_dim], name='previous_state_h')

    previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

    no_dropout = [0.0] * 6

    #logits, layers = BiRNN(batch_x=input_tensor,
                           #seq_length=seq_length if FLAGS.use_seq_length else None,
                           #dropout=no_dropout,
                           #batch_size=batch_size,
                           #n_steps=n_steps,
                           #previous_state=previous_state,
                           #tflite=tflite)

    logits, layers = SAN(batch_x=input_tensor,
                           seq_length=seq_length if FLAGS.use_seq_length else None,
                           dropout=no_dropout)
    

    # TF Lite runtime will check that input dimensions are 1, 2 or 4
    # by default we get 3, the middle one being batch_size which is forced to
    # one on inference graph, so remove that dimension
    if tflite:
        logits = tf.squeeze(logits, [1])

    # Apply softmax for CTC decoder
    logits = tf.nn.softmax(logits)

    new_state_c, new_state_h = layers['rnn_output_state']
    

    # Initial zero state
    if not tflite:
        zero_state = tf.zeros([batch_size, 2*Config.n_cell_dim], tf.float32)
        initialize_c = tf.assign(previous_state_c, zero_state)
        initialize_h = tf.assign(previous_state_h, zero_state)
        initialize_state = tf.group(initialize_c, initialize_h, name='initialize_state')
        with tf.control_dependencies([tf.assign(previous_state_c, new_state_c), tf.assign(previous_state_h, new_state_h)]):
            logits = tf.identity(logits, name='logits')

        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': logits,
                'initialize_state': initialize_state,
            },
            layers
        )
    else:
        logits = tf.identity(logits, name='logits')
        new_state_c = tf.identity(new_state_c, name='new_state_c')
        new_state_h = tf.identity(new_state_h, name='new_state_h')

        return (
            {
                'input': input_tensor,
                'previous_state_c': previous_state_c,
                'previous_state_h': previous_state_h,
            },
            {
                'outputs': logits,
                'new_state_c': new_state_c,
                'new_state_h': new_state_h,
            },
            layers
        )
'''

def create_inference_graph(batch_size=None,n_steps=-1):
    # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
    input_tensor = tf.placeholder(tf.float32, [batch_size, n_steps if n_steps > 0 else None, 2*Config.n_context+1, Config.n_input], name='input_node')
    seq_length = tf.placeholder(tf.int32, [batch_size], name='input_lengths')
    is_training = tf.placeholder(tf.bool, name = 'training_phase')
    dropout = [0.0] * 6
    # Calculate the logits of the batch using SAN
    logits = CNN(batch_x=input_tensor,dropout=dropout, is_training=is_training)
    #logits = BiRNN(input_tensor)
    #print (logits.shape)
    
    

    return (
        {
            'input': input_tensor,
            'input_lengths': seq_length,
            'training_phase': is_training,
            
        },
        {
            'outputs': logits,
        }
    )



def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    with tf.device('/cpu:0'):
        from tensorflow.python.framework.ops import Tensor, Operation

        tf.reset_default_graph()
        session = tf.Session(config=Config.session_config)
        inputs, outputs = create_inference_graph(batch_size=1)
        input_names = ",".join(tensor.op.name for tensor in inputs.values())
        output_names_tensors = [ tensor.op.name for tensor in outputs.values() if isinstance(tensor, Tensor) ]
        output_names_ops = [ tensor.name for tensor in outputs.values() if isinstance(tensor, Operation) ]
        output_names = ",".join(output_names_tensors + output_names_ops)
        #input_shapes = ":".join(",".join(map(str, tensor.shape)) for tensor in inputs.values())

        if not FLAGS.export_tflite:
            mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        else:
            # Create a saver using variables from the above newly created graph
            def fixup(name):
                if name.startswith('rnn/lstm_cell/'):
                    return name.replace('rnn/lstm_cell/', 'lstm_fused_cell/')
                return name

            mapping = {fixup(v.op.name): v for v in tf.global_variables()}

        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        checkpoint_path = checkpoint.model_checkpoint_path

        output_filename = 'output_graph.pb'
        if FLAGS.remove_export:
            if os.path.isdir(FLAGS.export_dir):
                log_info('Removing old export')
                shutil.rmtree(FLAGS.export_dir)
        try:
            output_graph_path = os.path.join(FLAGS.export_dir, output_filename)

            if not os.path.isdir(FLAGS.export_dir):
                os.makedirs(FLAGS.export_dir)

            def do_graph_freeze(output_file=None, output_node_names=None, variables_blacklist=None):
                freeze_graph.freeze_graph_with_def_protos(
                    input_graph_def=session.graph_def,
                    input_saver_def=saver.as_saver_def(),
                    input_checkpoint=checkpoint_path,
                    output_node_names=output_node_names,
                    restore_op_name=None,
                    filename_tensor_name=None,
                    output_graph=output_file,
                    clear_devices=False,
                    variable_names_blacklist=variables_blacklist,
                    initializer_nodes='')

            if not FLAGS.export_tflite:
                do_graph_freeze(output_file=output_graph_path, output_node_names=output_names, variables_blacklist='previous_state_c,previous_state_h')
            else:
                temp_fd, temp_freeze = tempfile.mkstemp(dir=FLAGS.export_dir)
                os.close(temp_fd)
                do_graph_freeze(output_file=temp_freeze, output_node_names=output_names, variables_blacklist='')
                output_tflite_path = os.path.join(FLAGS.export_dir, output_filename.replace('.pb', '.tflite'))
                class TFLiteFlags():
                    def __init__(self):
                        self.graph_def_file = temp_freeze
                        self.inference_type = 'FLOAT'
                        self.input_arrays   = input_names
                        #self.input_shapes   = input_shapes
                        self.output_arrays  = output_names
                        self.output_file    = output_tflite_path
                        self.output_format  = 'TFLITE'

                        default_empty = [
                            'inference_input_type',
                            'mean_values',
                            'default_ranges_min', 'default_ranges_max',
                            'drop_control_dependency',
                            'reorder_across_fake_quant',
                            'change_concat_input_ranges',
                            'allow_custom_ops',
                            'converter_mode',
                            'post_training_quantize',
                            'dump_graphviz_dir',
                            'dump_graphviz_video'
                        ]
                        for e in default_empty:
                            self.__dict__[e] = None

                flags = TFLiteFlags()
                tflite_convert._convert_model(flags)
                os.unlink(temp_freeze)
                log_info('Exported model for TF Lite engine as {}'.format(os.path.basename(output_tflite_path)))

            log_info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError as e:
            log_error(str(e))

def do_single_file_inference(input_file_path):
    with tf.Session(config=Config.session_config) as session:
        #inputs, outputs = create_inference_graph(batch_size=1, n_steps=-1)
        inputs, outputs = create_inference_graph(batch_size=1)
        # Create a saver using variables from the above newly created graph
        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        saver = tf.train.Saver(mapping)

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counteract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if not checkpoint:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)

        #session.run(outputs['initialize_state'])

        features = audiofile_to_input_vector(input_file_path, Config.n_input, Config.n_context)
        num_strides = len(features) - (Config.n_context * 2)

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2*Config.n_context+1
        features = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, Config.n_input),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        logits = session.run(outputs['outputs'], feed_dict = {
            inputs['input']: [features],
            inputs['input_lengths']: [num_strides],
            inputs['training_phase']: False,
        })

        #logits = np.squeeze(logits)
        #print(np.shape(logits))
        #np.save('logits.npy', logits)
        #logits = np.exp(logits) / np.expand_dims(np.sum(np.exp(logits),axis=1),axis=1)
        #scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        #FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                        #Config.alphabet)
        #decoded = ctc_beam_search_decoder(logits, Config.alphabet, FLAGS.beam_width, scorer=scorer)
        
        index = list(np.argsort(logits)[0,:,-1])
        words =  ' '.join([Config.alphabet.string_from_label(a) for a in index])
        print(words)        
        # Print highest probability result
        #print(decoded[0][1])


def main(_):
    initialize_globals()

    if FLAGS.train or FLAGS.test:
        if len(FLAGS.worker_hosts) == 0:
            # Only one local task: this process (default case - no cluster)
            with tf.Graph().as_default():
                tf.set_random_seed(FLAGS.random_seed)
                train()
            # Now do a final test epoch
            if FLAGS.test:
                with tf.Graph().as_default():
                    test()
            log_debug('Done.')
        else:
            # Create and start a server for the local task.
            server = tf.train.Server(Config.cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
            if FLAGS.job_name == 'ps':
                # We are a parameter server and therefore we just wait for all workers to finish
                # by waiting for their stop tokens.
                with tf.Session(server.target) as session:
                    for worker in FLAGS.worker_hosts:
                        log_debug('Waiting for stop token...')
                        token = session.run(Config.done_dequeues[FLAGS.task_index])
                        if token < 0:
                            log_debug('Got a kill switch token from worker %i.' % abs(token + 1))
                            break
                        log_debug('Got a stop token from worker %i.' % token)
                log_debug('Session closed.')

                if FLAGS.test:
                    test()
            elif FLAGS.job_name == 'worker':
                # We are a worker and therefore we have to do some work.
                # Assigns ops to the local worker by default.
                with tf.device(tf.train.replica_device_setter(
                               worker_device=Config.worker_device,
                               cluster=Config.cluster)):

                    # Do the training
                    train(server)

            log_debug('Server stopped.')

    # Are we the main process?
    if Config.is_chief:
        # Doing solo/post-processing work just on the main process...
        # Exporting the model
        if FLAGS.export_dir:
            export()

    if len(FLAGS.one_shot_infer):
        do_single_file_inference(FLAGS.one_shot_infer)

if __name__ == '__main__' :
    create_flags()
    tf.app.run(main)



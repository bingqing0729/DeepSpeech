import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
def variable_on_worker_level(name, shape, initializer):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_worker_level()``
    used to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


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
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
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

def SAN(batch_x, seq_length, dropout, reuse=False, batch_size=None, n_steps=-1, previous_state=None, tflite=False):
    
    #encoder
    layers = {}
    if not batch_size:
        batch_size = tf.shape(batch_x)[0]

    batch_x = tf.reshape(batch_x,[batch_size, n_steps, 143,1])
    batch_x_shape = tf.shape(batch_x)
    
    b1 = variable_on_worker_level('b1', [32], tf.zeros_initializer())
    h1 = variable_on_worker_level('h1', [11, 41, 1, 32], tf.contrib.layers.xavier_initializer())
    layer_1 = tf.nn.relu(tf.nn.conv2d(batch_x, h1, strides = [1, 3, 2, 1], padding = 'SAME') + b1)
    #layer_1 = batch_norm(layer_1, is_training, 1)
    layer_1 = tf.minimum(layer_1, 24)

    b2 = variable_on_worker_level('b2', [32], tf.zeros_initializer())
    h2 = variable_on_worker_level('h2', [3, 21, 32, 32], tf.contrib.layers.xavier_initializer())
    layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, h2, strides = [1, 1, 2, 1], padding = 'SAME') + b2)
    #layer_2 = batch_norm(layer_2, is_training, 2)
    layer_2 = tf.minimum(layer_2, 24)


    layer_2 = tf.reshape(layer_2, (batch_x_shape[0], -1, 36 * 32)) #[batch,n_steps,num_hiddens(15808)]

    encoder_out, _ = stacked_multihead_attention(layer_2,
                                              num_blocks=5,
                                              num_heads=4,
                                              use_residual=True,
                                              is_training=True,
                                              reuse=True)
    
    encoder_out = tf.transpose(encoder_out,[1,0,2]) #[n_step, batch, 15808]
    #print(encoder_out)
    # [?,?,15808]

    embeddings = variable_on_worker_level('embedding', [2752,128], tf.contrib.layers.xavier_initializer())
    sos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='SOS') * 2
    sos_step_embedded = tf.nn.embedding_lookup(embeddings, sos_time_slice)
    pad_step_embedded = tf.zeros([batch_size, 1152+128],dtype=tf.float32)

    def initial_fn():
        initial_elements_finished = (0 >= tf.shape(encoder_out)[0])  # all False at the initial step
        #print(sos_step_embedded)
        initial_input = tf.concat((sos_step_embedded, encoder_out[0]), 1)
        # print(initial_input) 
        return initial_elements_finished, initial_input

    def sample_fn(time, outputs, state):
        # 选择logit最大的下标作为sample
        #print(outputs) #[?,128]   
        prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
        #print(prediction_id) [?,]
        return prediction_id

    def next_inputs_fn(time, outputs, state, sample_ids):
        # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
        #print(sample_ids)
        pred_embedding = tf.nn.embedding_lookup(embeddings, sample_ids)
        # 输入是h_i+o_{i-1}+c_i
        #print(pred_embedding) [?,128]
        
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
                num_units=128, memory=memory, \
                    memory_sequence_length=encoder_out.get_shape().as_list()[0])
            cell = tf.contrib.rnn.LSTMCell(num_units=256)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=128)
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

	
    outputs, output_final_state = decode(my_helper, 'decode')
    rnn_output, _ = outputs
    layers['rnn_output_state'] = output_final_state

    #rnn_output = tf.reshape(rnn_output,[-1,128])
    #b6 = variable_on_worker_level('b6', [2752], tf.zeros_initializer())
    #h6 = variable_on_worker_level('h6', [128, 2752], tf.contrib.layers.xavier_initializer())
    #layer_6 = tf.add(tf.matmul(rnn_output, h6), b6)
    #layer_6 = tf.reshape(layer_6, [-1, batch_size, 2752], name="raw_logits")

    return rnn_output


def main():
    batch_x = tf.ones([10,2000,143])
    out = SAN(batch_x, 1, 1) 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #batch_x = tf.ones([10,200,494])
        result = sess.run(out)
        print(result)
        print(np.shape(result))


if __name__ == '__main__' :
    main()

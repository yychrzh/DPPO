from rpm import rpm
import tensorflow as tf
import numpy as np

HIDDEN_UNITS_1 = 64
HIDDEN_UNITS_2 = 64


class Fit_model(object):
    def __init__(self, input_dims, output_dims, epochs, batch_size, learning_rate):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.l_r = learning_rate

        self.sess = tf.Session()
        self.hidden_units_1 = HIDDEN_UNITS_1
        self.hidden_units_2 = HIDDEN_UNITS_2

    def create_dense_predict_network(self, name, input, trainable=True):
        with tf.variable_scope(name):
            # built value network
            l1 = tf.layers.dense(input, self.hidden_units_1, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, self.hidden_units_2, tf.nn.relu, trainable=trainable)
            output = tf.layers.dense(l2, self.output_dims, tf.nn.tanh, trainable=trainable)
        return output

    def train_step_generate(self):
        sess = self.sess

        input = tf.placeholder(tf.float32, [None, self.input_dims], 'input')
        label = tf.placeholder(tf.float32, [None, self.output_dims], 'label')

        with tf.variable_scope('predict'):
            output = self.create_dense_predict_network('predict_nn', input, trainable=True)
            loss_function = label - output
            loss = tf.reduce_mean(tf.square(loss_function))
            l_r = tf.Variable(self.l_r, name='l_r')
            train_op = tf.train.AdamOptimizer(l_r).minimize(loss)

        def update(data):
            [input_data, label_data] = data
            res = sess.run([loss, train_op], feed_dict={input: input_data, label: label_data})
            return res[0]

        def output_predict(input_d):
            s = np.array(input_d)
            if s.ndim < 2:
                s = s[np.newaxis, :]
            return sess.run(output, {input: s})[0, 0]


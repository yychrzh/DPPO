"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.
The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.3
gym 0.9.2

s: state
a: action
r: reward

"""

"""
12/17/2017
a single threading version revised by rzh
"""

import time
import tensorflow as tf
import numpy as np

class PPO(object):
    def __init__(self, state_space, action_space, max_episode_num, episode_lens, discount_factor=0.95,
                 actor_learning_rate=1e-3, critic_learning_rate=1e-3, mini_batch_size=128, epsilon=0.2):

        self.state_space = state_space
        self.action_space = action_space
        self.max_episode_num = max_episode_num
        self.episode_lens = episode_lens
        self.discount_factor = discount_factor
        self.a_lr = actor_learning_rate
        self.c_lr = critic_learning_rate
        self.batch_size = mini_batch_size
        self.epsilon = epsilon

        self.sess = tf.Session()

        self.update, self.choose_action, self.get_value = self.train_step_generate()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=1)

        # self.load_weights('100301')

    # create action network
    def create_actor_network(self, name, state, action_space, trainable=True):
        with tf.variable_scope(name):
            # two hidden layer
            l1 = tf.layers.dense(state, 256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l2, action_space, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, action_space, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # create value network
    def create_critic_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            # built value network
            l1 = tf.layers.dense(state, 256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu, trainable=trainable)
            value = tf.layers.dense(l2, 1)
        return value

    def train_step_generate(self):
        sess = self.sess

        # state
        state = tf.placeholder(tf.float32, [None, self.state_space], 'state')
        # discounted_return
        discounted_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        # built value network
        value = create_critic_network('value', state, trainable=True)
        # advantage function
        advantage_f = discounter_r - value

        # critic network learning rate
        c_lr = tf.Variable(self.c_lr, name='c_lr')

        # the critic network loss
        closs = tf.reduce_mean(tf.square(advantage_f))
        ctrain_op = tf.train.AdamOptimizer(c_lr).minimize(closs)

        # built actor network
        pi, pi_params = create_actor_network('pi', state, self.action_space, trainable=True)
        oldpi, oldpi_params = create_actor_network('oldpi', state, self.action_space, trainable=True)

        # sample one action
        sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        action = tf.placeholder(tf.float32, [None, self.action_space], 'action')
        advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        ratio = pi.prob(action) / (oldpi.prob(action) + 1e-5)
        surr = ratio * advantage  # surrogate loss

        # actor network learning rate
        a_lr = tf.Variable(self.a_lr, name='a_lr')
        # clipped surrogate objective
        aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon)*advantage))
        atrain_op = tf.train.AdamOptimizer(a_lr).minimize(aloss)

        def update(data):
            [state_d, action_d, reward_d] = data

            sess.run(update_oldpi_op)  # copy pi to old pi

            adv = sess.run(advantage_f, feed_dict={state: state_d, discounted_r: reward_d})

            res = sess.run([aloss, closs, atrain_op, ctrain_op],
                           feed_dict={state: state_d, action: action_d, discounted_r: reward_d, advantage: adv})
            return res[0], res[1]

        def choose_action(state_d):
            s = np.array(state_d)
            s = s[np.newaxis, :]
            a = sess.run(sample_op, {state: s})[0]
            return a  # np.clip(a, 0, 1)

        def get_value(state_d):
            s = np.array(state_d)
            if s.ndim < 2:
                s = s[np.newaxis, :]
            return sess.run(value, {state: s})[0, 0]

        return update, choose_action, get_value

    def save_weights(self, index):
        self.saver.save(self.sess, 'weights/'+index+'/ppo_weights')
        print('success save weights to weights/'+index+'ppo_weights')

    def load_weights(self, index):
        model_file = tf.train.latest_checkpoint('weights/'+index+'/')
        self.saver.restore(self.sess, model_file)
        print('success load weights from weights/'+index+'/ppo_weights')


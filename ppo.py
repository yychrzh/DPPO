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
#import matplotlib.pyplot as plt
#import gym, threading, queue
from multiprocessing.connection import Client

EP_MAX = 100000
EP_LEN = 2000
N_WORKER = 3          # parallel workers
GAMMA = 0.9           # reward discount factor
A_LR = 1e-4           # 0.0001  # learning rate for actor
C_LR = 1e-3           # 0.001  # learning rate for critic
MIN_BATCH_SIZE = 64   # minimum batch size for updating PPO
UPDATE_STEP = 10      # loop update operation n-steps
EPSILON = 0.2         # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = 36, 9  # state and action dimension

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
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # built value network
        l1 = tf.layers.dense(self.tfs, 256, tf.nn.relu)
        l2 = tf.layers.dense(l1, 128, tf.nn.relu)
        l3 = tf.layers.dense(l2, 128, tf.nn.relu)
        l4 = tf.layers.dense(l3, 128, tf.nn.relu)
        l5 = tf.layers.dense(l4, 128, tf.nn.relu)
        self.v = tf.layers.dense(l5, 1)

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=1)

        self.lock = threading.Lock()
        self.start_time = time.time()
        self.up_to_now_time = 0
        self.total_train_time = 0
        self.mean_train_time = 0

        # self.load_weights('100301')

    def update(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_POLICY_UPDATE_NUM, GLOBAL_SAMPLE_SIZE
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:

                UPDATE_EVENT.wait()  # wait until get batch of data

                # waiting for all workers put the data to the QUEUE
                while True:
                    if GLOBAL_SAMPLE_SIZE >= MIN_BATCH_SIZE:
                        print('got %d samples , begin one new update'%(GLOBAL_SAMPLE_SIZE))
                        GLOBAL_SAMPLE_SIZE = 0
                        break

                start_time = time.time()
                GLOBAL_POLICY_UPDATE_NUM += 1

                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]

                # save weights
                if (GLOBAL_EP+1)%500 == 0:
                    self.save_weights('100301')

                train_time = time.time() - start_time
                self.total_train_time += train_time
                self.mean_train_time = self.total_train_time/GLOBAL_POLICY_UPDATE_NUM

                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    # create action network
    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            # two hidden layer
            l1 = tf.layers.dense(self.tfs, 256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l2, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # create action network
    def create_actor_network(self, name, state_space, action_space, trainable=True):
        with tf.variable_scope(name):
            # two hidden layer
            l1 = tf.layers.dense(state_space, 256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l2, action_space, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, action_space, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # create value network
    def create_critic_network(self, name, state_space, trainable=True):
        with tf.variable_scope(name):
            # built value network
            l1 = tf.layers.dense(state_space, 256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu, trainable=trainable)
            value = tf.layers.dense(l2, 1)
        return value

    def choose_action(self, state):
        s = np.array(state)
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, 0, 1)

    def get_v(self, state):
        s = np.array(state)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save_weights(self, index):
        self.saver.save(self.sess, 'weights/'+index+'/ppo_weights')
        print('success save weights to weights/'+index+'ppo_weights')

    def load_weights(self, index):
        model_file = tf.train.latest_checkpoint('weights/'+index+'/')
        self.saver.restore(self.sess, model_file)
        print ('success load weights from weights/'+index+'/ppo_weights')
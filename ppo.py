import time
import tensorflow as tf
import numpy as np

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

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5, vf_coeff=0.5, ent_coeff=0.01),   # KL penalty
    dict(name='clip', epsilon=0.2, vf_coeff=0.5, ent_coeff=0.0),    # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

HIDDEN_UNITS_1 = 64
HIDDEN_UNITS_2 = 64


class PPO(object):
    def __init__(self, state_space, action_space, max_episode_num, episode_lens, discount_factor=0.95,
                 actor_learning_rate=1e-3, critic_learning_rate=1e-3, mini_batch_size=64, epochs=10):

        self.state_space = state_space
        self.action_space = action_space
        self.max_episode_num = max_episode_num
        self.episode_lens = episode_lens
        self.discount_factor = discount_factor
        self.a_lr = actor_learning_rate
        self.c_lr = critic_learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        if METHOD['name'] == 'clip':
            self.epsilon = METHOD['epsilon']

        self.sess = tf.Session()
        self.hidden_units_1 = HIDDEN_UNITS_1
        self.hidden_units_2 = HIDDEN_UNITS_2

        self.update, self.choose_action, self.get_value = self.train_step_generate()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=1)

    # create action network
    def create_actor_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            # two hidden layer
            l1 = tf.layers.dense(state, self.hidden_units_1, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, self.hidden_units_2, tf.nn.relu, trainable=trainable)
            mu = 2*tf.layers.dense(l2, self.action_space, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, self.action_space, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # create value network
    def create_critic_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            # built value network
            l1 = tf.layers.dense(state, self.hidden_units_1, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, self.hidden_units_2, tf.nn.relu, trainable=trainable)
            value = tf.layers.dense(l2, 1)
        return value

    def train_step_generate(self):
        sess = self.sess
        # 0.state
        state = tf.placeholder(tf.float32, [None, self.state_space], 'state')
        # 1.critic
        with tf.variable_scope('critic'):
            # discounted_return
            discounted_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            # built value network
            value = self.create_critic_network('value', state, trainable=True)
            # advantage function
            advantage_f = discounted_r - value
            # critic network learning rate
            c_lr = tf.Variable(self.c_lr, name='c_lr')
            # the critic network loss
            closs = tf.reduce_mean(tf.square(advantage_f))
            ctrain_op = tf.train.AdamOptimizer(c_lr).minimize(closs)

        # 2.actor
        # built actor network
        pi, pi_params = self.create_actor_network('pi', state, trainable=True)
        oldpi, oldpi_params = self.create_actor_network('oldpi', state, trainable=False)

        with tf.variable_scope('entropy_pen'):
            entropy = tf.reduce_mean(pi.entropy())

        # sample one action
        with tf.variable_scope('sample_action'):
            sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        with tf.variable_scope('update_oldpi'):
            update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        action = tf.placeholder(tf.float32, [None, self.action_space], 'action')
        advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(action) / (oldpi.prob(action) + 1e-5)
                surr = ratio * advantage  # surrogate loss

            if METHOD['name'] == 'kl_pen':
                # kl pen objective
                beta = tf.placeholder(tf.float32, None, 'belta')
                kl_div = tf.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl_div)
                aloss = -(tf.reduce_mean(surr - beta * kl_div))
            else:
                # clipped surrogate objective
                aloss = -tf.reduce_mean(tf.minimum(
                          surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon)*advantage))

            # actor network learning rate
            a_lr = tf.Variable(self.a_lr, name='a_lr')
        with tf.variable_scope('atrain'):
            atrain_op = tf.train.AdamOptimizer(a_lr).minimize(aloss)

        with tf.variable_scope('atrain'):
            t_lr = tf.Variable((self.c_lr+self.a_lr)/2, name='t_lr')
            total_loss = aloss + METHOD['vf_coeff'] * closs - METHOD['ent_coeff'] * entropy
            # train_op = tf.train.AdamOptimizer(t_lr).minimize(total_loss)

        mini_batch_size = self.mini_batch_size
        epochs = self.epochs

        def array_data_sample(data, batch_size):  # data: list of numpy array
            import random
            data_lens = len(data[0])
            batch_size = data_lens if data_lens > batch_size else int(data_lens/2)

            index = random.sample(range(data_lens), batch_size)

            batch = []
            for i in range(len(data)):
                sample = []
                for j in range(batch_size):
                    sample.append(data[i][index[j]])
                batch.append(np.vstack(sample))

            return batch

        def update(data):
            # [state_d, action_d, adv_d, dr_d] = data
            sess.run(update_oldpi_op)  # copy pi to old pi
            res1 = []
            res2 = []

            if METHOD['name'] == 'kl_pen':
                kl = 0
                for _ in range(epochs):
                    [state_d, action_d, adv_d, dr_d] = array_data_sample(data, mini_batch_size)
                    res1 = self.sess.run(
                         [aloss, atrain_op, kl_mean],
                         feed_dict={state: state_d, action: action_d, advantage: adv_d, belta: METHOD['lam']})
                    res2 = sess.run([closs, ctrain_op], feed_dict={state: state_d, discounted_r: dr_d})
                    # res1 = sess.run([aloss, closs, kl_mean, train_op], feed_dict={state: state_d, action: action_d,
                    #                            advantage: adv_d, beta: METHOD['lam'], discounted_r: dr_d})
                    kl = res1[2]
                    if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                        break
                if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    METHOD['lam'] /= 2
                elif kl > METHOD['kl_target'] * 1.5:
                    METHOD['lam'] *= 2
                METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # sometimes explode, this clipping is my solution
                return res1[0], res2[0]
            else:
                for _ in range(epochs):
                    [state_d, action_d, adv_d, dr_d] = array_data_sample(data, mini_batch_size)
                    res1 = sess.run([aloss, atrain_op], feed_dict={state: state_d,
                                                                    action: action_d, advantage: adv_d})
                    res2 = sess.run([closs, ctrain_op], feed_dict={state: state_d, discounted_r: dr_d})

                    # res1 = sess.run([aloss, closs, train_op], feed_dict={state: state_d, action: action_d,
                    #                                                    advantage: adv_d, discounted_r: dr_d})
                    return res1[0], res2[0]

        def choose_action(state_d):
            s = np.array(state_d)
            s = s[np.newaxis, :]
            a = sess.run(sample_op, {state: s})[0]
            # return a  # np.clip(a, 0, 1)
            return np.clip(a, -2, 2)

        def get_value(state_d):
            s = np.array(state_d)
            if s.ndim < 2:
                s = s[np.newaxis, :]
            return sess.run(value, {state: s})[0, 0]

        return update, choose_action, get_value

    def save_weights(self, index):
        # self.saver.save(self.sess, 'weights/'+index+'/ppo_weights')
        self.saver.save(self.sess, 'weights/' + index + '/ppo_weights.ckpt')
        print('success save weights to weights/'+index+'ppo_weights')

    def load_weights(self, index):
        # model_file = tf.train.latest_checkpoint('weights/'+index+'/')
        model_file = 'weights/'+index+'/ppo_weights.ckpt'
        self.saver.restore(self.sess, model_file)
        print('success load weights from weights/'+index+'/ppo_weights')


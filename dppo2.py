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

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
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

# send one episode start flag through multiprocessing.connection
def send_episode_start_flag(send_end):
    start_flag = [321]
    send_end.send(start_flag)

# send one episode end flag through multiprocessing.connection
def send_episode_end_flag(send_end):
    end_flag = [123]
    send_end.send(end_flag)

# send training end flag through multiprocessing.connection
def send_train_end_flag(send_end):
    end_flag = [999]
    send_end.send(end_flag)

# send connect flag through multiprocessing.connection
def send_connect_flag(send_end):
    end_flag = [111]
    send_end.send(end_flag)

def floatify(n_p):
    return [float(n_p[i]) for i in range(len(n_p))]

class PPO(object):
    def __init__(self):
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
            #four middle layer
            l1 = tf.layers.dense(self.tfs, 256, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 128, tf.nn.relu, trainable=trainable)
            l3 = tf.layers.dense(l2, 128, tf.nn.relu, trainable=trainable)
            l4 = tf.layers.dense(l3, 128, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l4, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l4, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

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

class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.ppo = GLOBAL_PPO
        self.lock = GLOBAL_PPO.lock
        self.start_time = GLOBAL_PPO.start_time

    def work(self, name, conn):
        """
        conn: communicating tools, like multiprocessing.connection
        return: None
        """
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, GLOBAL_POLICY_UPDATE_NUM, GLOBAL_SAMPLE_SIZE
        policy_update_num = GLOBAL_POLICY_UPDATE_NUM

        while not COORD.should_stop():

            # send start flag to the env_processing
            send_episode_start_flag(conn)

            #recv the init observation
            data = conn.recv()
            s = data[0]
            if s is None:
                print ('recv error')
                return 0

            ep_r = 0
            steps = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            s_time = time.time()

            while True and steps < EP_LEN:

                if not ROLLING_EVENT.is_set():                 # while global PPO is updating
                    ROLLING_EVENT.wait()                       # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data

                # predict action under current policy
                a = self.ppo.choose_action(s)
                a = floatify(a)
                # send the action to env processing
                conn.send(a)

                #recv the next state and reward
                data = conn.recv()

                if data is not None:
                    s_ = data[0]
                    r = data[1]
                    done = data[2]

                buffer_s.append(s)   #state before action
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_
                ep_r += r
                steps += 1

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if done or steps == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:

                    print('worker %d send %d samples to the ppo_update'%(name, len(buffer_r)))
                    GLOBAL_SAMPLE_SIZE += len(buffer_r)

                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        #send train end flag to the env processing
                        send_train_end_flag(conn)
                        COORD.request_stop()
                        break

                if done:
                    #one episode end and send the reset flag to the env processing
                    send_episode_end_flag(conn)
                    break

            GLOBAL_EP += 1
            u_time = time.time() - self.start_time
            total_time = time.time() - s_time
            #print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )
            print('>>>episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward {:.2f}'.format(steps,
                   total_time, total_time/steps, ep_r),
                  '\n>>>episode_num : %d , up_to_now_time : %d h %d m %d s , update_num %d , mean_train_time %.4f sec'%(GLOBAL_EP, u_time/3600, (u_time%3600)/60,
                   (u_time%3600)%60, GLOBAL_POLICY_UPDATE_NUM,self.ppo.mean_train_time))

password_list = [b'secret password A', b'secret password B', b'secret password C', b'secret password D',
                 b'secret password E', b'secret password F', b'secret password G', b'secret password H',
                 b'secret password I', b'secret password J', b'secret password K', b'secret password L',
                 b'secret password M', b'secret password N', b'secret password O', b'secret password P']

env_process = 3
address_num_base = 1995

if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = []

    GLOBAL_UPDATE_COUNTER = 0
    GLOBAL_EP = 0
    GLOBAL_POLICY_UPDATE_NUM = 0
    GLOBAL_SAMPLE_SIZE = 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []

    for i in range(env_process):
        address_num = address_num_base * (i + 1)
        address = ('localhost', address_num)
        conn = Client(address, authkey=password_list[i])
        worker = Worker(wid=i)
        workers.append(worker)
        t = threading.Thread(target=worker.work, args=(i, conn))
        threads.append(t)
        send_connect_flag(conn)
        print ('send connect flag to the env processing {}...'.format(i))
        time.sleep(1)

    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))

    print('start all worker threading...')
    for x in threads:
        x.start()
    print('all worker threading start success')

    COORD.join(threads)
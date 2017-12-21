import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
from ppo import PPO

EP_MAX = 4000
EP_LEN = 128*2
HORIZON = 128
GAMMA = 0.99          # reward discount factor
LAMDA = 1
A_LR = 1e-4           # 0.0001  # learning rate for actor
C_LR = 3e-4           # 0.001  # learning rate for critic
EPOCHS = 4
MINI_BATCH_SIZE = 64  # minimum batch size for updating PPO
GAME = 'Pendulum-v0'
S_DIM = 3             # state and action dimension
A_DIM = 1             # action dimension
SAVA_INDEX = '122001'


# calculate the T time-steps advantage function A1, A2, A3, ...., AT
def calculate_advantage(ppo, trajectory):
    [bs, ba, br, bd] = trajectory

    T = len(ba)
    gamma = GAMMA
    lam = LAMDA

    last_adv = 0
    advantage = [None]*T
    discounted_return = [None]*T
    for t in reversed(range(T)):  # step T-1, T-2 ... 1
        delta = br[t] + (0 if bd[t] else gamma * ppo.get_value(bs[t+1])) - ppo.get_value(bs[t])
        advantage[t] = delta + gamma * lam * (1 - bd[t]) * last_adv
        last_adv = advantage[t]

        if t == T - 1:
            discounted_return[t] = br[t] + (0 if bd[t] else gamma * ppo.get_value(bs[t+1]))
            # print(discounted_return[t], bd[t])
        else:
            discounted_return[t] = br[t] + gamma * (1 - bd[t]) * discounted_return[t+1]
    return advantage, discounted_return


# execute the env for one episode, got the trajectory and train on it
def execute_one_episode(env, ppo):
    time_steps = 0
    episode_r = 0
    aloss = 0
    closs = 0
    terminate_flag = 0
    start_time = time.time()

    # env_reset
    state = env.reset()

    while time_steps < EP_LEN:
        if terminate_flag is True:
            break

        steps = 0
        buffer_s, buffer_a, buffer_r, buffer_d, buffer_advantage, buffer_dr = [], [], [], [], [], []
        while steps < HORIZON:
            # env.render()
            # get action
            action = ppo.choose_action(state)
            # execute one action
            state_after_action, reward, done, _ = env.step(action)

            buffer_s.append(state)
            buffer_a.append(action)
            buffer_r.append((reward + 8) / 8)  # normalize reward, find to be useful
            buffer_d.append(done)
            state = state_after_action
            episode_r += reward
            time_steps += 1
            steps += 1
            if done is True:
                terminate_flag = done
                break

        buffer_s.append(state)   # add the terminate state to the buffer
        # print(len(buffer_s), len(buffer_a), len(buffer_r), len(buffer_d))
        buffer_advantage, buffer_dr = calculate_advantage(ppo, [buffer_s, buffer_a, buffer_r, buffer_d])
        buffer_s.pop()           # remove the terminate state
        # print(len(buffer_s), len(buffer_a), len(buffer_r), len(buffer_d))

        bs, ba = np.vstack(buffer_s), np.vstack(buffer_a)
        # badv, bdr = np.array(buffer_advantage)[:, np.newaxis], np.array(buffer_dr)[:, np.newaxis]
        badv, bdr = np.vstack(buffer_advantage), np.vstack(buffer_dr)
        aloss, closs = ppo.update([bs, ba, badv, bdr])

    return [time_steps, episode_r, (time.time() - start_time), aloss, closs]


# test the trained weights for 1000 steps
def test():
    env = gym.make(GAME).unwrapped

    agent = PPO(state_space=S_DIM, action_space=A_DIM, max_episode_num=EP_MAX, episode_lens=EP_LEN,
                discount_factor=GAMMA, actor_learning_rate=A_LR, critic_learning_rate=C_LR,
                mini_batch_size=MINI_BATCH_SIZE, epochs=EPOCHS)

    agent.load_weights(SAVA_INDEX)
    # env_reset
    # state = env.reset()
    # print(state)
    # state = env.reset()
    # print(state)
    state = env.reset()
    print(state)
    steps = 0
    episode_r = 0
    all_value = []

    while steps < 1000:
        env.render()
        # get action
        action = agent.choose_action(state)
        # execute one action
        state_after_action, reward, done, _ = env.step(action)
        steps += 1
        episode_r += reward
        state = state_after_action
        state_value = agent.get_value(state)
        all_value.append(state_value)

    plt.plot(np.arange(len(all_value)), all_value)
    plt.xlabel('state')
    plt.ylabel('state value')
    plt.show()

    print("test 1000 steps, got reward: %i" % episode_r)


def train():
    env = gym.make(GAME).unwrapped
    all_ep_r = []

    agent = PPO(state_space=S_DIM, action_space=A_DIM, max_episode_num=EP_MAX, episode_lens=EP_LEN,
                discount_factor=GAMMA, actor_learning_rate=A_LR, critic_learning_rate=C_LR,
                mini_batch_size=MINI_BATCH_SIZE, epochs=EPOCHS)

    # load weights
    agent.load_weights(SAVA_INDEX)

    # run(env, agent)
    for i in range(EP_MAX):
        [steps, episode_r, c_time, aloss, closs] = execute_one_episode(env, agent)
        print('Ep: %4d' % i, "|Ep_r: %i" % episode_r, '|aloss: %8.4f' % aloss, '|closs: %8.4f' % closs,
              '|steps: %4d' % steps, '|time: %6.4f' % c_time)

        if i == 0:
            all_ep_r.append(episode_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + episode_r * 0.1)

    agent.save_weights(SAVA_INDEX)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()


if __name__ == '__main__':

    train()
    # test()
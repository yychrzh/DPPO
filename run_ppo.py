import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from ppo import PPO

EP_MAX = 3000
EP_LEN = 128 * 5
HORIZON = 128
GAMMA = 0.99          # reward discount factor
LAMDA = 0.95
A_LR = 3e-4           # 0.0001  # learning rate for actor
C_LR = 5e-4           # 0.001  # learning rate for critic
EPOCHS = 4
MINI_BATCH_SIZE = 32  # minimum batch size for updating PPO
A_UPDATE_STEP = 10    # loop update operation n-steps
C_UPDATE_STEP = 10    # loop update operation n-steps
EPSILON = 0.2         # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM = 3             # state and action dimension
A_DIM = 1             # action dimension


def calculate_advantage(ppo, trajectory):
    [bs, ba, br, bd] = trajectory

    T = len(ba)
    gamma = GAMMA
    lam = LAMDA

    last_adv = 0
    advantage = [None]*T
    last_dr = br[-1] + (0 if bd[-1] else gamma * ppo.get_value(bs[-1]))
    discounted_return = [None]*T
    for t in reversed(range(T)):  # step T-1, T-2 ... 1
        delta = br[t] + (0 if bd[t] else gamma * ppo.get_value(bs[t+1])) - ppo.get_value(bs[t])
        advantage[t] = delta + gamma * lam * (1 - bd[t]) * last_adv
        last_adv = advantage[t]

        discounted_return[t] = br[t] + gamma * last_dr
        last_dr = discounted_return[t]
    return advantage, discounted_return


def execute_one_episode(env, ppo):
    time_steps = 0
    episode_r = 0
    terminate_flag = 0

    # env_reset
    state = env.reset()

    while time_steps < EP_LEN:
        if terminate_flag is True:
            break

        buffer_s, buffer_a, buffer_r, buffer_d, buffer_advantage, buffer_dr = [], [], [], [], [], []
        while time_steps < HORIZON:
            env.render()
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
            if done is True:
                terminate_flag = done
                break
        buffer_s.append(state)   # add the terminate state to the buffer
        buffer_advantage, buffer_dr = calculate_advantage(ppo, (buffer_s, buffer_a, buffer_r, buffer_d))
        buffer_s.pop()           # remove the terminate state

        bs, ba = np.vstack(buffer_s), np.vstack(buffer_a)
        badv, bdr = np.array(buffer_advantage)[:, np.newaxis], np.array(buffer_dr)[:, np.newaxis]
        ppo.update([bs, ba, badv, bdr])


def run(env, ppo):
    all_ep_r = []
    episode_num = 0

    while episode_num < EP_MAX:
        episode_num += 1
        time_steps = 0
        episode_r = 0
        done = False

        # env_reset
        state = env.reset()

        buffer_s, buffer_a, buffer_r, buffer_d = [], [], [], []
        aloss = 0
        closs = 0

        while time_steps < HORIZON:
            env.render()
            time_steps += 1
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
            if done is True:
                break

            # update ppo
            if (time_steps + 1) % MINI_BATCH_SIZE == 0 or time_steps == EP_LEN - 1 or done is True:
                if done is True:   # terminate state, the value will be set to zero
                    v_s_ = 0
                else:
                    v_s_ = ppo.get_value(state_after_action)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                # buffer_s, buffer_a, buffer_r = [], [], []
                [aloss, closs] = ppo.update([bs, ba, br])
                break

        if episode_num == 1:
            all_ep_r.append(episode_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + episode_r * 0.1)
        print('Ep: %i' % episode_num, "|Ep_r: %i" % episode_r, '|aloss: %8.4f' % aloss, '|closs: %8.4f' % closs,
              '|time_steps: %d' % time_steps)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()

    ppo.save_weights('121801')


def main():
    env = gym.make('Pendulum-v0').unwrapped

    agent = PPO(state_space=S_DIM, action_space=A_DIM, max_episode_num=EP_MAX, episode_lens=EP_LEN,
                discount_factor=GAMMA, actor_learning_rate=A_LR, critic_learning_rate=C_LR,
                mini_batch_size=MINI_BATCH_SIZE, epsilon=EPSILON,
                actor_update_stpes=A_UPDATE_STEP, critic_update_steps=C_UPDATE_STEP)

    run(env, agent)


if __name__ == '__main__':

    main()
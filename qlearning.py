import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    """
    创建q表

    :param n_states:    状态数
    :param actions:     动作集
    :return:
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table


# q_table
"""
    left    right
0   0.0     0.0
1   0.0     0.0
... ...
"""


def choose_action(state, q_table):
    """
    定义动作

    :param state:   状态
    :param q_table: q表
    :return: 动作
    """
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax(axis=0)
    return action_name


def get_env_feedback(S, A):
    """
    环境反馈

    :param S: 该时刻状态
    :param A: 动作
    :return:
        S_: 下一时刻状态
        R:  奖励
    """
    if A == "right":
        if S == N_STATES-2:
            S_ = 'terminal'
            R = 1
        else :
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        isterminal = False
        update_env(S, episode, step_counter)
        while not isterminal:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                isterminal = True

            q_table.loc[S, A] = q_table.loc[S, A] + ALPHA * (q_target - q_predict)
            S = S_
            step_counter += 1
            update_env(S, episode, step_counter)
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
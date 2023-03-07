import numpy as np
import discrete_pendulum
import matplotlib.pyplot as plt

"""
Initial hw1 for Scott Bout (bout2)
"""


class SARSA:

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.A = np.zeros(self.env.num_states)
        self.rew_plot_list = []

    def update_q_table(self, alpha, rew, state, action, next_state, next_action):
        self.Q[state][action] = self.Q[state][action] + alpha * (rew + self.gamma *
                                                                 self.Q[next_state][next_action] - self.Q[state][action])

    def reset_tables(self):
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.A = np.zeros(self.env.num_states)

    def rollout(self, epsilon_threshold, alpha, episode_length=int(1e2)):
        rew_list = []
        for s in range(len(self.Q)):
            epsilon = np.random.uniform()
            if epsilon > epsilon_threshold:
                self.A[s] = np.argmax(self.Q[s])
            else:
                self.A[s] = np.random.randint(0, self.env.num_actions)
        s = self.env.reset()
        for step in range(episode_length):
            a = int(self.A[s])
            (s1, rew, done) = self.env.step(a)
            rew_list.append(rew)
            a1 = int(self.A[s1])
            self.update_q_table(alpha, rew, s, a, s1, a1)
            if done:
                self.rew_plot_list.append(sum(rew_list))
                rew_list.clear()
                s = self.env.reset()
            else:
                s = s1

        return self.rew_plot_list

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(1e2)):
        plot_list = []
        for episode in range(total_timesteps):
            list1 = self.rollout(epsilon_threshold=epsilon_threshold, alpha=alpha)
            plot_list.append(sum(list1))
            self.rew_plot_list.clear()

        return plot_list

    def play(self):
        s = self.env.reset()
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
            'theta': [self.env.x[0]],  # agent does not have access to this, but helpful for display
            'thetadot': [self.env.x[1]],  # agent does not have access to this, but helpful for display
        }
        done = False
        step = 0
        while not done:
            # self.env.render()
            a = np.argmax(self.Q[s])
            (s, r, done) = self.env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(self.env.x[0])
            log['thetadot'].append(self.env.x[1])
            if done:
                s = self.env.reset()
            else:
                step += 1

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(log['t'], log['s'])
        ax[0].plot(log['t'][:-1], log['a'])
        ax[0].plot(log['t'][:-1], log['r'])
        ax[0].legend(['s', 'a', 'r'])
        ax[1].plot(log['t'], log['theta'])
        ax[1].plot(log['t'], log['thetadot'])
        ax[1].legend(['theta', 'thetadot'])
        plt.title('trajectory example')
        plt.show()

    def plot(self):
        list1 = self.learn(epsilon_threshold=0.1, alpha=1e-5)
        plt.plot(list1)
        self.reset_tables()

        list2 = self.learn(epsilon_threshold=0.5, alpha=1e-5)
        plt.plot(list2)
        self.reset_tables()

        list3 = self.learn(epsilon_threshold=0.01, alpha=1e-5)
        plt.plot(list3)
        plt.legend(['e = 0.1, alpha = 1e-5', 'e = 0.5, alpha = 1e-5', 'e = 0.01, alpha = 1e-5'])
        plt.xlabel('episodes')
        plt.ylabel('sum of rewards')
        plt.title('return vs episodes for varying epsilon')
        plt.show()
        self.reset_tables()

        list4 = self.learn(epsilon_threshold=0.1, alpha=1e-5)
        plt.plot(list4)
        self.reset_tables()

        list5 = self.learn(epsilon_threshold=0.1, alpha=1e-2)
        plt.plot(list5)
        self.reset_tables()

        list6 = self.learn(epsilon_threshold=0.1, alpha=1e-7)
        plt.plot(list6)
        plt.legend(['e = 0.1, alpha = 1e-5', 'e = 0.1, alpha = 1e-2', 'e = 0.1, alpha = 1e-7'])
        plt.xlabel('episodes')
        plt.ylabel('sum of rewards')
        plt.title('return vs episodes for varying alpha')
        plt.show()

        self.reset_tables()
        _ = self.learn(epsilon_threshold=0.1, alpha=1e-5)
        self.play()

    def get_policy(self):
        policy = np.zeros(self.env.num_states)
        for s in range(len(self.Q)):
            policy[s] = np.argmax(self.Q[s])

        return policy


class QLearning:
    """
    Q(s,a) = Q_{old}(s,a) + alpha * (R_{t+1} + gamma * max_{a} Q_{old}(s',a) - Q_{old}(s,a)
    """

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.rew_plot_list = []

    def update_q_table(self, alpha, rew, state, action, next_state):
        self.Q[state][action] = self.Q[state][action] + alpha * (rew + self.gamma *
                                                                 max(self.Q[next_state]) - self.Q[state][action])

    def reset_q_table(self):
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))

    def rollout(self, epsilon_threshold, alpha, episode_length=int(1e2)):
        rew_list = []
        s = self.env.reset()
        for step in range(episode_length):
            epsilon = np.random.uniform()
            if epsilon > epsilon_threshold:
                a = np.argmax(self.Q[s])
            else:
                a = np.random.randint(0, self.env.num_actions)
                if a == np.argmax(self.Q[s]):
                    a = np.random.randint(0, self.env.num_actions)
            a = int(a)
            (s1, rew, done) = self.env.step(a)
            rew_list.append(rew)
            self.update_q_table(alpha, rew, s, a, s1)
            if done:
                self.rew_plot_list.append(sum(rew_list))
                rew_list.clear()
                s = self.env.reset()
            else:
                s = s1

        return self.rew_plot_list

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(1e2)):
        plot_list = []
        for episode in range(total_timesteps):
            list1 = self.rollout(epsilon_threshold=epsilon_threshold, alpha=alpha)
            plot_list.append(sum(list1))
            self.rew_plot_list.clear()

        return plot_list

    def play(self):
        s = self.env.reset()
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
            'theta': [self.env.x[0]],  # agent does not have access to this, but helpful for display
            'thetadot': [self.env.x[1]],  # agent does not have access to this, but helpful for display
        }
        done = False
        step = 0
        while not done:
            # self.env.render()
            print(f'self.q = {self.Q}')
            a = np.argmax(self.Q[s])
            (s, r, done) = self.env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(self.env.x[0])
            log['thetadot'].append(self.env.x[1])
            if done:
                s = self.env.reset()
            else:
                step += 1

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(log['t'], log['s'])
        ax[0].plot(log['t'][:-1], log['a'])
        ax[0].plot(log['t'][:-1], log['r'])
        ax[0].legend(['s', 'a', 'r'])
        ax[1].plot(log['t'], log['theta'])
        ax[1].plot(log['t'], log['thetadot'])
        ax[1].legend(['theta', 'thetadot'])
        plt.title('trajectory example')
        plt.show()

    def plot(self):
        list1 = self.learn(epsilon_threshold=0.1, alpha=1e-5)
        plt.plot(list1)
        self.reset_q_table()

        list2 = self.learn(epsilon_threshold=0.5, alpha=1e-5)
        plt.plot(list2)
        self.reset_q_table()

        list3 = self.learn(epsilon_threshold=0.01, alpha=1e-5)
        plt.plot(list3)
        plt.legend(['e = 0.1, alpha = 1e-5', 'e = 0.5, alpha = 1e-5', 'e = 0.01, alpha = 1e-5'])
        plt.xlabel('episodes')
        plt.ylabel('sum of rewards')
        plt.title('return vs episodes for varying epsilon')
        plt.show()
        self.reset_q_table()

        list4 = self.learn(epsilon_threshold=0.1, alpha=1e-5)
        plt.plot(list4)
        self.reset_q_table()

        list5 = self.learn(epsilon_threshold=0.1, alpha=1e-2)
        plt.plot(list5)
        self.reset_q_table()

        list6 = self.learn(epsilon_threshold=0.1, alpha=1e-7)
        plt.plot(list6)
        plt.legend(['e = 0.1, alpha = 1e-5', 'e = 0.1, alpha = 1e-2', 'e = 0.1, alpha = 1e-7'])
        plt.xlabel('episodes')
        plt.ylabel('sum of rewards')
        plt.title('return vs episodes for varying alpha')
        plt.show()

        self.reset_q_table()

        _ = self.learn(epsilon_threshold=0.1, alpha=1e-5)
        self.play()

    def get_policy(self):
        policy = np.zeros(self.env.num_states)
        for s in range(len(self.Q)):
            policy[s] = np.argmax(self.Q[s])

        return policy


class TDzero:

    def __init__(self, env, gamma, policy):
        self.env = env
        self.gamma = gamma
        self.policy = policy
        self.values = np.zeros(self.env.num_states)
        self.rew_plot_list = []

    def update_values_table(self, alpha, rew, state, next_state):
        self.values[state] = self.values[state] + alpha * (rew + self.gamma *
                                                           self.values[next_state] - self.values[state])

    def rollout(self, alpha, episode_length=int(1e2)):
        rew_list = []
        s = self.env.reset()
        for step in range(episode_length):
            a = int(self.policy[s])
            # print(f's = {s}, a = {a}')
            (s1, rew, done) = self.env.step(a)
            rew_list.append(rew)
            self.update_values_table(alpha=alpha, rew=rew, state=s, next_state=s1)
            # print(f'values = {self.values}')
            if done:
                self.rew_plot_list.append(sum(rew_list))
                rew_list.clear()
                s = self.env.reset()
            else:
                s = s1

        return self.rew_plot_list

    def learn(self, alpha, total_timesteps=int(1e2)):
        plot_list = []
        for episode in range(total_timesteps):
            list1 = self.rollout(alpha=alpha)
            plot_list.append(sum(list1))
            self.rew_plot_list.clear()

        return plot_list

    def plot(self):
        l1 = self.learn(alpha=1e-5)
        plt.plot(l1)
        plt.show()


def main(mode=0, submode=0):
    envs = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)
    gamma = 0.95
    """
    mode = 0 activates SARSA
    mode = 1 activates Q-Learning
    mode = 2 activates TD(0)
        submode = 0 activates SARSA
        submode = 1 activates Q-Learning
    """
    # mode = 1
    # submode = 0
    if mode == 0:
        policy = SARSA(envs, gamma)
        policy.plot()
    elif mode == 1:
        policy = QLearning(envs, gamma)
        policy.plot()
    elif mode == 2:
        if submode == 0:
            policy = SARSA(envs, gamma)
            _ = policy.learn(epsilon_threshold=0.1, alpha=1e-5)
            sarsa = policy.get_policy()
            values = TDzero(envs, gamma, policy=sarsa)
            values.plot()
        elif submode == 1:
            policy = QLearning(envs, gamma)
            _ = policy.learn(epsilon_threshold=0.1, alpha=1e-5)
            qlearning = policy.get_policy()
            values = TDzero(envs, gamma, policy=qlearning)
            values.plot()


if __name__ == '__main__':
    main(mode=2)

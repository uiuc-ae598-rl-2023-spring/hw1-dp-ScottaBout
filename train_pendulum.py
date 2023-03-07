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
        self.save_fig = False

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

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(10e3), anneal=False):
        plot_list = []
        anneal_list = [10., 9., 8., 7., 6., 5., 4., 3., 2., 1.]
        for episode in range(total_timesteps):
            if anneal:
                if total_timesteps / (episode + 1) in anneal_list:
                    print(f'old epislon threshold = {epsilon_threshold}')
                    epsilon_threshold -= 0.1
                    print(f'new epsilon threshold = {epsilon_threshold}')
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
        if self.save_fig:
            plt.savefig('figures/pendulum/sarsa_example_trajectory_pendulum.png')
        plt.show()

    def plot(self, save_fig, graph=0):
        self.save_fig = save_fig
        if graph == 0:
            list1 = self.learn(epsilon_threshold=0.1, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list1)
            self.reset_tables()

            list2 = self.learn(epsilon_threshold=0.5, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list2)
            self.reset_tables()

            list3 = self.learn(epsilon_threshold=0.01, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list3)
            plt.legend(['e = 0.1, alpha = 0.3', 'e = 0.5, alpha = 0.3', 'e = 0.01, alpha = 0.3'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying epsilon')
            if save_fig:
                plt.savefig('figures/pendulum/sarsa_example_varying_epsilon_pendulum.png')
            plt.show()
            self.reset_tables()

        elif graph == 1:

            list4 = self.learn(epsilon_threshold=0.7, alpha=1e-5, total_timesteps=int(1e3))
            plt.plot(list4)
            self.reset_tables()

            list5 = self.learn(epsilon_threshold=0.7, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list5)
            self.reset_tables()

            list6 = self.learn(epsilon_threshold=0.7, alpha=1e-7, total_timesteps=int(1e3))
            plt.plot(list6)
            plt.legend(['e = 0.7, alpha = 1e-5', 'e = 0.7, alpha = 0.3', 'e = 0.7, alpha = 1e-7'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying alpha')
            if save_fig:
                plt.savefig('figures/pendulum/sarsa_example_varying_alpha_pendulum.png')
            plt.show()

            self.reset_tables()

        elif graph == 2:
            list1 = self.learn(epsilon_threshold=0.9, alpha=0.3, anneal=True)
            plt.plot(list1)
            plt.xlabel(f'episodes')
            plt.ylabel(f'sum of rewards')
            plt.title(f'return over time')
            if save_fig:
                plt.savefig('figures/pendulum/sarsa_example_learning_curve_pendulum.png')
            plt.show()
            self.play()
            pol, sa = self.get_policy()
            plt.plot(pol, 's')
            plt.title(f'Policy')
            plt.xlabel(f'State')
            plt.ylabel(f'Action')
            if save_fig:
                plt.savefig('figures/pendulum/sarsa_example_policy_pendulum.png')
            plt.show()
            plt.plot(sa, 's')
            plt.title(f'State-Action Values')
            plt.xlabel(f'State')
            plt.ylabel(f'State-Action Value')
            if save_fig:
                plt.savefig(f'figures/pendulum/sarsa_example_state-action_pendulum.png')
            plt.show()

    def get_policy(self):
        policy = np.zeros(self.env.num_states)
        sa = np.zeros(self.env.num_states)
        for s in range(len(self.Q)):
            policy[s] = np.argmax(self.Q[s])
            sa[s] = self.Q[s][np.argmax(self.Q[s])]

        return policy, sa


class QLearning:
    """
    Q(s,a) = Q_{old}(s,a) + alpha * (R_{t+1} + gamma * max_{a} Q_{old}(s',a) - Q_{old}(s,a)
    """

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.rew_plot_list = []
        self.save_fig = False

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

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(10e3), anneal=False):
        plot_list = []
        anneal_list = [10.0, 9.0, 8.0, 7.0, 6., 5., 4., 3., 2., 1.]
        for episode in range(total_timesteps):
            if anneal:
                if total_timesteps / (episode + 1) in anneal_list:
                    print(f'old epislon threshold = {epsilon_threshold}')
                    epsilon_threshold -= 0.1
                    print(f'new epsilon threshold = {epsilon_threshold}')
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
            # print(f'self.q = {self.Q}')
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
        if self.save_fig:
            plt.savefig(f'figures/pendulum/qlearning_example_trajectory.png')
        plt.show()

    def plot(self, save_fig, graph=0):
        self.save_fig = save_fig
        if graph == 0:
            list1 = self.learn(epsilon_threshold=0.1, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list1)
            self.reset_q_table()

            list2 = self.learn(epsilon_threshold=0.5, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list2)
            self.reset_q_table()

            list3 = self.learn(epsilon_threshold=0.01, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list3)
            plt.legend(['e = 0.1, alpha = 0.3', 'e = 0.5, alpha = 0.3', 'e = 0.01, alpha = 0.3'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying epsilon')
            if save_fig:
                plt.savefig(f'figures/pendulum/qlearning_example_varying_epsilon_pendulum.png')
            plt.show()
            self.reset_q_table()

        elif graph == 1:
            list4 = self.learn(epsilon_threshold=0.3, alpha=1e-5, total_timesteps=int(1e3))
            plt.plot(list4)
            self.reset_q_table()

            list5 = self.learn(epsilon_threshold=0.3, alpha=0.3, total_timesteps=int(1e3))
            plt.plot(list5)
            self.reset_q_table()

            list6 = self.learn(epsilon_threshold=0.3, alpha=1e-7, total_timesteps=int(1e3))
            plt.plot(list6)
            plt.legend(['e = 0.3, alpha = 1e-5', 'e = 0.3, alpha = 0.3', 'e = 0.3, alpha = 1e-7'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying alpha')
            if save_fig:
                plt.savefig(f'figures/pendulum/qlearning_example_varying_alpha_pendulum.png')
            plt.show()

            self.reset_q_table()

        elif graph == 2:
            list1 = self.learn(epsilon_threshold=0.9, alpha=0.3, anneal=True)
            plt.plot(list1)
            plt.xlabel(f'episodes')
            plt.ylabel(f'sum of rewards')
            plt.title(f'return over time')
            if save_fig:
                plt.savefig('figures/pendulum/qlearning_example_learning_curve_pendulum.png')
            plt.show()
            self.play()
            pol, sa = self.get_policy()
            plt.plot(pol, 's')
            plt.title(f'Policy')
            plt.xlabel(f'State')
            plt.ylabel(f'Action')
            if save_fig:
                plt.savefig('figures/pendulum/qlearning_example_policy_pendulum.png')
            plt.show()
            plt.plot(sa, 's')
            plt.title(f'State-Action Values')
            plt.xlabel(f'State')
            plt.ylabel(f'State-Action Value')
            if save_fig:
                plt.savefig(f'figures/pendulum/qlearning_example_state-action_pendulum.png')
            plt.show()

    def get_policy(self):
        policy = np.zeros(self.env.num_states)
        sa = np.zeros(self.env.num_states)
        for s in range(len(self.Q)):
            policy[s] = np.argmax(self.Q[s])
            sa[s] = self.Q[s][np.argmax(self.Q[s])]

        return policy, sa


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

    def plot(self, save_fig):
        l1 = self.learn(alpha=0.3)
        plt.plot(self.values, 's')
        plt.title(f'State Values')
        plt.xlabel(f'State')
        plt.ylabel(f'State-Value')
        if save_fig:
            plt.savefig(f'figures/pendulum/sarsa_example_td0_state-value_pendulum.png')
        plt.show()


def main(mode=0, submode=0):
    envs = discrete_pendulum.Pendulum()
    gamma = 0.95

    # mode = 1
    # submode = 0
    if mode == 0:
        policy = SARSA(envs, gamma)
        policy.plot(save_fig=False, graph=1)

    elif mode == 1:
        policy = QLearning(envs, gamma)
        policy.plot(save_fig=False, graph=2)

    elif mode == 2:
        if submode == 0:
            policy = SARSA(envs, gamma)
            _ = policy.learn(epsilon_threshold=0.9, alpha=0.3, anneal=True)
            sarsa, _ = policy.get_policy()
            values = TDzero(envs, gamma, policy=sarsa)
            values.plot(save_fig=True)
        elif submode == 1:
            policy = QLearning(envs, gamma)
            _ = policy.learn(epsilon_threshold=0.9, alpha=0.3, anneal=True)
            qlearning, _ = policy.get_policy()
            values = TDzero(envs, gamma, policy=qlearning)
            values.plot(save_fig=True)


if __name__ == '__main__':
    """
    mode = 0 activates SARSA
    mode = 1 activates Q-Learning
    mode = 2 activates TD(0)
        submode = 0 activates SARSA
        submode = 1 activates Q-Learning
    """
    main(mode=2, submode=0)

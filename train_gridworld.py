import numpy as np
import gridworld
import matplotlib.pyplot as plt


class PolicyIteration:
    """
    Policy Iteration
    v(s) = E[R_{t+1} + GAMMA * V(S_{t+1} | S_{t} = S]
    v(s) = r(s) + gamma * SUM p(s' | s,a = pi(s)) v(s')
    where pi(s) = argmax_{a} [sum_{s'} p(s' | s,a) v(s')
    """

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.values = np.zeros(self.env.num_states)
        self.old_values = np.ones(self.env.num_states)
        self.actions = np.zeros(self.env.num_states)
        self.old_actions = np.ones(self.env.num_states)
        self.values_list_plot = []

    def evaluate(self, total_timesteps=int(1e5)):
        for k in range(total_timesteps):
            for s in range(self.env.num_states):
                self.env.s = s
                (s1, r1, _) = self.env.step(self.actions[s])
                p = self.env.p(s1, s, self.actions[s])
                re = r1 + self.gamma * self.values[s1]
                self.values[s] = np.sum(p * re)

            self.values_list_plot.append(np.mean(self.values))
            if np.allclose(self.old_values, self.values, rtol=0, atol=1e-20):
                break
            else:
                self.old_values = self.values.copy()

        return self.values

    def improve(self):
        v_list = np.zeros(self.env.num_actions)
        actions = np.zeros(self.env.num_states)
        for s in range(self.env.num_states):
            for a in range(self.env.num_actions):
                self.env.s = s
                (s1, r1, _) = self.env.step(a)
                p = self.env.p(s1, s, a)
                v_list[a] = r1 + self.gamma * np.sum(p * self.values[s1])
            actions[s] = np.argmax(v_list)

        self.actions = actions

    def learn_actions(self):
        for k in range(100):
            self.evaluate()
            self.improve()

        plt.plot(self.values_list_plot)
        plt.show()

        return self.actions

    def play(self):
        actions = self.learn_actions().astype(int)
        s = self.env.reset()
        done = False
        step = 0
        while not done:
            self.env.render()
            a = actions[s]
            (s, r, done) = self.env.step(a)
            if done:
                s = self.env.reset()
            else:
                step += 1

    def plot(self):
        self.learn_actions()


class ValueIteration:
    """
    Value Iteration
    v(s) = max_{a} E[R_{t+1} + GAMMA * V(S_{t+1} | S_{t} = S | A_{t} = A]
    """

    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.values = np.zeros(self.env.num_states)
        self.old_values = np.ones(self.env.num_states)
        self.action_space = self.env.num_actions
        self.values_list_plot = []

    def learn_values(self, total_timesteps=1e5):
        # iterate over states
        for k in range(int(total_timesteps)):
            for s in range(self.env.num_states):
                v_list = np.zeros(self.env.num_actions)
                for a in range(self.env.num_actions):
                    self.env.s = s
                    s1, r1, _ = self.env.step(a)
                    p = self.env.p(s1, s, a)
                    v_list[a] = r1 + self.gamma * np.sum(p * self.values[s1])

                self.values[s] = max(v_list)

            self.values_list_plot.append(np.mean(self.values))

            if np.allclose(self.old_values, self.values, rtol=0, atol=1e-20):
                plt.plot(self.values_list_plot)
                plt.show()
                break
            else:
                self.old_values = self.values.copy()

        return self.values

    def get_actions(self, values):
        v_list = np.zeros(self.env.num_actions)
        actions = np.zeros(self.env.num_states)
        for s in range(self.env.num_states):
            for a in range(self.env.num_actions):
                self.env.s = s
                (s1, r1, _) = self.env.step(a)
                p = self.env.p(s1, s, a)
                v_list[a] = r1 + self.gamma * np.sum(p * values[s1])
            actions[s] = np.argmax(v_list)

        return actions.astype(int)

    def play(self):
        values = self.learn_values()
        actions = self.get_actions(values)
        s = self.env.reset()
        done = False
        step = 0
        while not done:
            self.env.render()
            a = actions[s]
            (s, r, done) = self.env.step(a)
            if done:
                s = self.env.reset()
            else:
                step += 1

    def plot(self):
        self.learn_values()


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
                self.A[s] = np.random.randint(0, 4)
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

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(1e3)):
        plot_list = []
        for episode in range(total_timesteps):
            list1 = self.rollout(epsilon_threshold=epsilon_threshold, alpha=alpha)
            plot_list.append(sum(list1))
            self.rew_plot_list.clear()

        return plot_list

    def play(self):
        s = self.env.reset()
        done = False
        step = 0
        while not done:
            self.env.render()
            a = np.argmax(self.Q[s])
            (s, r, done) = self.env.step(a)
            if done:
                s = self.env.reset()
            else:
                step += 1

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
                a = np.random.randint(0, 4)
                if a == np.argmax(self.Q[s]):
                    a = np.random.randint(0, 4)
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

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(1e3)):
        plot_list = []
        for episode in range(total_timesteps):
            list1 = self.rollout(epsilon_threshold=epsilon_threshold, alpha=alpha)
            plot_list.append(sum(list1))
            self.rew_plot_list.clear()

        return plot_list

    def play(self):
        s = self.env.reset()
        done = False
        step = 0
        while not done:
            self.env.render()
            a = np.argmax(self.Q[s])
            (s, r, done) = self.env.step(a)
            if done:
                s = self.env.reset()
            else:
                step += 1

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

    def learn(self, alpha, total_timesteps=int(1e3)):
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


if __name__ == '__main__':
    envs = gridworld.GridWorld()
    gamma = 0.95
    """
    mode = 0 activates ValueIteration
    mode = 1 activates PolicyIteration
    mode = 2 activates SARSA
    mode = 3 activates Q-Learning
    mode = 4 activates TD(0)
        submode = 0 activates SARSA
        submode = 1 activates Q-Learning
    """
    mode = 4
    submode = 1
    if mode == 0:
        policy = ValueIteration(envs, gamma)
        policy.plot()
    elif mode == 1:
        policy = PolicyIteration(envs, gamma)
        policy.plot()
    elif mode == 2:
        policy = SARSA(envs, gamma)
        policy.plot()
    elif mode == 3:
        policy = QLearning(envs, gamma)
        policy.plot()
    elif mode == 4:
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

import numpy as np
import gridworld
import matplotlib.pyplot as plt

"""
Initial hw1 for Scott Bout (bout2)
"""


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
        self.save_fig = False

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

        return self.actions

    def play(self):
        actions = self.learn_actions().astype(int)
        plt.plot(self.values_list_plot)
        plt.xlabel('iterations')
        plt.ylabel('values')
        plt.title('values over time')
        if self.save_fig:
            plt.savefig(f'figures/gridworld/policyiteration_example_learning_curve.png')
        plt.show()
        s = self.env.reset()
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }
        done = False
        step = 0
        while not done:
            # self.env.render()
            a = actions[s]
            (s, r, done) = self.env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            if done:
                s = self.env.reset()
            else:
                step += 1

        plt.plot(log['t'], log['s'])
        plt.plot(log['t'][:-1], log['a'])
        plt.plot(log['t'][:-1], log['r'])
        plt.legend(['s', 'a', 'r'])
        plt.title('trajectory example')
        if self.save_fig:
            plt.savefig(f'figures/gridworld/policyiteration_example_trajectory_gridworld.png')
        plt.show()

        aa = actions.reshape((5, 5))
        pp = np.full((5, 5), fill_value='_____')
        for i in range(pp.shape[0]):
            for j in range(pp.shape[1]):
                if int(aa[i][j]) == 0:
                    pp[i][j] = "Right"
                elif int(aa[i][j]) == 1:
                    pp[i][j] = 'Up'
                elif int(aa[i][j]) == 2:
                    pp[i][j] = "Left"
                else:
                    pp[i][j] = "Down"

        plt.imshow(aa, cmap="Reds")
        plt.title('Policy')
        for i in range(pp.shape[0]):
            for j in range(pp.shape[1]):
                plt.text(j, i, f'{pp[i][j]}', color='Black', ha='center', va='center')
        if self.save_fig:
            plt.savefig('figures/gridworld/policyiteration_example_policy_gridworld.png')
        plt.show()

        vv = self.values.reshape((5, 5))
        plt.imshow(vv, cmap='Blues')
        plt.title(f'State-Values')
        for i in range(vv.shape[0]):
            for j in range(vv.shape[0]):
                plt.text(j, i, f'{vv[i][j]:0.2f}', ha='center', va='center')
        if self.save_fig:
            plt.savefig(f'figures/gridworld/policyiteration_example_values_gridworld.png')
        plt.show()

    def plot(self, save_fig):
        self.save_fig = save_fig
        # self.learn_actions()
        self.play()


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
        self.save_fig = False

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
                # plt.plot(self.values_list_plot)
                # plt.show()
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
        plt.plot(self.values_list_plot)
        plt.title(f'Values over time')
        plt.xlabel(f'Iterations')
        plt.ylabel(f'Mean Value')
        if self.save_fig:
            plt.savefig(f'figures/gridworld/valueiteration_example_learning_curve_gridworld.png')
        plt.show()
        actions = self.get_actions(values)
        s = self.env.reset()
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }
        done = False
        step = 0
        while not done:
            # self.env.render()
            a = actions[s]
            (s, r, done) = self.env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            if done:
                s = self.env.reset()
            else:
                step += 1

        plt.plot(log['t'], log['s'])
        plt.plot(log['t'][:-1], log['a'])
        plt.plot(log['t'][:-1], log['r'])
        plt.legend(['s', 'a', 'r'])
        plt.title('trajectory example')
        plt.xlabel('timestep')
        plt.ylabel('state')
        if self.save_fig:
            plt.savefig('figures/gridworld/valueiteration_example_trajectory_gridworld.png')
        plt.show()

        aa = actions.reshape((5, 5))
        pp = np.full((5, 5), fill_value='_____')
        for i in range(pp.shape[0]):
            for j in range(pp.shape[1]):
                if int(aa[i][j]) == 0:
                    pp[i][j] = "Right"
                elif int(aa[i][j]) == 1:
                    pp[i][j] = 'Up'
                elif int(aa[i][j]) == 2:
                    pp[i][j] = "Left"
                else:
                    pp[i][j] = "Down"

        plt.imshow(aa, cmap="Reds")
        plt.title('Policy')
        for i in range(pp.shape[0]):
            for j in range(pp.shape[1]):
                plt.text(j, i, f'{pp[i][j]}', color='Black', ha='center', va='center')
        if self.save_fig:
            plt.savefig('figures/gridworld/valueiteration_example_policy_gridworld.png')
        plt.show()

        vv = values.reshape((5, 5))
        plt.imshow(vv, cmap='Blues')
        plt.title(f'State-Values')
        for i in range(vv.shape[0]):
            for j in range(vv.shape[0]):
                plt.text(j, i, f'{vv[i][j]:0.2f}', ha='center', va='center')
        if self.save_fig:
            plt.savefig(f'figures/gridworld/valueiteration_example_values_gridworld.png')
        plt.show()

    def plot(self, save_fig):
        self.save_fig = save_fig
        self.play()


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

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(5e3)):
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
            if done:
                s = self.env.reset()
            else:
                step += 1

        plt.plot(log['t'], log['s'])
        plt.plot(log['t'][:-1], log['a'])
        plt.plot(log['t'][:-1], log['r'])
        plt.legend(['s', 'a', 'r'])
        plt.title('trajectory example')
        if self.save_fig:
            plt.savefig('figures/gridworld/sarsa_example_trajectory_gridworld.png')
        plt.show()

    def plot(self, graph=0, save_fig=False):
        self.save_fig = save_fig
        if graph == 0:
            list1 = self.learn(epsilon_threshold=0.1, alpha=0.5)
            plt.plot(list1)
            self.reset_tables()

            list2 = self.learn(epsilon_threshold=0.5, alpha=0.5)
            plt.plot(list2)
            self.reset_tables()

            list3 = self.learn(epsilon_threshold=0.01, alpha=0.5)
            plt.plot(list3)
            plt.legend(['e = 0.1, alpha = 0.5', 'e = 0.5, alpha = 0.5', 'e = 0.01, alpha = 0.5'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying epsilon')
            if save_fig:
                plt.savefig('figures/gridworld/sarsa_example_varying_epsilon.png')
            plt.show()
            self.reset_tables()

        elif graph == 1:

            list4 = self.learn(epsilon_threshold=0.1, alpha=1e-5)
            plt.plot(list4)
            self.reset_tables()

            list5 = self.learn(epsilon_threshold=0.1, alpha=1e-2)
            plt.plot(list5)
            self.reset_tables()

            list6 = self.learn(epsilon_threshold=0.1, alpha=0.5)
            plt.plot(list6)
            plt.legend(['e = 0.1, alpha = 1e-5', 'e = 0.1, alpha = 1e-2', 'e = 0.1, alpha = 0.5'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying alpha')
            if save_fig:
                plt.savefig('figures/gridworld/sarsa_example_varying_alpha.png')
            plt.show()

            self.reset_tables()

        elif graph == 2:
            list1 = self.learn(epsilon_threshold=0.1, alpha=0.5)
            plt.plot(list1)
            plt.legend(['e = 0.1, alpha = 0.5'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes')
            if save_fig:
                plt.savefig(f'figures/gridworld/sarsa_example_learning_curve_gridworld.png')
            plt.show()
            self.play()
            p, sa = self.get_policy()
            # print(f'policy = {p.reshape((5, 5))} \n'
            #       f'state action = {sa.reshape((5, 5))}')
            p = p.reshape((5, 5))
            pp = np.full((5, 5), fill_value='_____')
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if int(p[i][j]) == 0:
                        pp[i][j] = 'Right'
                    elif int(p[i][j]) == 1:
                        pp[i][j] = "Up"
                    elif int(p[i][j]) == 2:
                        pp[i][j] = "Left"
                    else:
                        pp[i][j] = "Down"

            # print(f'pp = {pp}')
            sa = sa.reshape((5, 5))
            # fig, ax = plt.subplots()
            plt.imshow(sa, cmap='Blues')
            plt.title('State-action values')
            for i in range(sa.shape[0]):
                for j in range(sa.shape[1]):
                    plt.text(j, i, f'{sa[i][j]:0.2f}', color='Black', ha='center', va='center')
            if save_fig:
                plt.savefig('figures/gridworld/sarsa_example_state-action_gridworld.png')
            plt.show()

            plt.imshow(p, cmap="Reds")
            plt.title('Policy')
            for i in range(pp.shape[0]):
                for j in range(pp.shape[1]):
                    plt.text(j, i, f'{pp[i][j]}', color='Black', ha='center', va='center')
            if save_fig:
                plt.savefig('figures/gridworld/sarsa_example_policy_gridworld.png')
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

    def learn(self, epsilon_threshold, alpha, total_timesteps=int(5e3)):
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
            if done:
                s = self.env.reset()
            else:
                step += 1

        plt.plot(log['t'], log['s'])
        plt.plot(log['t'][:-1], log['a'])
        plt.plot(log['t'][:-1], log['r'])
        plt.legend(['s', 'a', 'r'])
        plt.title('trajectory example')
        plt.xlabel('timestep')
        plt.ylabel('state')
        if self.save_fig:
            plt.savefig(f'figures/gridworld/qlearning_example_trajectory_gridworld.png')
        plt.show()

    def plot(self, graph=2, save_fig=False):
        self.save_fig = save_fig
        if graph == 0:
            list1 = self.learn(epsilon_threshold=0.1, alpha=0.5)
            plt.plot(list1)
            self.reset_q_table()

            list2 = self.learn(epsilon_threshold=0.5, alpha=0.5)
            plt.plot(list2)
            self.reset_q_table()

            list3 = self.learn(epsilon_threshold=0.01, alpha=0.5)
            plt.plot(list3)
            plt.legend([f'e = 0.1, alpha = 0.5', 'e = 0.5, alpha = 0.5', 'e = 0.01, alpha = 0.5'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying epsilon')
            if save_fig:
                plt.savefig(f'figures/gridworld/qlearning_example_varying_epsilon_gridworld.png')
            plt.show()
            self.reset_q_table()

        elif graph == 1:

            list4 = self.learn(epsilon_threshold=0.1, alpha=1e-5)
            plt.plot(list4)
            self.reset_q_table()

            list5 = self.learn(epsilon_threshold=0.1, alpha=1e-2)
            plt.plot(list5)
            self.reset_q_table()

            list6 = self.learn(epsilon_threshold=0.1, alpha=0.5)
            plt.plot(list6)
            plt.legend(['e = 0.1, alpha = 1e-5', 'e = 0.1, alpha = 1e-2', 'e = 0.1, alpha = 0.5'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes for varying alpha')
            if save_fig:
                plt.savefig(f'figures/gridworld/qlearning_example_varying_alpha_gridworld.png')
            plt.show()

            self.reset_q_table()

        elif graph == 2:
            list1 = self.learn(epsilon_threshold=0.1, alpha=0.5)
            plt.plot(list1)
            plt.legend(['e = 0.1, alpha = 0.5'])
            plt.xlabel('episodes')
            plt.ylabel('sum of rewards')
            plt.title('return vs episodes')
            if save_fig:
                plt.savefig(f'figures/gridworld/qlearning_example_learning_curve_gridworld.png')
            plt.show()
            self.play()
            p, sa = self.get_policy()
            # print(f'policy = {p.reshape((5, 5))} \n'
            #       f'state action = {sa.reshape((5, 5))}')
            p = p.reshape((5, 5))
            pp = np.full((5, 5), fill_value='_____')
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if int(p[i][j]) == 0:
                        pp[i][j] = 'Right'
                    elif int(p[i][j]) == 1:
                        pp[i][j] = "Up"
                    elif int(p[i][j]) == 2:
                        pp[i][j] = "Left"
                    else:
                        pp[i][j] = "Down"

            # print(f'pp = {pp}')
            sa = sa.reshape((5, 5))
            # fig, ax = plt.subplots()
            plt.imshow(sa, cmap='Blues')
            plt.title('State-action values')
            for i in range(sa.shape[0]):
                for j in range(sa.shape[1]):
                    plt.text(j, i, f'{sa[i][j]:0.2f}', color='Black', ha='center', va='center')
            if save_fig:
                plt.savefig(f'figures/gridworld/qlearning_example_state-action_gridworld.png')
            plt.show()
            plt.imshow(p, cmap="Reds")
            plt.title('Policy')
            for i in range(pp.shape[0]):
                for j in range(pp.shape[1]):
                    plt.text(j, i, f'{pp[i][j]}', color='Black', ha='center', va='center')
            if save_fig:
                plt.savefig(f'figures/gridworld/qlearning_example_policy_gridworld.png')
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

    def learn(self, alpha, total_timesteps=int(1e4)):
        plot_list = []
        for episode in range(total_timesteps):
            list1 = self.rollout(alpha=alpha)
            plot_list.append(sum(list1))
            self.rew_plot_list.clear()

        return plot_list, self.values

    def plot(self, save_fig):
        l1, vals = self.learn(alpha=0.5)
        vals = vals.reshape((5, 5))
        # print(f'values = {vals.reshape((5, 5))}')
        plt.imshow(vals, cmap="Greens")
        plt.title('State-Values')
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                plt.text(j, i, f'{vals[i][j]:0.2f}', color='Black', ha='center', va='center')
        if save_fig:
            plt.savefig(f'figures/gridworld/qlearning_example_state-value_gridworld.png')
        plt.show()
        # plt.plot(l1)
        # plt.show()


def main(mode=0, submode=0):
    envs = gridworld.GridWorld()
    gamma = 0.95
    # mode = 2
    # submode = 1
    if mode == 0:
        policy = ValueIteration(envs, gamma)
        policy.plot(save_fig=False)
    elif mode == 1:
        policy = PolicyIteration(envs, gamma)
        policy.plot(save_fig=False)

    elif mode == 2:
        policy = SARSA(envs, gamma)
        policy.plot(graph=1, save_fig=False)

    elif mode == 3:
        policy = QLearning(envs, gamma)
        policy.plot(graph=1, save_fig=False)

    elif mode == 4:
        if submode == 0:
            policy = SARSA(envs, gamma)
            _ = policy.learn(epsilon_threshold=0.1, alpha=0.5)
            sarsa, _ = policy.get_policy()
            values = TDzero(envs, gamma, policy=sarsa)
            values.plot(save_fig=False)
        elif submode == 1:
            policy = QLearning(envs, gamma)
            _ = policy.learn(epsilon_threshold=0.1, alpha=0.5)
            qlearning, _ = policy.get_policy()
            values = TDzero(envs, gamma, policy=qlearning)
            values.plot(save_fig=False)


if __name__ == '__main__':
    """
       mode = 0 activates ValueIteration
       mode = 1 activates PolicyIteration
       mode = 2 activates SARSA
       mode = 3 activates Q-Learning
       mode = 4 activates TD(0)
           submode = 0 activates SARSA
           submode = 1 activates Q-Learning
       """
    main(mode=4, submode=1)

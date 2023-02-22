import numpy as np

import gridworld
import numpy

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
        self.values = np.zeros((5, 5))

    def learn(self, total_timesteps):
        s = self.env.reset()

        # rollout
        for i in range(total_timesteps):
            pass


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

            if np.allclose(self.old_values, self.values, rtol=0, atol=1e-20):
                break
            else:
                self.old_values = self.values.copy()
##
        return self.values

    def get_actions(self, values):
        v_list = np.zeros(self.env.num_actions)
        actions = np.zeros(self.env.num_states)
        for s in range(self.env.num_states):
            for a in range(self.env.num_actions):
                self.env.s = s
                (s1, _, _) = self.env.step(a)
                p = self.env.p(s1, s, a)
                v_list[a] = np.sum(p * values[s1])
            actions[s] = np.argmax(v_list)

        return actions.astype(int)

    def play(self):
        values = self.learn_values()
        actions = self.get_actions(values)
        # print(f'actions = {actions}')
        s = self.env.reset()
        DONE = False
        step = 0
        while not DONE:
            self.env.render()
            # print(f'state = {s}')
            a = actions[s]
            # print(f'action = {a}')
            (s, r, done) = self.env.step(a)
            # print(f'next state = {s}')
            if done:
                # print(f'step = {step}')
                s = self.env.reset()
                DONE = done
            else:
                step += 1


class SARSA:

    def __init__(self):
        pass


class QLearning:

    def __init__(self):
        pass


envs = gridworld.GridWorld()
gamma = 0.95
policy = ValueIteration(envs, gamma)
policy.play()
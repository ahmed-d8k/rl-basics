from src.agent import Agent
import numpy as np


class BasicAgent(Agent):
    def __init__(self, print_mode=False):
        self.last_action = None
        self.last_state = None
        self.step_size = 0.01
        self.discount = 0.95
        self.experience = {}
        self.experience[(0,0,0,0)] = np.random.randn(4)
        self.history = []
        self.print_mode = print_mode

    def toggle_print(self):
        if self.print_mode:
            self.print_mode = False
        else:
            self.print_mode = True

    def get_obs(self, state):
        return (round(state[2], 1), round(state[3], 1), round(state[4], 1), round(state[5], 1))

    def add_state_to_history(self, state, action):
        obs_s = self.get_obs(state)
        self.history.append((obs_s, action))

    def history_update(self, reward):
        for state_action in self.history:
            state = state_action[0]
            action = state_action[1]
            self.experience[state][action] += reward*self.step_size

    def discount_history_update(self, reward):
        total_steps = len(self.history)
        t = 0
        for state_action in self.history:
            t += 1
            disc_fac = self.discount**(total_steps - t - 1)
            state = state_action[0]
            action = state_action[1]
            self.experience[state][action] += reward*self.step_size
    
    def adjust_reward(self, old_reward, state, terminal=False):
        reward = old_reward
        # Safe landing reward
        if (state[6] and state[7]) and ((state[3] >= -0.2) and (state[3] < 0.05)) and terminal:
            if self.print_mode:
                print(f"Safe: vy={state[3]}, l1={state[6]}, l2={state[7]}")
                print(f"x={state[0]}")
            reward += 5000
            if (state[0] >= -0.2) and (state[0] <= 0.2):
                if self.print_mode:
                    print(f"Landed in goal as well")
        # low speed but one leg
        elif (state[6] or state[7]) and ((state[3] >= -0.2) and (state[3] < 0.05)) and terminal:
            if self.print_mode:
                print(f"ls_1l: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward -= 10
        # Both leg but high speed
        elif (state[6] and state[7]) and ((state[3] >= -0.6) and (state[3] < 0.05)) and terminal:
            if self.print_mode:
                print(f"hs_bl: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward -= 10
        # one leg and high speed
        elif (state[6] or state[7]) and ((state[3] >= -0.6) and (state[3] < 0.05)) and terminal:
            if self.print_mode:
                print(f"hs_1l: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward -= 30
        elif terminal:
            if self.print_mode:
                print(f"Crash: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward -= 5000
        return reward


    def softmax(self, x):
        exp_x = np.exp(x-np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def policy(self, state):
        s = self.get_obs(state)
        if s not in self.experience:
            self.experience[s] = np.random.randn(4)
        action = np.argmax(self.softmax(self.experience[s]))
        return action

    def update(self, reward):
        ls = self.get_obs(self.last_state)
        self.experience[ls][self.last_action] += reward*self.step_size

    def start_step(self):
        action = 0
        self.last_action = action
        self.last_state = [0,0,0,0,0,0,0,0]
        self.history = []
        return action

    def step(self, state, reward):
        reward = self.adjust_reward(reward, state)
        action = self.policy(state)
        self.update(reward)
        self.add_state_to_history(state, action)
        self.last_action = action
        self.last_state = state
        return action, reward

    def terminal_step(self, state, reward):
        reward = self.adjust_reward(reward, state, terminal=True)
        self.discount_history_update(reward)
        return reward
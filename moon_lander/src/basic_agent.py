from src.agent import Agent
import numpy as np
import src.lander_states as ls

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
        return (round(state[ls.vx_id], 1), round(state[ls.vy_id], 1), round(state[ls.angle_id], 1), round(state[ls.ang_vel_id], 1))

    def add_state_action_to_history(self, state, action):
        obs = self.get_obs(state)
        self.history.append((obs, action))

    def history_update(self, reward):
        for state_action in self.history:
            state = state_action[0]
            action = state_action[1]
            self.experience[state][action] += reward*self.step_size

    def discount_history_update(self, reward):
        total_steps = len(self.history)
        t = -1
        for state_action in self.history:
            t += 1
            retro_t = total_steps - t - 1
            disc_fac = self.discount**(retro_t)
            state = state_action[0]
            action = state_action[1]
            self.experience[state][action] += reward*self.step_size*disc_fac

    def reward_update(self, old_reward, state):
        reward = old_reward
        return reward
    
    def terminal_reward_update(self, old_reward, state, terminal=True):
        reward = old_reward
        if ls.safe_landing(state, terminal=terminal):
            if self.print_mode:
                print(f"Safe: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward += 5000
            if ls.lander_in_goal(state):
                if self.print_mode:
                    print(f"Landed in goal as well")
        elif ls.safe_speed_one_leg_landing(state, terminal=terminal):
            if self.print_mode:
                print(f"ls_1l: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward += 30
        elif ls.risky_speed_both_legs_landing(state, terminal=terminal):
            if self.print_mode:
                print(f"hs_bl: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward += 30
        elif ls.risky_speed_one_leg_landing(state, terminal=terminal):
            if self.print_mode:
                print(f"hs_1l: vy={state[3]}, l1={state[6]}, l2={state[7]}")
            reward += 10
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
        reward = self.reward_update(reward, state)
        action = self.policy(state)
        self.update(reward)
        self.add_state_action_to_history(state, action)
        self.last_action = action
        self.last_state = state
        return action, reward

    def terminal_step(self, state, reward):
        reward = self.terminal_reward_update(reward, state)
        self.discount_history_update(reward)
        return reward
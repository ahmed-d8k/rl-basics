x_id = 0
y_id = 1
vx_id = 2
vy_id = 3
angle_id = 4
ang_vel_id = 5
right_leg_contact_id = 6
left_leg_contact_id = 7
min_safe_speed = -0.2
min_risky_speed = -0.6
max_safe_speed = 0.05
goal_min_x = -0.2
goal_max_x = 0.2

def risky_speed_one_leg_landing(state, terminal=False):
    if single_leg_contact(state[right_leg_contact_id], state[left_leg_contact_id]) and risky_landing_speed(state[vy_id]) and terminal:
        return True
    else:
        return False

def risky_speed_both_legs_landing(state, terminal=False):
    if double_leg_contact(state[right_leg_contact_id], state[left_leg_contact_id]) and risky_landing_speed(state[vy_id]) and terminal:
        return True
    else:
        return False

def safe_speed_one_leg_landing(state, terminal=False):
    if single_leg_contact(state[right_leg_contact_id], state[left_leg_contact_id]) and safe_landing_speed(state[vy_id]) and terminal:
        return True
    else:
        return False

def safe_landing(state, terminal=False):
    if double_leg_contact(state[right_leg_contact_id], state[left_leg_contact_id]) and safe_landing_speed(state[vy_id]) and terminal:
        return True
    else:
        return False

def safe_landing_speed(vy):
    if (vy >= min_safe_speed) and (vy < max_safe_speed):
        return True
    else:
        return False

def risky_landing_speed(vy):
    if (vy >= min_risky_speed) and (vy < max_safe_speed):
        return True
    else:
        return False

def single_leg_contact(right_leg_contact, left_leg_contact):
    if right_leg_contact or left_leg_contact:
        return True
    else:
        return False

def double_leg_contact(right_leg_contact, left_leg_contact):
    if (right_leg_contact and left_leg_contact):
        return True
    else:
        return False

def lander_in_goal(state):
    if (state[x_id] >= goal_min_x) and (state[x_id] <= goal_max_x):
        return True
    else:
        return False

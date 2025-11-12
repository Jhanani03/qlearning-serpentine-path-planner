"""
Q-learning–Based Autonomous Path Planner
----------------------------------------
This script implements a Q-learning–based autonomous path planner that enables 
serpentine trajectory coverage across solar panel surfaces (approximated as a sinusoidal function).

Author: Jhanani
Date: November 2025
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

# Parameters
A = 1  # Amplitude of the sine wave
k = 1  # Frequency of the sine wave
x_min = 0  # Minimum x-coordinate
x_max = 10  # Maximum x-coordinate
y_min = -A  # Minimum y-coordinate
y_max = A  # Maximum y-coordinate
num_episodes = 200  # Increased number of episodes for more training
alpha = 0.05  # Reduced learning rate for more stable learning
gamma = 0.9  # Discount factor
epsilon = 0.05  # Decreased exploration rate to promote exploitation
actions = ['left', 'right', 'up', 'down']  # Actions: left, right, up, down

# Define the sinusoidal function using numpy
def sine_function(x):
    return A * np.sin(k * x)  # Changed to numpy sin for array support

# Define the reward function: negative distance from the sinusoidal path, but less negative
def reward_function(x, y):
    return -abs(y - sine_function(x)) + 0.5  # Added constant reward to offset negative values

# Initialize Q-table
x_steps = 10  # Reduced number of x steps for quicker training
y_steps = 10  # Reduced number of y steps for quicker training
q_table = np.zeros((x_steps, y_steps, len(actions)))  # Q-table: state (x, y) x actions

# Discretize the state space
def discretize_state(x, y):
    x_discretized = int((x - x_min) / (x_max - x_min) * (x_steps - 1))
    y_discretized = int((y - y_min) / (y_max - y_min) * (y_steps - 1))
    return x_discretized, y_discretized

# Choose an action using epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        return random.choice(range(len(actions)))
    else:
        # Exploitation: choose the action with the highest Q-value
        return np.argmax(q_table[state[0], state[1]])

# Perform an action and return the next state and reward
def perform_action(x, y, action):
    if action == 0:  # Move left
        x_new = max(x - 0.1, x_min)
        y_new = sine_function(x_new)
    elif action == 1:  # Move right
        x_new = min(x + 0.1, x_max)
        y_new = sine_function(x_new)
    elif action == 2:  # Move up
        y_new = min(y + 0.1, y_max)
        x_new = x
    elif action == 3:  # Move down
        y_new = max(y - 0.1, y_min)
        x_new = x

    reward = reward_function(x_new, y_new)
    state_new = discretize_state(x_new, y_new)

    return state_new, reward, x_new, y_new

# Maximum steps per episode to avoid running too long
max_steps_per_episode = 200  # Increased steps per episode

# Q-learning algorithm
def q_learning():
    for episode in range(num_episodes):
        x = random.uniform(x_min, x_max)  # Start at a random x position
        y = sine_function(x)  # Initial y position on the sine curve
        state = discretize_state(x, y)

        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            # Choose action
            action = choose_action(state)

            # Perform the action and get the next state and reward
            next_state, reward, x_new, y_new = perform_action(x, y, action)

            # Q-value update
            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action] += alpha * (
                reward + gamma * q_table[next_state[0], next_state[1], best_next_action] - q_table[state[0], state[1], action]
            )

            # Transition to the next state
            state = next_state
            x, y = x_new, y_new

            # Increment step count
            steps += 1

            # Print progress every 100 episodes
            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes}, X: {x}, Y: {y}, Steps: {steps}")

# Plotting the learned path
def plot_path():
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = sine_function(x_vals)  # Now using numpy array-friendly sine_function

    # Simulate the robot following the path based on the learned Q-values
    x = random.uniform(x_min, x_max)
    y = sine_function(x)
    robot_path_x = [x]
    robot_path_y = [y]

    for _ in range(100):
        state = discretize_state(x, y)
        action = np.argmax(q_table[state[0], state[1]])  # Choose the best action
        _, _, x_new, y_new = perform_action(x, y, action)
        robot_path_x.append(x_new)
        robot_path_y.append(y_new)
        x, y = x_new, y_new

    plt.plot(x_vals, y_vals, label="Sinusoidal Path", color="blue")
    plt.plot(robot_path_x, robot_path_y, label="Robot Path", color="red")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Robot Path Following Sinusoidal Curve")
    plt.show()

# Measure training time
start_time = time.time()

# Run Q-learning
q_learning()

end_time = time.time()

print(f"Training took {end_time - start_time} seconds.")

# Plot the learned path
plot_path()

# Print the Q-table after training
print("Trained Q-table:")
print(q_table)

# Visualize a slice of the Q-table (for a specific x or y value)
# Example: Show Q-values for all actions at state (x=5, y=5)
x_index = 5
y_index = 5
print(f"Q-values at state (x={x_index}, y={y_index}):")
print(q_table[x_index, y_index])

# Alternatively, visualize the Q-values for all states for a particular action
action_index = 0  # 0 corresponds to 'left', for example
plt.imshow(q_table[:, :, action_index], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title(f"Q-values for action {actions[action_index]} (0 = left)")
plt.xlabel("Y-discretized states")
plt.ylabel("X-discretized states")
plt.show()

# Save the Q-table to a file
np.save('q_table.npy', q_table)

# To load the Q-table later
q_table_loaded = np.load('q_table.npy')
print("Loaded Q-table:")
print(q_table_loaded)

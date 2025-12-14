"""
Privacy Attack in Ridesharing Services: Passenger Destination Inference with Dynamic Pricing
Implementation of Double DQN for timing decision in privacy attack scenario.

This module implements a reinforcement learning system to determine the optimal timing
for privacy attacks in ridesharing services, based on the paper:
"How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker's Perspective"

"""

import gc
import math
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
# Hyperparameters and configuration constants
MAX_TRAJ_LEN = 81  # Maximum trajectory length
D = 3000  # Distance threshold in meters (attack range)
TOPK = 5  # Number of top candidate destinations to consider
CLUSTER_NO = 82  # Target cluster number for attack
MODEL_NAME = "top-5_3000m"  # Model name for saving/loading

# Reward parameters
R_MISSED = -10  # Reward for missing a target (攻击失败)
R_FAILED = -10  # Reward for attacking a non-target (误攻击)
R_PASS = 5  # Reward for correctly passing a non-target (正确放弃)

# TensorBoard logging
log_dir = "tensorboard_logs/" + MODEL_NAME
summary_writer = tf.summary.create_file_writer(log_dir)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def pkl_gen(file_path):
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# Load cluster data (centroids of destination clusters)
cluster_data = pd.read_csv(
    "../processed_data/clusters_5_ring/mean_shift clustering/965_clusters_center_coords_with_grid_number.csv"
)[["grid_x", "grid_y"]].values.tolist()

# Initialize target cluster coordinates
cluster_x = -1
cluster_y = -1

# Load pre-computed prediction data
with open("predict_circle_" + str(D) + "m.pkl", "rb") as f:
    predict_circle = pickle.load(f)  # Trajectories within attack range

with open("predict_info_top5.pkl", "rb") as f:
    pred_model = pickle.load(f)  # Pre-trained CBAM model predictions

# Load test data (trajectories and true destinations)
test_data = pd.read_csv("../testing data/test_all_level8_new_clusters_idx.csv")[
    ["coords_of_traj", "dest_cluster"]
]
test_data["coords_of_traj"] = test_data["coords_of_traj"].apply(lambda x: eval(x))
data = test_data.values.tolist()
del test_data  # Free memory


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

class TrajEnv:
"""
    Reinforcement Learning Environment for trajectory-based attack timing.

    Simulates the decision-making process of an attacker trying to determine
    the optimal time to launch a privacy attack based on partial trajectory
Information and destination predictions.
    """

    def __init__(self):
        """
        Initialize the environment.
        """
        self.total_traj_len = None  # Total length of current trajectory
        self.traj_no = None  # Trajectory index
        self.traj_prefix = None  # Current prefix length
        self.y_true = None  # True destination cluster
        self.cur_x = None  # Current x coordinate
        self.cur_y = None  # Current y coordinate
        self.last_MD = None  # Last mean distance to target cluster

    def reset(self, traj):
        self.traj_no = traj[0]
        self.traj_prefix = traj[1]
        self.total_traj_len = len(data[self.traj_no][0])

        # Get current position
        t = data[self.traj_no][0][self.traj_prefix]
        self.cur_x = t[0]
        self.cur_y = t[1]
        self.y_true = data[self.traj_no][1]

        # Calculate mean distance to target cluster from top-K predictions
        y_pred = pred_model[self.traj_no][self.traj_prefix]
        total_distance = 0
        for p in y_pred:
            total_distance += math.sqrt(
                pow(cluster_data[p][0] - cluster_x, 2) +
                pow(cluster_data[p][1] - cluster_y, 2)
            )
        self.last_MD = total_distance / (TOPK * 255 * math.sqrt(2))

        # Normalize coordinates and create state vector
        normalized_x = np.array(t[0]) / pow(2, 8)
        normalized_y = np.array(t[1]) / pow(2, 8)
        state_vector = np.concatenate([normalized_x, normalized_y,
                                       np.array(self.last_MD).reshape(1)])

        return state_vector.reshape(3)

    def step(self, action, state):
        y_pred = pred_model[self.traj_no][self.traj_prefix]
        cur_MD = state[2]  # Current mean distance
        time = -1  # Initialize time remaining

        # Calculate distance to target cluster center
        distance_to_cluster = math.sqrt(
            pow(self.cur_x - cluster_x, 2) + pow(self.cur_y - cluster_y, 2)
        )

        # Reward logic based on scenario classification from paper
        if distance_to_cluster <= D / 105:  # Inside attack range
            # Target vehicle (true destination in predictions)
            if self.y_true in y_pred:
                # Vehicle has arrived at destination
                if self.traj_prefix >= self.total_traj_len - 1:
                    reward = R_MISSED  # Case 1 & 2: Missed attack
                    done = True
                else:
                    if action == 0:  # Wait
                        # Case 4: Continue observing
                        reward = 10 * (cur_MD - self.last_MD)
                        done = False
                    else:  # Attack
                        # Case 3: Successful attack
                        time = self.total_traj_len - self.traj_prefix - 1
                        reward = 10 + 0.5 * time
                        done = True
            else:  # Non-target vehicle
                if action == 1:  # Attack
                    # Case 5 & 7: Failed attack (wrong target)
                    reward = R_FAILED
                    done = True
                else:  # Wait
                    if self.traj_prefix >= self.total_traj_len - 1:
                        # Case 6: Correctly passed non-target
                        reward = R_PASS
                        done = True
                    else:
                        # Case 8: Continue observing non-target
                        reward = 10 * (cur_MD - self.last_MD)
                        done = False
        else:  # Outside attack range
            if action == 1:  # Attack (premature)
                # Case 9, 11, 13: Invalid attack location
                reward = R_MISSED if self.y_true in y_pred else R_FAILED
                done = True
            else:  # Wait
                if self.y_true in y_pred:
                    # Case 10: Target but too far
                    reward = 10 * (cur_MD - self.last_MD)
                    done = False
                else:
                    # Case 12 & 14: Non-target, too far
                    reward = R_PASS
                    done = True

        # Update last mean distance
        self.last_MD = cur_MD

        # Check if trajectory prediction data exhausted
        if self.traj_prefix >= len(pred_model[self.traj_no]) - 1:
            done = True

        # Move to next time step if not done
        if not done:
            self.traj_prefix += 1
            y_pred = pred_model[self.traj_no][self.traj_prefix]

            # Recalculate mean distance
            total_distance = 0
            for p in y_pred:
                total_distance += math.sqrt(
                    pow(cluster_data[p][0] - cluster_x, 2) +
                    pow(cluster_data[p][1] - cluster_y, 2)
                )
            cur_MD = total_distance / (TOPK * 255 * math.sqrt(2))

            # Get next position
            t = data[self.traj_no][0][self.traj_prefix]
        else:
            t = data[self.traj_no][0][self.traj_prefix]

        # Construct next state
        normalized_x = np.array(t[0]) / pow(2, 8)
        normalized_y = np.array(t[1]) / pow(2, 8)
        next_state = np.concatenate([normalized_x, normalized_y,
                                     np.array(cur_MD).reshape(1)]).reshape(3)

        return next_state, reward, done, time


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.

    Used to break temporal correlations in sequential experience data,
    improving training stability.
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        """Return current buffer size."""
        return len(self.memory)


# ============================================================================
# DQN NETWORK
# ============================================================================

class DQNNetwork:
    """
    Deep Q-Network for action-value function approximation.

    Implements both online and target networks for stable training.
    """

    def __init__(self, action_dim, load=False):
        self.action_dim = action_dim

        if load:
            self.load_model("DQN decision models/" + MODEL_NAME)
        else:
            self.model = self._build_model(action_dim)
            self.target_model = self._build_model(action_dim)
            self.update_target_network()

    def _build_model(self, action_dim):
        model = Sequential([
            # Input layer: state vector of size 3 [x, y, mean_distance]
            Dense(64, activation='relu', input_shape=(3,)),
            Dense(64, activation='relu'),
            # Output layer: Q-values for each action
            Dense(action_dim)
        ])

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
        try:
            self.model = tf.keras.models.load_model(model_path)
            # Clone for target network
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            print(f"✓ Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def update_target_network(self):
        """Synchronize target network weights with online network."""
        self.target_model.set_weights(self.model.get_weights())

    def train_step(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            # Forward pass
            q_values = self.model(states)

            # Select Q-values for taken actions
            mask = tf.one_hot(actions, self.model.output.shape[1])
            q_values_selected = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)

            # Calculate MSE loss
            loss = tf.reduce_mean(tf.square(q_targets - q_values_selected))

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Gradient clipping for stability
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

        # Apply gradients
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        return loss.numpy()


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    """
    Deep Q-Learning Agent with Double DQN and experience replay.

    Implements ε-greedy policy for exploration and learns optimal
    attack timing through interaction with the environment.
    """

    def __init__(self, action_size, load=False):
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.998  # Exploration decay rate
        self.batch_size = 64  # Training batch size
        self.update_target_freq = 1000  # Target network update frequency

        # Experience replay buffer
        self.memory = ReplayBuffer(3000)

        # Initialize networks
        if load:
            self.network = DQNNetwork(action_size, load=True)
        else:
            self.network = DQNNetwork(action_size)

        # Training counter
        self.steps_done = 0

    def select_action(self, state, test=False):
        # Exploration: random action
        if not test and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        # Exploitation: greedy action
        else:
            state_batch = np.expand_dims(state, axis=0)
            q_values = self.network.model.predict(state_batch, verbose=0)[0]
            return np.argmax(q_values)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def DDQN_learn(self):
        # Check if enough experience collected
        if len(self.memory) < self.batch_size:
            return 0

        # Sample random batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Double DQN: online network selects, target network evaluates
        # 1. Select best actions using online network
        next_actions = np.argmax(
            self.network.model.predict(next_states, verbose=0), axis=1
        )

        # 2. Evaluate Q-values using target network
        next_q_values = self.network.target_model.predict(next_states, verbose=0)

        # 3. Get Q-values for selected actions
        double_q = np.array([
            next_q_values[i, action] for i, action in enumerate(next_actions)
        ])

        # 4. Compute target Q-values
        targets = rewards + (1 - dones) * self.gamma * double_q

        # 5. Update online network
        loss = self.network.train_step(states, actions, targets)

        return loss

    def update_target_network(self):
        self.network.update_target_network()


# ============================================================================
# TEST AGENT
# ============================================================================

class TestAgent:
    def __init__(self, model):
        if isinstance(model, str):
            self.model = tf.keras.models.load_model(model)
        else:
            self.model = model

    def select_action(self, state):
        state_batch = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_batch, verbose=0)[0]
        return np.argmax(q_values)


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def save_model(agent, filepath="dqn_model"):
    agent.network.model.save(filepath)
    print(f"✓ Model saved to {filepath}")


def save_weights(agent, filepath="dqn_weights.h5"):
    agent.network.model.save_weights(filepath)
    print(f"✓ Model weights saved to {filepath}")


def load_model(filepath="model/dqn_model"):
    loaded_model = tf.keras.models.load_model(filepath)
    print(f"✓ Model loaded from {filepath}")
    return loaded_model


def load_weights(agent, filepath="dqn_weights.h5"):
    agent.network.model.load_weights(filepath)
    agent.network.target_model.load_weights(filepath)
    print(f"✓ Model weights loaded from {filepath}")
    return agent


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train(agent, env):
    scores = []  # Store episode rewards
    i_episode = 0  # Episode counter

    # Train on specific cluster (for now, single cluster)
    for cluster_no in range(CLUSTER_NO, CLUSTER_NO + 1):
        global cluster_x, cluster_y
        cluster_x = cluster_data[cluster_no][0]
        cluster_y = cluster_data[cluster_no][1]

        count_traj = 0
        traj_total = len(predict_circle[cluster_no])

        # Iterate through trajectories in the cluster
        for traj in predict_circle[cluster_no]:
            state = env.reset(traj)
            episode_reward = 0
            episode_loss = 0
            steps = 0

            done = False
            # Episode loop
            while not done:
                # Select and execute action
                action = agent.select_action(state, test=False)
                next_state, reward, done, _ = env.step(action, state)

                # Store experience
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state

                # Learn from experience
                loss = agent.DDQN_learn()
                if loss:
                    episode_loss += loss

                # Periodic target network update
                if agent.steps_done % agent.update_target_freq == 0:
                    agent.update_target_network()

                agent.steps_done += 1
                episode_reward += reward
                steps += 1

            # Decay exploration rate
            agent.update_epsilon()

            # Record metrics
            scores.append(episode_reward)
            with summary_writer.as_default():
                tf.summary.scalar("reward", episode_reward, step=i_episode)
                tf.summary.scalar(
                    "prefix_percentage",
                    env.traj_prefix / env.total_traj_len,
                    step=i_episode
                )
                summary_writer.flush()

            # Progress logging
            if i_episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_loss = episode_loss / steps if steps > 0 else 0
                print(
                    f"Cluster {cluster_no}, Traj {count_traj}/{traj_total}, "
                    f"Episode {i_episode}, Score: {episode_reward:.2f}, "
                    f"Avg Score: {avg_score:.2f}, Loss: {avg_loss:.4f}, "
                    f"Epsilon: {agent.epsilon:.3f}, Steps: {steps}"
                )

            i_episode += 1
            count_traj += 1

        # Clean up memory
        gc.collect()

    return scores


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_env(agent=None, env=None, load=False):
    if load:
        env = TrajEnv()
        agent = TestAgent(load_model("DQN decision models/" + MODEL_NAME))

    # Evaluation metrics
    scores = []  # Episode rewards
    TP = 0  # True Positives: successful attacks on targets
    FP = 0  # False Positives: attacks on non-targets
    MISS = 0  # Missed targets
    TIME = []  # Time remaining after successful attacks
    STEPS = []  # Steps per episode
    PERCENT = []  # Prefix percentage at attack time

    i_episode = 0
    ALL_TARGET = 0  # Total number of target vehicles

    # Test on specific cluster
    for cluster_no in range(CLUSTER_NO, CLUSTER_NO + 1):
        global cluster_x, cluster_y
        cluster_x = cluster_data[cluster_no][0]
        cluster_y = cluster_data[cluster_no][1]

        for traj in predict_circle[cluster_no]:
            state = env.reset(traj)
            episode_reward = 0
            steps = 0

            done = False
            while not done:
                # Greedy action selection (no exploration)
                action = agent.select_action(state)
                next_state, reward, done, time = env.step(action, state)
                state = next_state

                # Record results when episode ends
                if done:
                    if reward >= 10:  # Successful attack
                        TP += 1
                        PERCENT.append(env.traj_prefix / env.total_traj_len)
                        TIME.append(time)
                    elif reward == R_FAILED:  # False positive
                        FP += 1
                        PERCENT.append(env.traj_prefix / env.total_traj_len)
                    elif reward == R_MISSED:  # Missed target
                        MISS += 1

                episode_reward += reward
                steps += 1

            # Record episode metrics
            STEPS.append(steps)
            scores.append(episode_reward)
            i_episode += 1

            print(
                f"Test #{i_episode} completed: "
                f"Total reward={episode_reward:.2f}, Steps={steps}"
            )

    # Calculate statistics
    avg_reward = np.mean(scores)
    ALL_TARGET = TP + MISS

    # Print comprehensive evaluation
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"Missed Targets (MISS): {MISS}")
    print(f"Total Targets: {ALL_TARGET}")
    print(f"Successful Attacks: {len(TIME)}")
    print("-" * 60)
    print(f"Accuracy (TP/(TP+FP)): {TP / (TP + FP):.3f}")
    print(f"False Positive Rate: {FP / (TP + FP):.3f}")
    print(f"Coverage Rate (TP/ALL_TARGET): {TP / ALL_TARGET:.3f}")
    print(f"Miss Rate: {MISS / ALL_TARGET:.3f}")
    print(f"Avg Remaining Time: {np.mean(TIME):.1f} steps")
    print(f"Avg Steps per Episode: {np.mean(STEPS):.1f}")
    print(f"Avg Prefix Percentage: {np.mean(PERCENT):.3f}")
    print("=" * 60)

    return {
        'avg_reward': avg_reward,
        'TP': TP, 'FP': FP, 'MISS': MISS,
        'accuracy': TP / (TP + FP),
        'coverage': TP / ALL_TARGET,
        'avg_remaining_time': np.mean(TIME),
        'avg_steps': np.mean(STEPS),
        'avg_prefix_percent': np.mean(PERCENT)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("PRIVACY ATTACK TIMING WITH DEEP REINFORCEMENT LEARNING")
    print("=" * 60)

    # Initialize environment and agent
    env = TrajEnv()
    agent = DQNAgent(action_size=2, load=False)

    print("Starting training...")
    print("-" * 60)

    # Train agent
    scores = train(agent, env)

    # Save trained model
    save_model(agent, "DQN decision models/" + MODEL_NAME)
    with open("DQN decision models/" + MODEL_NAME + "_scores.pkl", "wb") as f:
        pickle.dump(scores, f)

    # Visualize learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.6)
    plt.title("DQN Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.show()

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    test_env(agent, env, load=False)


if __name__ == "__main__":
    # Uncomment to run training
    # main()

    # Run testing with pre-trained model
    test_env(load=True)

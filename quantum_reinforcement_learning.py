# Update package resources to account for version changes.
import importlib, pkg_resources
from gym.envs.toy_text.frozen_lake import generate_random_map
importlib.reload(pkg_resources)

import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')
from PIL import Image

# desc_map = [
#     "SFFF",
#     "FHFH",
#     "FFFH",
#     "HFFG"
# ]

# desc_map = [
#     "SFFFF",
#     "FHFHF",
#     "FFFHF",
#     "HFFFF",
#     "FFHFG"
# ]

# desc_map = [
#     "SFFFFF",
#     "FHFHFF",
#     "FFFHFF",
#     "HFFFFH",
#     "FFFFFF",
#     "FFHFFG"
# ]

desc_map = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# desc_map = generate_random_map(size = 6)

is_slippery = True

def save_gif_policy(model, config, text):
    env = gym.make(config["env_name"], render_mode="rgb_array")
    state, _ = env.reset()
    state = state_convertion(state, n_qubits)
    episode_reward = 0
    frames = []
    while True:
        im = Image.fromarray(env.render())
        frames.append(im)
        policy = model([tf.convert_to_tensor([state/config["state_bounds"]])])
        action = np.random.choice(n_actions, p=policy.numpy()[0])
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        state = state_convertion(state, n_qubits)
        if terminated or truncated:
            print(episode_reward)
            break
    env.close()
    frames[1].save(f'./images/{config["env_name"]}_policy_{text}.gif',
                save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)

def save_gif_Q(model, config, text):
    env = gym.make(config["env_name"], desc = desc_map, is_slippery = is_slippery, render_mode="rgb_array")
    state, _ = env.reset()
    state = state_convertion(state, n_qubits)
    episode_reward = 0
    frames = []
    while True:
        im = Image.fromarray(env.render())
        frames.append(im)
        q_vals = model([tf.convert_to_tensor([state/config["state_bounds"]])])
        action = int(tf.argmax(q_vals[0]).numpy())
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        state = state_convertion(state, n_qubits)
        if terminated or truncated:
            break
    env.close()
    frames[1].save(f'./images/{config["env_name"]}_Q_{text}_{len(desc_map)}x{len(desc_map[0])}_{is_slippery}.gif',
                save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)
    
    return episode_reward

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)
    
    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))
    
    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))
    
    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))
    
    return circuit, list(params.flat), list(inputs.flat)

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )
        
        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )
        
        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        
        return self.computation_layer([tiled_up_circuits, joined_vars])
    
class Alternating(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Alternating, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))
    
def generate_model_policy(qubits, n_layers, n_actions, beta, observables):
    """Generates a Keras model for a data re-uploading PQC policy."""

    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables)([input_tensor])
    process = tf.keras.Sequential([
        Alternating(n_actions),
        tf.keras.layers.Lambda(lambda x: x * beta),
        tf.keras.layers.Softmax()
    ], name="observables-policy")
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model

def gather_episodes(state_bounds, n_actions, model, n_episodes, env_name):
    """Interact with environment in batched fashion."""

    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    states = [state_convertion(e.reset()[0], n_qubits) for e in envs]
    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        # Compute policy for all unfinished envs in parallel
        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        # Store action and transition all environments to the next state
        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, terminated, truncated, _ = envs[i].step(action)
            states[i] = state_convertion(states[i], n_qubits)
            done[i] = terminated or truncated
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    return trajectories

def compute_returns(rewards_history, gamma):
    """Compute discounted returns with discount factor `gamma`."""
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()
    
    return returns

@tf.function
def reinforce_update(states, actions, returns, model, config):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs * returns) / config["batch_size"]
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([config["optimizer_in"], config["optimizer_var"], config["optimizer_out"]], [config["w_in"], config["w_var"], config["w_out"]]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

def generate_model_Qlearning(qubits, n_layers, n_actions, observables, target):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Alternating(len(observables))], name=target*"Target"+"Q-values")
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

    return model

def interact_env(state_bounds, state, model, epsilon, n_actions, env):
    # Preprocess state
    state_array = np.array(state) 
    state = tf.convert_to_tensor([state_array])

    # Sample action
    coin = np.random.random()
    if coin > epsilon:
        q_vals = model([state])
        action = int(tf.argmax(q_vals[0]).numpy())
    else:
        action = np.random.choice(n_actions)

    # Apply sampled action in the environment, receive reward and next state
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = state_convertion(next_state, n_qubits)
    next_state /= state_bounds
    done = terminated or truncated
    
    interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                   'reward': reward, 'done':np.float32(done)}
    
    return interaction

@tf.function
def Q_learning_update(states, actions, rewards, next_states, done, model, model_target, gamma, n_actions, config):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    rewards = tf.convert_to_tensor(rewards)
    next_states = tf.convert_to_tensor(next_states)
    done = tf.convert_to_tensor(done)

    # Compute their target q_values and the masks on sampled actions
    future_rewards = model_target([next_states])
    target_q_values = rewards + (gamma * tf.reduce_max(future_rewards, axis=1)
                                                   * (1.0 - done))
    masks = tf.one_hot(actions, n_actions)

    # Train the model on the states and target Q-values
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([states])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip([config["optimizer_in"], config["optimizer_var"], config["optimizer_out"]], [config["w_in"], config["w_var"], config["w_out"]]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

def train_model_policy(config):
    model = generate_model_policy(qubits, n_layers, n_actions, 1.0, observables)
    
    save_gif_policy(model, config, "start")

    # Start training the agent
    episode_reward_history = []
    for batch in range(config["n_episodes"] // config["batch_size"]):
        # Gather episodes
        episodes = gather_episodes(config["state_bounds"], n_actions, model, config["batch_size"], config["env_name"])
        
        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([compute_returns(ep_rwds, config["gamma"]) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])
        
        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model, config)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))
            
        avg_rewards = np.mean(episode_reward_history[-10:])

        print('Finished episode', (batch + 1) * config["batch_size"],
            'Average rewards: ', avg_rewards)
        
        if avg_rewards >= config["terminate_reward"]:
            break

    plt.figure(figsize=(10,5))
    plt.plot(episode_reward_history)
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.savefig(f'./images/{config["env_name"]}_policy.png')
    
    save_gif_policy(model, config, "end")

def train_model_Q(config):
    model = generate_model_Qlearning(qubits, n_layers, n_actions, observables, False)
    model_target = generate_model_Qlearning(qubits, n_layers, n_actions, observables, True)
    model_target.set_weights(model.get_weights())
    
    save_gif_Q(model, config, "start")
    env = gym.make(config["env_name"], desc = desc_map, is_slippery = is_slippery)
    replay_memory = deque(maxlen=config["max_memory_length"])
    
    episode_reward_history = []
    step_count = 0
    epsilon = config["epsilon"]
    for episode in range(config["n_episodes"]):
        episode_reward = 0
        state, _ = env.reset()
        state = state_convertion(state, n_qubits)
        state /= config["state_bounds"]
        
        while True:
            # Interact with env
            interaction = interact_env(config["state_bounds"], state, model, epsilon, n_actions, env)
            
            # Store interaction in the replay memory
            replay_memory.append(interaction)
            
            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1
            
            # Update model
            if step_count % config["steps_per_update"] == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(replay_memory, size=config["batch_size"])
                Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                model, model_target, gamma, n_actions, config)
            
            # Update target model
            if step_count % config["steps_per_target_update"] == 0:
                model_target.set_weights(model.get_weights())
            
            # Check if the episode is finished
            if interaction['done']:
                break

        # Decay epsilon
        epsilon = max(epsilon * config["decay_epsilon"], config["epsilon_min"])
        episode_reward_history.append(episode_reward)
        if (episode+1) % 10 == 0:
            avg_rewards = np.mean(episode_reward_history[-10:])
            print("Episode {}/{}, average last 10 rewards {}".format(
                episode+1, config["n_episodes"], avg_rewards))
            if avg_rewards >= config["terminate_reward"]:
                break
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_reward_history)
    plt.xlabel('Epsiode')
    plt.ylabel('Collected rewards')
    plt.savefig(f'./images/{config["env_name"]}_Q_{len(desc_map)}x{len(desc_map[0])}_{is_slippery}.png')
    
    episode_reward = 0
    while episode_reward == 0:
        episode_reward = save_gif_Q(model, config, "end")
    
def state_convertion(state, num):
    if isinstance(state, int):
        result = []
        for i in range(num):
            result.append(state % 2)
            state //= 2
        return list(reversed(result))
    else:
        return state
    

if __name__ == "__main__":
    
    n_qubits = 6 # Dimension of the state vectors in CartPole
    n_layers = 15 # Number of layers in the PQC
    n_actions = 4 # Number of actions in CartPole

    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    # observables = [ops[0] * ops[1], ops[2] * ops[3]]
    observables = [ops[0], ops[1], ops[2], ops[3]]
    
    # env_name = "CartPole-v1"
    env_name = "FrozenLake-v1"
    
    # state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    state_bounds = np.array([1, 1, 1, 1, 1, 1])
    gamma = 1
    terminate_reward = 0.9
    
    config = {
        "env_name": env_name,
        "state_bounds": state_bounds,
        "gamma": gamma,
        "terminate_reward": terminate_reward,
        "batch_size": 10,
        "n_episodes": 10000,
        "optimizer_in": tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True),
        "optimizer_var": tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True),
        "optimizer_out": tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True),
        "w_in": 1, 
        "w_var": 0, 
        "w_out": 2,
    }
    
    # train_model_policy(config)
    
    config = {
        "env_name": env_name,
        "state_bounds": state_bounds,
        "gamma": gamma,
        "terminate_reward": terminate_reward,
        "n_episodes": 2000,
        "max_memory_length": 10000,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "decay_epsilon": 0.99,
        "batch_size": 16,
        "steps_per_update": 5,
        "steps_per_target_update": 10,
        "optimizer_in": tf.keras.optimizers.Adam(learning_rate=0.00, amsgrad=True),
        "optimizer_var": tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
        "optimizer_out": tf.keras.optimizers.Adam(learning_rate=0., amsgrad=True),
        "w_in": 1, 
        "w_var": 0, 
        "w_out": 2,
    }
    
    train_model_Q(config)
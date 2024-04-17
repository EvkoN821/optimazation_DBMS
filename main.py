import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from environment import MySQLEnv
from knobs_def import knobs_default, knobs_max, knobs_min
import ctypes
import sys
import psutil
from elevate import elevate


elevate()
knobs_default = knobs_default
knobs_max = knobs_max
knobs_min = knobs_min

n_states = 74
n_actions = len(knobs_default)#98
knobs_min_list = list(knobs_min.values())
knobs_max_list = list(knobs_max.values())
knobs_default_list = list(knobs_default.values())
knobs_names = list(knobs_default.keys())
env = MySQLEnv(knobs_names, knobs_default_list)
env.init()


class ParameterNoise(keras.layers.Layer):
    def __init__(self, units):
        super(ParameterNoise, self).__init__()
        self.units = units
        self.sigma_init_value = 0.05

    def build(self, input_shape):
        w_init = tf.random_uniform_initializer(-math.sqrt(3 / self.units), math.sqrt(3 / self.units))
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units)), trainable=True)
        b_init = tf.random_uniform_initializer(-math.sqrt(3 / self.units), math.sqrt(3 / self.units))
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,)), trainable=True)

        sigma_init = tf.keras.initializers.Constant(value=self.sigma_init_value)
        self.sigma_w = tf.Variable(initial_value=sigma_init(shape=(input_shape[-1], self.units)), trainable=True)
        self.sigma_b = tf.Variable(initial_value=sigma_init(shape=(self.units,)), trainable=True)
        self.epsilon_w = tf.Variable(initial_value=tf.zeros((input_shape[-1], self.units)), trainable=False)
        self.epsilon_b = tf.Variable(initial_value=tf.zeros((self.units,)), trainable=False)

    def call(self, inputs):
        return tf.matmul(inputs, self.w+self.sigma_w*self.epsilon_w) + (self.b+self.sigma_b*self.epsilon_b)

    def sample_noise(self):
        self.epsilon_w = tf.random.uniform(shape=(self.units[-1], self.units))
        self.epsilon_b = tf.random.uniform(shape=(self.units,))


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, capacity=100000, batch_size=16):
        self.capacity = capacity
        self.batch_size = batch_size
        self.counter = 0
        self.state_buffer = np.zeros((self.capacity, n_states))
        self.action_buffer = np.zeros((self.capacity, n_actions))
        self.latency_buffer = np.zeros((self.capacity, 1))
        self.reward_buffer = np.zeros((self.capacity, 1))
        self.next_state_buffer = np.zeros((self.capacity, n_states))

    def record(self, observation):
        index = self.counter % self.capacity
        self.state_buffer[index] = observation[0]
        self.action_buffer[index] = observation[1]
        self.reward_buffer[index] = observation[2]
        self.next_state_buffer[index] = observation[3]
        self.latency_buffer[index] = observation[4]
        self.counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)

            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))


    def learn(self):

        record_range = min(self.counter, self.capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights, weights, tau):
    # print(f"{target_weights = } \n {weights = }")
    list_weight = []
    # print(f"\n\n {type(target_weights) = } \n")
    for (a, b) in zip(target_weights, weights):
        # print(f"\n{a = },\n {b = }")
        # flag = input()
        if a.dtype == tf.float32:
            a = tf.convert_to_tensor(a, dtype=tf.float32)
            das = b * tau + a * (1 - tau)
            # a.assign(b * tau + a * (1 - tau))
            # print(f"{das = }")
            # a = das
            list_weight.append(das)
        else:
            list_weight.append(a)
        # tf.compat.v1.assign(a, b * tau + a * (1 - tau))

        # a.assing(tf.multiply(b, tau) + tf.multiply(a,(1-tau)))

        # for i in range(len(a)):
        #     a[i] = b[i]*tau + a[i]*(1-tau)
        # list_weight.append(a)
    return list_weight


def get_model_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # print(f"\n{last_init = }\n\n")
    inputs = keras.Input(shape=(n_states, ))

    x = keras.layers.Dense(units=128)(inputs)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(units=128)(x)
    x = keras.activations.tanh(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(units=64)(x)
    x = keras.activations.tanh(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.BatchNormalization()(x)
    # outputs = keras.layers.Dense(units=n_actions, activation="sigmoid")(x) #, kernel_initializer=last_init)(x)
    outputs = keras.layers.Dense(units=n_actions, activation="sigmoid", kernel_initializer=last_init)(x)
    # print(f"\n{outputs = }\n\n")
    outputs = ParameterNoise(units=n_actions)(outputs)
    # print(f"\n{outputs = }\n\n")
    # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # print(inputs, outputs)
    return keras.Model(inputs, outputs, name="actor")


def get_model_critic():
    state_input = keras.layers.Input(shape=(n_states, ))
    x_s = keras.layers.Dense(units=128)(state_input)

    action_input = keras.layers.Input(shape=(n_actions, ))
    x_a = keras.layers.Dense(units=128)(action_input)

    concat = keras.layers.Concatenate()([x_s, x_a])

    x = keras.layers.Dense(units=256)(concat)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(units=64)(x)
    x = keras.activations.tanh(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(units=1, activation="tanh")(x)
    # print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    # print([state_input, action_input], outputs)
    return keras.Model([state_input, action_input], outputs, name="critic")


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise
    legal_actions = np.clip(sampled_actions, knobs_min_list, knobs_max_list)
    return np.squeeze(legal_actions)


if __name__ == "__main__":
    print("start")

    elevate()
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_model_actor()
    critic_model = get_model_critic()

    target_actor = get_model_actor()
    target_critic = get_model_critic()

    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    critic_lr = 0.002
    actor_lr = 0.001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    total_episodes = 350
    total_steps = 4

    gamma = 0.99

    tau = 0.005

    buffer = Buffer(50000, 16)

    ep_reward_list = []
    avg_reward_list = []
    last_actions = []

    for ep in range(total_episodes):

        prev_state = env.get_internal_metrics()
        episodic_reward = 0

        print(f"-----episode_{ep}-----")
        for i in range(total_steps):
            print("--step_" + str(i) + "--")
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            actions = policy(tf_prev_state, ou_noise)
            last_actions = actions
            print(f"{actions = }")
            state, reward, done, info = env.step(actions)
            print(f"{state =}")
            print(f"latency = {info}")
            buffer.record((prev_state, actions, reward, state, info))
            episodic_reward += reward

            buffer.learn()


            target_actor.set_weights(update_target(target_actor.get_weights(), actor_model.get_weights(), tau))
            target_critic.set_weights(update_target(target_critic.get_weights(), critic_model.get_weights(), tau))


            prev_state = state

        ep_reward_list.append(episodic_reward)

        print(f"Episode {ep}:  Avg Reward = {episodic_reward}")

    with open("output.csv", "w") as f:
        f.write('latency'+";"+';'.join(knobs_names) + "\n")
        for i in range(total_steps*total_episodes):
            f.write(str(buffer.latency_buffer[i][0])+";" + ';'.join(map(str, buffer.action_buffer[i])) + "\n")
    print('gg')
    input("press enter to exist")
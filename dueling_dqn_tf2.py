import gym
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter
from itertools import count
from collections import deque
import time
from tensorflow.keras import Model, layers, optimizers
import random


class Memory(object):
	def __init__(self, memory_size: int) -> None:
		self.memory_size = memory_size
		self.buffer = deque(maxlen=self.memory_size)

	def add(self, experience) -> None:
		self.buffer.append(experience)

	def size(self):
		return len(self.buffer)

	def sample(self, batch_size: int, continuous: bool = True):
		if batch_size > len(self.buffer):
			batch_size = len(self.buffer)
		if continuous:
			rand = random.randint(0, len(self.buffer) - batch_size)
			return [self.buffer[i] for i in range(rand, rand + batch_size)]
		else:
			indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
			return [self.buffer[i] for i in indexes]

	def clear(self):
		self.buffer.clear()


class Qnetwork(Model):
	def __init__(self):
		super(Qnetwork, self).__init__()

		self.fc1 = layers.Dense(64, activation=tf.nn.relu)
		self.fc_value = layers.Dense(256, activation=tf.nn.relu)
		self.fc_adv = layers.Dense(256, activation=tf.nn.relu)

		self.value = layers.Dense(1)
		self.adv = layers.Dense(2)


	def call(self, x):
		x = self.fc1(x)
		value = self.fc_value(x)
		adv = self.fc_adv(x)

		value = self.value(value)
		adv = self.adv(adv)

		advAverage = tf.reduce_mean(adv, axis=1, keepdims=True)
		Q = value + adv - advAverage
		return Q


	def select_action(self, x):
		x = x[np.newaxis, :]
		Q = self.call(x)
		a = tf.argmax(Q, axis=1)
		a = int(np.squeeze(a))
		return a


GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 16

UPDATE_STEPS = 4

if __name__ == '__main__':
	tf.keras.backend.set_floatx('float64')

	memory_replay = Memory(REPLAY_MEMORY)

	epsilon = INITIAL_EPSILON
	learn_steps = 0

	writer = SummaryWriter('ddqn-tf2')
	begin_learn = False
	episode_reward = 0

	online_q = Qnetwork()
	target_q = Qnetwork()
	target_q.set_weights(online_q.get_weights())

	env = gym.make('CartPole-v0')
	state = env.reset()
	
	mse = tf.keras.losses.MeanSquaredError()
	optim = optimizers.Adam(1e-4)
	for epoch in count():
		state = env.reset()
		episode_reward = 0
		for time_steps in range(200):
			p = random.random()
			if p < epsilon:
				action = random.randint(0, 1)
			else:
				action = online_q.select_action(state)
			next_state, reward, done, _ = env.step(action)
			episode_reward += reward
			memory_replay.add((state, next_state, action, reward, done))
			if memory_replay.size() > 128:
				if begin_learn is False:
					print('learn begin!')
					begin_learn = True
				learn_steps += 1
				if learn_steps % UPDATE_STEPS == 0:
					target_q.set_weights(online_q.get_weights())
				batch = memory_replay.sample(BATCH, False)
				batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

				batch_state = np.asarray(batch_state)
				batch_next_state = np.asarray(batch_next_state)
				batch_action = np.asarray(batch_action)
				batch_reward = np.asarray(batch_reward)
				batch_done = np.asarray(batch_done)

				online_next_a = tf.cast(tf.argmax(online_q(batch_next_state), axis=1), tf.int32)
				online_next_a_index = tf.stack([tf.range(tf.shape(online_next_a)[0]), online_next_a], axis=1)
				q_ = target_q(batch_next_state)
				y = batch_reward + (1 - batch_done) * GAMMA * tf.stop_gradient(tf.gather_nd(q_, online_next_a_index))

				with tf.GradientTape() as g:
					batch_action = tf.cast(batch_action, tf.int32)
					batch_action_index = tf.stack([tf.range(tf.shape(batch_action)[0]), batch_action], axis=1)
					q = online_q(batch_state)
					loss = mse(tf.gather_nd(q, batch_action_index), y)

				grads = g.gradient(loss, online_q.trainable_variables)
				optim.apply_gradients(zip(grads, online_q.trainable_variables))

				if epsilon > FINAL_EPSILON:
					epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
				writer.add_scalar('loss', float(loss), global_step=learn_steps)
			if done:
				break
			state = next_state
		writer.add_scalar('episode reward', episode_reward, global_step=epoch)
		if (epoch + 1) % 10 == 0:
			print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
			online_q.save_weights('ddqn-tf2.h5')
			print("Model saved!")


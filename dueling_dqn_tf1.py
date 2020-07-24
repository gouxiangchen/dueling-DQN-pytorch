import tensorflow as tf
import numpy as np
import gym
from collections import deque
from tensorboardX import SummaryWriter
from itertools import count
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

state_dim = 4
action_dim = 2 # left, right

GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 16

UPDATE_STEPS = 4

class Qnet:
	def __init__(self):
		self.batch_state = tf.placeholder(tf.float32, [None, state_dim], name='batch_state')
		self.batch_next_state = tf.placeholder(tf.float32, [None, state_dim], name='batch_next_state')

		self.batch_action = tf.placeholder(tf.int32, [None, ], name='batch_action')
		self.batch_reward = tf.placeholder(tf.float32, [None, ], name='batch_reward')
		self.batch_done = tf.placeholder(tf.float32, [None, ], name='batch_done')

		self.online_next_a = tf.placeholder(tf.int32, [None, ], name='online_next_a')

		with tf.variable_scope('online_Q'):
			q_fc1 = tf.layers.dense(self.batch_state, 64, tf.nn.relu, name='fc1')
			q_value_fc = tf.layers.dense(q_fc1, 256, tf.nn.relu, name='value_fc')
			q_adv_fc = tf.layers.dense(q_fc1, 256, tf.nn.relu, name='adv_fc')

			q_value = tf.layers.dense(q_value_fc, 1, name='value')
			q_adv = tf.layers.dense(q_adv_fc, action_dim, name='adv')

			advAverage = tf.reduce_mean(q_adv, axis=1, keepdims=True, name='adv_average')
			self.Q = q_value + q_adv - advAverage




		with tf.variable_scope('target_Q'):
			target_q_fc1 = tf.layers.dense(self.batch_next_state, 64, tf.nn.relu, name='fc1', trainable=False)
			target_q_value_fc = tf.layers.dense(target_q_fc1, 256, tf.nn.relu, name='value_fc', trainable=False)
			target_q_adv_fc = tf.layers.dense(target_q_fc1, 256, tf.nn.relu, name='adv_fc', trainable=False)

			target_q_value = tf.layers.dense(target_q_value_fc, 1, name='value', trainable=False)
			target_q_adv = tf.layers.dense(target_q_adv_fc, action_dim, name='adv', trainable=False)

			target_advAverage = tf.reduce_mean(target_q_adv, axis=1, keepdims=True, name='adv_average')
			self.target_Q = target_q_value + target_q_adv - target_advAverage

		self.q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online_Q')
		self.target_q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_Q')

		with tf.variable_scope('target_update'):
			self.target_hard_update = [tf.assign(t, e)
						for t, e in zip(self.target_q_params, self.q_params)]


		with tf.variable_scope('loss_function'):
			self.online_a = tf.cast(tf.argmax(self.Q, axis=1), tf.int32)


			online_a_index = tf.stack([tf.range(tf.shape(self.online_next_a)[0]), self.online_next_a], axis=1)
			target_next_q_value = tf.gather_nd(self.target_Q, online_a_index)

			y = tf.stop_gradient(self.batch_reward + (1 - self.batch_done) * GAMMA * target_next_q_value)
			action_index =  tf.stack([tf.range(tf.shape(self.batch_action)[0]), self.batch_action], axis=1)
			self.loss = tf.reduce_mean(tf.squared_difference(tf.gather_nd(self.Q, action_index), y))
			self.optim = tf.train.AdamOptimizer(1e-4).minimize(self.loss, var_list=self.q_params)

		with tf.variable_scope('model_save'):
			self.saver = tf.train.Saver(max_to_keep=1)

		self.initializer = tf.global_variables_initializer()

	def choose_action(self, sess, state):
		return sess.run(self.online_a, feed_dict={self.batch_state:state[np.newaxis, :]})[0]

	def initialize(self, sess):
		sess.run(self.initializer)

	def update_target_network(self, sess):
		sess.run(self.target_hard_update)

	def save_model(self, sess):
		self.saver.save(sess, 'dqn_model/model.ckpt')

	def load_model(self, sess):
		self.saver.restore(sess, 'dqn_model/model.ckpt')

	def optimize(self, sess, batch_state, batch_next_state, batch_action, batch_reward):
		online_next_a = sess.run(self.online_a, feed_dict={self.batch_state:batch_next_state})

		loss, _ = sess.run([self.loss, self.optim], feed_dict={self.online_next_a:online_next_a, self.batch_state:batch_state,
													self.batch_next_state:batch_next_state, self.batch_action:batch_action,
													self.batch_done:batch_done, self.batch_reward:batch_reward})
		return loss


		
if __name__ == '__main__':
	memory_replay = Memory(REPLAY_MEMORY)

	epsilon = INITIAL_EPSILON
	learn_steps = 0

	writer = SummaryWriter('ddqn')
	begin_learn = False
	episode_reward = 0

	ddqn = Qnet()

	with tf.Session() as sess:
		sess.graph.finalize()

		ddqn.initialize(sess)
		ddqn.update_target_network(sess)

		env = gym.make('CartPole-v0')
		state = env.reset()
		

		for epoch in count():
			state = env.reset()
			episode_reward = 0
			for time_steps in range(200):
				p = random.random()
				if p < epsilon:
					action = random.randint(0, 1)
				else:
					action = ddqn.choose_action(sess, state)
				next_state, reward, done, _ = env.step(action)
				episode_reward += reward
				memory_replay.add((state, next_state, action, reward, done))
				if memory_replay.size() > 128:
					if begin_learn is False:
						print('learn begin!')
						begin_learn = True
					learn_steps += 1
					if learn_steps % UPDATE_STEPS == 0:
						ddqn.update_target_network(sess)
					batch = memory_replay.sample(BATCH, False)
					batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

					batch_state = np.asarray(batch_state)
					batch_next_state = np.asarray(batch_next_state)
					batch_action = np.asarray(batch_action)
					batch_reward = np.asarray(batch_reward)
					batch_done = np.asarray(batch_done)

					
					loss = ddqn.optimize(sess, batch_state, batch_next_state, batch_action, batch_reward)

					if epsilon > FINAL_EPSILON:
						epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
					writer.add_scalar('loss', loss, global_step=learn_steps)
				if done:
					break
				state = next_state
			writer.add_scalar('episode reward', episode_reward, global_step=epoch)
			if (epoch + 1) % 10 == 0:
				print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
				ddqn.save_model(sess)
				print("Model saved!")

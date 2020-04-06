from ddqn_tf1 import Qnet
import tensorflow as tf
import numpy as np
import gym
from itertools import count
import random


if __name__ == '__main__':
	ddqn = Qnet()
	with tf.Session() as sess:
		ddqn.load_model(sess)
		env = gym.make('CartPole-v0')
		for epoch in count():
			state = env.reset()
			episode_reward = 0
			for time_steps in range(200):
				env.render()
				action = ddqn.choose_action(sess, state)
				next_state, reward, done, _ = env.step(action)
				episode_reward += reward
				state = next_state

			print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

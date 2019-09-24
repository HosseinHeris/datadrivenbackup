import copy
from collections import deque
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bonsai_tools
import joblib
import sklearn
import math
from gym.utils import seeding
from gym import spaces,logger
import time
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras import optimizers
#from predictor import ModelPredictor
 

class Star():

	def __init__(self, predict=False, modeltype='poly'):
	
		print('using supervised learning models for training ....')
		time.sleep(1)
		self.addnoise=True
		self.state_space_dim=4
		self.action_space_dim=1
		self.markovian_order=2
		predictor=ModelPredictor(modeltype=modeltype)

	def set_action(self, action):
		self.current_action=action

	def reset(self):

		if self.modeltype=='lstm':
			self.state=deque(self.np_random.uniform(low=-0.05, high=0.05, size=(self.markovian_order*self.state_space_dim,)),maxlen=self.markovian_order*self.state_space_dim)
			self.action_history=deque(np.zeros(shape=(self.markovian_order*self.action_space_dim,)),maxlen=self.markovian_order*self.action_space_dim)
		else:
			self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.state_space_dim,))

		self.steps_beyond_done = None
		self.iteration=0
		return np.array(self.state)


	def get_state(self):
		if self.modeltype=='lstm':
			state=np.ravel(self.state)
			action_history=np.ravel(self.action_history)
			#print('shape of all state array is:', state.shape)
			for i in range(0,self.markovian_order):
				for j in range (0, self.state_space_dim):
					key='state'+str(i)+str(j)
					self.state_to_brain[key]=state[j+i*self.state_space_dim]
			for i in range(0,self.markovian_order):
				for j in range (0, self.action_space_dim):
					key='action'+str(i)+str(j)
					self.state_to_brain[key]=action_history[j+i*self.action_space_dim]
		else:
			self.state_to_brain={
				"position": float(self.state[0]),
				"velocity": float(self.state[1]),
				"angle": float(self.state[2]),
				"rotation": float(self.state[3])
			}
		return self.state_to_brain

	def get_terminal(self, state):
		state=np.ravel(self.state)
		print('state for checking terminal conditions: ', state)
		x=state[0]
		theta=state[2]
		#self.iteration+=1
		done =  x < -self.x_threshold \
				or x > self.x_threshold \
				or theta < -self.theta_threshold_radians \
				or theta > self.theta_threshold_radians \
				or self.iteration > 200
		done = bool(done)
		self.terminal=done
		return done

	def get_reward(self, state, terminal):
		done = self.terminal 

		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0 
		return float(reward)

	def define_logged_observations(self):

		logged_observations = {}
		logged_observations.update(self.state_to_brain)
		logged_observations.update(self.current_action)		
		return logged_observations


if __name__ == "__main__":
	"""use star.py as main to test star piece without bonsai platform in the loop
	"""
	# # TODO: provide some instructions for testing
	# print("Testing star.py")
	# star = Star()
	# star.simulator_reset_config()
	# state = star.get_state()
	

	# #model = Model()
	# #model.simulator_initialize()
	# #model.simulator_reset()
	# #observations = model.simulator_get_observations()
	# observations_df = pd.DataFrame()
	# for i in range(284):
	# 	brain_action = simple_brain_controller(state)
	# 	sim_action = star.brain_action_to_sim_action(brain_action)
	# 	star.set_action(sim_action)
	# 	#model.simulator_step(action)
	# 	terminal = star.get_terminal(state)
	# 	reward = star.get_reward(state, terminal)
	# 	state = star.get_state()
	# 	print(state)

	# 	observations = star.model.simulator_get_observations()
	# 	observations.update(state)
	# 	observations.update({'terminal':terminal})
	# 	observations.update({'reward':reward})
	# 	observations.update({'brain_action':brain_action})
	# 	observations.update({'sim_action':sim_action})
	# 	observations_df = observations_df.append(observations,ignore_index=True)
	# 	print(observations)	

	# observations_df.plot(title='simulation integration plot')
	# plt.xlabel('iteration count')
	# plt.ylabel('observations')
	# plt.show()

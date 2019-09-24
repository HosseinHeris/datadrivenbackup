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
		#predictor=ModelPredictor(modeltype=modeltype)

		self.modeltype=modeltype
		print(modeltype, ' is used as the data driven model to train brain.')
		if modeltype=='gb':	
			for i in range(0,self.state_space_dim):
				filename='./models/gbmodel'+str(i)+'.sav'
				loaded_model=joblib.load(filename)
				setattr(self,'model'+str(i),loaded_model)
		elif modeltype=='poly':
			self.polydegree=joblib.load('./models/polydegree.sav')
			print('poyl degree is :', self.polydegree)
			for i in range(0, self.state_space_dim):
				filename='./models/polymodel'+str(i)+'.sav'
				loaded_model=joblib.load(filename)
				setattr(self,'model'+str(i),loaded_model)
		elif modeltype=='nn':
			self.model=load_model('./models/nnmodel.h5')
			print(self.model)
		elif modeltype=='lstm':
			self.model=load_model('./models/lstmmodel.h5')
			self.state_to_brain=self._generate_automated_state_names()
			#self.action_history_to_brain=self._generate_automated_actions_name()
		else:
			print('you need to specify which data driven is being used')
			

# initilization of the environment 
		self.steps_beyond_done = None
		self.iteration=0
		self.x_threshold = 2.4
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		self.high = np.array([
			self.x_threshold * 2,
			np.finfo(np.float32).max,
			self.theta_threshold_radians * 2,
			np.finfo(np.float32).max])
		self.seed()
		self.viewer = None
		self.state = None 
		self._setup_spaces()
		self.current_action=0

		if predict==True:
			print('to predict using actual gym environment use gym_cartpole_simulator.py file  ...')
			print('hint: python gym_cartpole_simulator.py --brain cartpolemodel --predict')
			time.sleep(5)
			
	def _generate_automated_state_names(self):
		state_to_brain=dict()
		for i in range(0,self.markovian_order):
			for j in range (0, self.state_space_dim):
				key='state'+str(i)+str(j)
				value=0
				state_to_brain[key]=value
		
		for i in range(0,self.markovian_order):
			for j in range (0, self.action_space_dim):
				key='action'+str(i)+str(j)
				value=0
				state_to_brain[key]=value
		print(state_to_brain)
		return state_to_brain

	def _generate_automated_actions_names(self):
		action_history_to_brain=dict()
		for i in range(0,self.markovian_order): 
			for j in range(0, self.action_space_dim):
				key='action'+str(i)+str(j)
				value=0
				action_history_to_brain[key]=value
		print(action_history)
		return action_history_to_brain
					
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		print('np_random:',self.np_random)
		time.sleep(1)
		return [seed] 
	def _setup_spaces(self):
		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)
	def set_action(self, action):
		self.current_action=action
		if self.modeltype=='lstm':
			#print('all lstm states are: ',self.state)
			if self.addnoise==True:
				model_input_state=np.reshape(np.array(self.state)+np.random.uniform(low=-0.1,high=0.1, size=self.markovian_order*self.state_space_dim), newshape=(self.markovian_order, self.state_space_dim))
			else:
				model_input_state=np.reshape(np.ravel(self.state), newshape=(self.markovian_order, self.state_space_dim))

			for key in action.keys():
				self.action_history.appendleft(action[key])
				#print('key is: ', key)
				#print(key, ':', action[key])
			#print('action history is:', np.array(self.action_history))
			model_input_actions=np.reshape(np.ravel(self.action_history), newshape=(self.markovian_order, self.action_space_dim))
			model_input=np.append(model_input_state, model_input_actions, axis=1)
			
			newstates=np.ravel(self.model.predict(np.array([model_input])))
			print('new state are:', newstates)
			for i in range(self.state_space_dim,0,-1):
				print('the ith state appended to the left of state', i, 'with value of: ', newstates[i-1])
				self.state=deque(self.state, maxlen=self.markovian_order*self.state_space_dim)  # I am not sure why i have to define deque again here. somewhere it becomes numpy array. 
				self.state.appendleft(newstates[i-1])
			#print('current lstm state after action is:', self.state)
		else:
			if self.addnoise==True:
				model_input=np.append(self.state+np.random.uniform(low=-0.1,high=0.1, size=self.state_space_dim), np.array(action['command']))
			else: 
				model_input=np.append(self.state, np.array(action['command']))		

		if self.modeltype=='gb':
			self.state=[]
			for i in range(0, self.state_space_dim):
				ithmodel=getattr(self,'model'+str(i))
				self.state=np.append(self.state, ithmodel.predict(np.array([model_input])),axis=0)
		
		elif self.modeltype=='poly':
			self.state=[]
			#model_input=model_input.reshape(1,-1)
			print('shape of input is: ', model_input.shape)
			model_input=self.polydegree.fit_transform([model_input])
			print('model input after transformation is: ', model_input)
			print('shape of input is: ', model_input.shape)
			model_input=model_input.reshape(1,-1)
			for i in range(0, self.state_space_dim):
				ithmodel=getattr(self,'model'+str(i))					
				self.state=np.append(self.state, ithmodel.predict(np.array(model_input)),axis=0)
				##print('self.state is .. :', self.state)			

		elif self.modeltype=='nn':
			self.state=[]
			#print('original model input is: ', model_input)
			#model_input=np.reshape(model_input, (-1,))
			print('model summary is:', self.model.summary())
			print('model input after reshaping is: ', model_input)
			print('reshape of input is: ', model_input.shape)
			self.state=self.model.predict(np.array([model_input]))
			print('self.state is .. :', self.state)
		elif self.modeltype=='lstm':
			pass

		self.state=np.ravel(self.state)
		
		return self.state

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, csv, sys, time, argparse
#import sim.cartpole as cartpole #importing simulator
import sim.house_simulator as house_simulator

class Model():
	"""
	Instantiation of simulator for training with Bonsai. 
	For new simulators, you should modify the following methods:
	1. `simulator_initialize`: define any parameters here for the start of your simulator, i.e., 
		parameters that will persist throughout all episodes of training. 
		The number of arguments here should hopefully minimal. 
		The last call in this method should be to `simulator_reset` -> 
	2. `simulator_reset`: this method will be called at the start of every new episode. 
		Use this method to reset variables/parameters created in previous episode that are no longer needed. 
		The only arguments here should be a dictionary named `config` 
		which will be passed to `simulator_config` below, and any simulator specific variable. 
		This method should also call `simulator_config` ->
	3. `simulator_config`: this method is called by `simulator_reset`. 
		This is where you define state parameters specific for the new episode. 
		Pass the `config` argument here to define state parameters for the simulator. 
		This method should end by calling the simulator specified in the `sim` directory:
		TODO: define how to utilize _inkling_ variables in  `simulator_config` 
	4. `simulator_step`: this method is called at every step. 
		It should take an argument named `action`, which will take the action from `star.set_action`. 
		This method should also update any necessary state variables from state.
	5. `simulator_get_observations`: Return observations from simulator. 
		Define all state variables you want to return. Note, these are not the 
		variables that are necessarily consumed by the state of your RL system. 
		Those are defined star.py. Instead, these are variables you want to log.
	6. `simulator_step`: Complete one step. 
		Should call `simulator_set_action` once.

	"""

	def __init__(self, K: float = 0.5, C: float = 0.3, 
				occupancy: int = 1, Qhvac: int = 9, predict: bool = False):

		self.modeldir = 'sim'
		os.chdir('./' +  self.modeldir)
		print("Using simulator file from: ", os.getcwd())

		self.simulation_state = None

		self.K = K
		self.C = C
		self.occupancy = occupancy
		self.Qhvac = Qhvac
		self.predict = predict

		self.simulator_initialize()

	def simulator_initialize(simulator
		self, 
		number_of_days: int = 1, 
		number_of_hours: int = 24, 
		timestep: int = 5,
		):
		"""
		Initialize simulator environment and with initial conditions once, at launch.
		
		Parameters
		----------
		number_of_days : int, optional
			[description] (the default is 1, which [default_description])
		number_of_hours : int, optional
			[description] (the default is 24, which [default_description])
		timestep : int, optional
			[description] (the default is 5, which [default_description])
		
		"""
		if self.predict:  def train_nn_model(self, x_train, y_train):
        self.model.fit(X_train, Y_train, epoches=20, batchsize=128)
        return self.model

    def evaluate_nn_model(x_test, y_test):
        self.score=self.model.evaluate(x_test,y_test, batch_size=128)
        return self.score
			self.number_  def train_nn_model(self, x_train, y_train):
        self.model.fit(X_train, Y_train, epoches=20, batchsize=128)
        return self.model

    def evaluate_nn_model(x_test, y_test):
        self.score=self.model.evaluate(x_test,y_test, batch_size=128)
        return self.score
			self.number_  def train_nn_model(self, x_train, y_train):
        self.model.fit(X_train, Y_train, epoches=20, batchsize=128)
        return self.model

    def evaluate_nn_model(x_test, y_test):
        self.score=self.model.evaluate(x_test,y_test, batch_size=128)
        return self.score
			self.timeste  def train_nn_model(self, x_train, y_train):
        self.model.fit(X_train, Y_train, epoches=20, batchsize=128)
        return self.model

    def evaluate_nn_model(x_test, y_test):
        self.score=self.model.evaluate(x_test,y_test, batch_size=128)
        return self.score
			self.max_iterations = int((self.number_of_days*self.number_of_hours)*60 / self.timestep)
		else:
			self.number_of_days = number_of_days
			self.number_of_hours = number_of_hours
			self.timestep = timestep
			self.max_iterations = int((self.number_of_days*self.number_of_hours)*60 / self.timestep)

		self.simulator_reset()

	def simulator_reset(self, config=None):
		"""Resets the simulator, executed at every episode start.
		
		Parameters
		----------
		config : dictionary
			Configuration parameters to be passed on to `simulator_configure`
		
		"""
		self.iteration = 0
		self.simulator_configure(config)
	
	def simulator_configure(self, config, initial_hvac: int = 0):
		"""Set up simulator initial conditions and configuration at the start of every episode
		
		Parameters
		----------
		TODO: assign config params to sim config variables and assign them from `star.simulator_reset_config`t st
		predict : bool, optional
			Whether to run train or eval loop (the default is False, which invokes training loop)
		
		"""
		action = {
			"hvacON": initial_hvac
		}

		if self.predict:
			# max predict timesteps at 1
			self.Tset_schedule = np.full(self.max_iterations, 25)
			self.Tout_schedule = np.full(self.max_iterations, 32)
			for d in range(self.number_of_days):
				a = time_to_index(d,9)
				b = time_to_index(d,17)
				self.Tset_schedule[a:b]= np.random.randint(low=18,high=25)
				c = time_to_index(d,0)
				d = time_to_index(d,24)
				self.Tout_schedule[c:d] = np.random.randint(low=28,high=35)
			self.Tin_initial = 30 
		else:
			self.Tset_schedule = np.full(self.max_iterations, 25)
			self.Tset_schedule[96:204]=20
			self.Tout_schedule = np.full(self.max_iterations, 32)
			self.Tin_initial = 30

		self.simulator = house_simulator.House(
			self.K, self.C, self.Tout_schedule[0], self.Tin_initial, 
			self.Tset_schedule[0], self.occupancy, self.Qhvac, action['hvacON']
		)
		
	def simulator_get_observations(self):
		"""Return observations from simulator, executed every iteration.
		
		Returns
		-------
		dictionary
			Dictionary of state elements at current iteration
		"""
		observations = {
			'Tset': float(self.simulator.Tset),
			'Tset1':float(self.Tset_schedule[self.iteration+1]),
			'Tset2':float(self.Tset_schedule[self.iteration+2]),
			'Tset3':float(self.Tset_schedule[self.iteration+3]), 
			'Tin': float(self.simulator.Tin),
			'Tout': float(self.simulator.Tout),
			'power':float(self.simulator.get_Power())
		}

		return observations

	def simulator_set_action(self, action):
		"""Execute action provided.
		
		Parameters
		----------
		action : dictionary
			Which action(s) to execute
		
		"""
		self.simulator.update_hvacON(action['hvacON'])

	def simulator_step(self, action):
		"""Execute simulator for one time step.
		
		Parameters
		----------
		action : action
			Action to execute through `simulator_set_action`
		"""
		self.simulator_set_action(action)
		self.iteration += 1

		self.simulator.update_Tin()
		self.simulator.update_Tset(self.Tset_schedule[self.iteration])
		self.simulator.update_Tout(self.Tout_schedule[self.iteration])
		self.simulator.update_occupancy(self.occupancy)

def time_to_index(days, hours, timestep=5):
	hours_index = int(hours*60/timestep)
	days_index = int(days*24*60/timestep)
	return hours_index + days_index

def simple_controller(observations):
	if observations['Tin'] > observations['Tset1']:
		command = 0
	else:
		command = 1
	action = {'hvacON':command}
	return action

if __name__ == "__main__":
	"""use simulator_integration.py as main to test simulator_integration piece alone
	"""
	print("Running simulator_integration.py")
	model = Model()
	model.simulator_initialize()
	model.simulator_reset()
	observations = model.simulator_get_observations()
	observations_df = pd.DataFrame()
	for i in range(284):
		action = simple_controller(observations)
		model.simulator_step(action)
		observations = model.simulator_get_observations()
		observations_df = observations_df.append(observations,ignore_index=True)
		print(observations)	

	observations_df.plot(title='simulation integration plot')
	plt.xlabel('iteration count')
	plt.ylabel('onservations')
	plt.show()

	

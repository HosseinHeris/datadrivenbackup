class ModelPredictor():
    def __init__(self, modeltype='gb'):
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
		else:
			print('you need to specify which data driven is being used')
    
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

    def predict(self, state, action=None, action_history=None):
        if self.modeltype=='lstm':
            if self.addnoise==True:
                model_input_state=np.reshape(np.array(self.state)+np.random.uniform(low=-0.1,high=0.1, size=self.markovian_order*self.state_space_dim), newshape=(self.markovian_order, self.state_space_dim))
            else:
                model_input_state=np.reshape(np.ravel(self.state), newshape=(self.markovian_order, self.state_space_dim))

            for key in action.keys():
                self.action_history.appendleft(action[key])

            model_input_actions=np.reshape(np.ravel(self.action_history), newshape=(self.markovian_order, self.action_space_dim))
            model_input=np.append(model_input_state, model_input_actions, axis=1)
            
            newstates=np.ravel(self.model.predict(np.array([model_input])))
            print('new state are:', newstates)
            for i in range(self.state_space_dim,0,-1):
                print('the ith state appended to the left of state', i, 'with value of: ', newstates[i-1])
                self.state=deque(self.state, maxlen=self.markovian_order*self.state_space_dim)  # I am not sure why i have to define deque again here. somewhere it becomes numpy array. 
                self.state.appendleft(newstates[i-1])

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
            newstates=scaler_y_set.inverse_transform(newstates)
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

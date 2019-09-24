import math
import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
import time
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import save_model
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from cartpole_sampler import generate_cartepole_data
from cartpole_sampler import generate_lstm_cartepole_data
from env_data_modeler import env_nn_modeler
from env_data_modeler import env_gb_modeler
from env_data_modeler import env_lstm_modeler
from env_data_modeler import env_poly_modeler
from env_data_modeler import create_nn_model_wrapper
from env_data_modeler import create_lstm_model_wrapper
import h5py
import argparse
import pickle
from cartpole_sampler import generate_cartepole_data
from cartpole_sampler import generate_lstm_cartepole_data
from gym_sampler import generate_data
from gym_sampler import generate_lstm_data
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor


if __name__=="__main__":

    state_space_dim=int(28)
    action_space_dim=int(8)  


    with open('./env_data/x_set.pickle', 'rb') as f:
        x_set = pickle.load(f)
    with open('./env_data/reward_set.pickle', 'rb') as f:
        y_set = pickle.load(f)
    y_set=np.ravel(y_set)

    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.33, random_state=42)

    gb_estimator=env_gb_modeler()
    gb_estimator.create_gb_model()
    gb_estimator.train_gb_model(x_train,y_train)
    score=gb_estimator.evaluate_gb_model(x_test, y_test)
    print('evaluation score for reward prediction is:', score)
    modelname='./models/gbmodelreward.sav'
    joblib.dump(gb_estimator.model, modelname)

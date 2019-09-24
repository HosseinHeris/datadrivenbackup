import gym
import numpy as np
import pickle
def generate_cartepole_data(total_sample=1000):
    env = gym.make('CartPole-v0')
    x_set=np.empty(shape=(total_sample,5))
    y_set=np.empty(shape=(total_sample,4))
    n=0
    for i_episode in range(100000):
        observation = env.reset()
        if n>total_sample-1:
            break
        else:
            pass 
        for t in range(100):
            if n>total_sample-1:
                break
                print('maximum number of samples recoded from the environment')
            else:
                pass 
            #env.render()
            action = env.action_space.sample()
            data_set=np.append(observation, np.array(action))
            x_set[n,:]=data_set
            observation, reward, done, info = env.step(action)
            y_set[n,:]=observation
            n+=1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    with open('x_set.pickle', 'wb') as f:
        pickle.dump(x_set, f, pickle.HIGHEST_PROTOCOL)
    with open('y_set.pickle', 'wb') as f:
        pickle.dump(y_set, f, pickle.HIGHEST_PROTOCOL)
    return x_set, y_set

def generate_lstm_cartepole_data(total_sample=1000, markovian_order=3):
    #reshape input to be [samples, time steps, features]
    env = gym.make('CartPole-v0')
    x_set=np.empty(shape=(total_sample, 5))
    y_set=np.empty(shape=(total_sample, 4))
    n=0
    for i_episode in range(1000000):
        observation = env.reset()
        if n>total_sample-1:
            break
        else:
            pass 
        for t in range(100):
            if n>total_sample-1:
                break
                print('maximum number of samples recoded from the environment')
            else:
                pass 
            #env.render()
            action = env.action_space.sample()
            data_set=np.append(observation, np.array(action))
            x_set[n,:]=data_set
            observation, reward, done, info = env.step(action)
            y_set[n,:]=observation
            n+=1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    x_set_lstm=np.empty(shape=(total_sample-markovian_order+1,markovian_order, 5))
    y_set_lstm=np.empty(shape=(total_sample-markovian_order+1, 4))
    for i in range(0, total_sample-markovian_order+1):
        a=x_set[i:(i+markovian_order),:]  #time steps, features
        b=y_set[i+markovian_order-1,:]
        # print('shape of a is: ', a.shape)
        # print('shape of b is:', b.shape)
        x_set_lstm[i,:,:]=a
        y_set_lstm[i,:]=b
    
    with open('x_set.pickle', 'wb') as f:
        pickle.dump(x_set_lstm, f, pickle.HIGHEST_PROTOCOL)
    with open('y_set.pickle', 'wb') as f:
        pickle.dump(y_set_lstm, f, pickle.HIGHEST_PROTOCOL)        
    return x_set_lstm, y_set_lstm
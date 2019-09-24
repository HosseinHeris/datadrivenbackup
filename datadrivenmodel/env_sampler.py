from star import Star
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num-samples", type=int, default=100000, help="define number of samples for random sampling out of the environment")
parser.add_argument("--markovian-order", type=int, default=0, help="define the order of markovian chain, i.e. number of past time points to include in the data driven model")

if __name__="__main__":
    star = Star()
    episode_length=284
    markovian_order=
    for in range (0, parser.num_samples):
	state = star.get_state()

	for i in range(284):
        star.simulator_reset_config()
        observations_df = pd.DataFrame()
		brain_action = simple_brain_controller(state)
		sim_action = action = {'hvacON':np.random.randint(0,2)}
		star.set_action(sim_action)
		terminal = star.get_terminal(state)
		reward = star.get_reward(state, terminal)
		state = star.get_state()
		print(state)

		observations = star.model.simulator_get_observations()
		observations.update(state)
		observations.update({'terminal':terminal})
		observations.update({'reward':reward})
		observations.update({'brain_action':brain_action})
		observations.update({'sim_action':sim_action})
		observations_df = observations_df.append(observations,ignore_index=True)
		print(observations)	

	observations_df.plot(title='simulation integration plot')
	plt.xlabel('iteration count')
	plt.ylabel('observations')
	plt.show()
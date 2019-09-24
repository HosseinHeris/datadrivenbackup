
def auto_ink(state_space_dim=4, action_space_dim=2, lstm=False, markovian_order=0):
    outF = open("Continuous_control.txt", "w")
    # write line by line to output file
    outF.write("\n")
    outF.write('schema State')
    outF.write("\n")
    for i in range(0, markovian_order+1):
        for j in range(0, state_space_dim):
            statename = "    Float32 state"+str(i)+str(j)+","
            outF.write(statename)
            outF.write("\n")
    outF.write("end")   
    outF.write("\n") 
    outF.write("schema Action")
    outF.write("\n")
    for i in range(0, markovian_order+1):
        for j in range(0, action_space_dim):
            actionname = "    Float32{-1:1} action"+str(i)+str(j)+","
            outF.write(actionname)
            outF.write("\n")
    outF.write("end")   
    outF.write("\n") 
    outF.write("schema SimulatorConfig")
    outF.write("\n") 
    outF.write("    Int8 dummy")
    outF.write("\n") 
    outF.write("end")
    outF.write("\n") 

    multiline="""
concept Continuous_Control is estimator 
    predicts (Action) # this concept controls 'Action' (defined above)
    follows input(State) # this concept Accepts as input 'State' (defined above)
    feeds output # this concept feeds directly the output of a BRAIN to the simulator/deployment (in more cadvanced solutions, a concept can feed other concepts)
end

simulator the_simulator(SimulatorConfig) # the simulator name must match the class name declared in simulator_bonsai_bridge.py.
    action (Action) # received as action: 'Action' defined above
    state (State) # returns as state: 'State' defined above
end

curriculum my_curriculum 
    train Continuous_Control
    using algorithm PPO
        timesteps_per_batch => 100000
    end
    with simulator the_simulator
    objective get_reward # the objective must match the reward function name in the simulator. in this case get_reward()
        lesson my_first_lesson # a curriculum can include multiple lessons that progressively increase the difficulty of a problem using 'SimulatorConfig'
            configure
                constrain dummy with Int8{-1} # In this simple example we don't use SimulatorConfig
            until
                maximize get_reward # the objective must match the reward function name in the simulator. in this case get_reward()
end
    """
    outF.write(multiline)
    outF.close

if __name__=="__main__":
    auto_ink(state_space_dim=28, action_space_dim=8, lstm=False, markovian_order=0)
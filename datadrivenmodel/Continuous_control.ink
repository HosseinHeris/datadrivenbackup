
schema State
    Float32 state00,
    Float32 state01,
    Float32 state02,
    Float32 state03,
    Float32 state04,
    Float32 state05,
    Float32 state06,
    Float32 state07,
    Float32 state08,
    Float32 state09,
    Float32 state010,
    Float32 state011,
    Float32 state012,
    Float32 state013,
    Float32 state014,
    Float32 state015,
    Float32 state016,
    Float32 state017,
    Float32 state018,
    Float32 state019,
    Float32 state020,
    Float32 state021,
    Float32 state022,
    Float32 state023,
    Float32 state024,
    Float32 state025,
    Float32 state026,
    Float32 state027
end
schema Action
    Float32{-1:1} action00,
    Float32{-1:1} action01,
    Float32{-1:1} action02,
    Float32{-1:1} action03,
    Float32{-1:1} action04,
    Float32{-1:1} action05,
    Float32{-1:1} action06,
    Float32{-1:1} action07
end
schema SimulatorConfig
    Int8 dummy
end

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
        timesteps_per_batch => 10000
    end
    with simulator the_simulator
    objective get_reward # the objective must match the reward function name in the simulator. in this case get_reward()
        lesson my_first_lesson # a curriculum can include multiple lessons that progressively increase the difficulty of a problem using 'SimulatorConfig'
            configure
                constrain dummy with Int8{-1} # In this simple example we don't use SimulatorConfig
            until
                maximize get_reward # the objective must match the reward function name in the simulator. in this case get_reward()
end
    
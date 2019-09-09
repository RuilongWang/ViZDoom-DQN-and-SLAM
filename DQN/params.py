# MODEL HYPERPARAMETERS
state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size =  3            #game.get_available_buttons_size()  # 3 possible actions: left, right, shoot
learning_rate = 0.0002  # Alpha (aka learning rate)

# TRAINING HYPERPARAMETERS
total_episodes = 500  # Total episodes for training
max_steps = 100  # Max possible steps in an episode
batch_size = 64
stack_size = 4

map_buffer_allowed = False  # saving every frame, pass to SLAM
save_image = False      # save every frame
# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95  # Discounting rate

# MEMORY HYPERPARAMETERS
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000  # Number of experiences the Memory can keep

training = False

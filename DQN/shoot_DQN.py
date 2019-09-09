import random
import warnings
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
import vizdoom
from skimage import transform
from pathlib import Path
import pickle

import Doom
from params import *

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_environment():
    game = vizdoom.DoomGame()

    # Load configuration
    game.load_config("basic.cfg")

    # Load scenario
    # game.set_doom_scenario_path("basic.wad")
    # game.set_doom_map("map01")

    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vizdoom.AutomapMode.OBJECTS_WITH_SIZE)
    # game.set_mode(vizdoom.Mode.SPECTATOR)

    game.init()

    # possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config

    # Crop the screen remove ceil
    cropped_frame = frame[30:-10, 30:-30]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame


def stack_frames(stacked_frames, stack_size, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

        # new episode, stack same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess, DQN):
    # Choose action a from state s using epsilon greedy.

    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if explore_probability > exp_exp_tradeoff:
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        Qs = sess.run(DQN.output, feed_dict={DQN.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


def main():
    game, possible_actions = create_environment()



    # filesaver = open("memory.obj", 'wb')
    # fileopener = open("memory.obj", 'rb')

    # episode_render = False

    # Reset the graph
    tf.compat.v1.reset_default_graph()

    # Instantiate the DQNetwork
    DQN = Doom.DQNetwork(state_size, action_size, learning_rate)

    # Instantiate memory
    memory = Doom.Memory(max_size=memory_size)

    # Render the environment
    game.new_episode()

    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

    # pretrain to get first batch size of memory
    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, stack_size, state, True)

        # Random action
        action = random.choice(possible_actions)

        # Get the rewards
        reward = game.make_action(action)

        # Look if the episode is finished
        done = game.is_episode_finished()

        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, stack_size, state, True)

        else:
            # Get the next state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, stack_size, next_state, False)

            # Add experience to memory
            memory.add((state, action, reward, next_state, done))

            state = next_state

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter("./tensorboard/dqn/1")

    # Losses
    tf.summary.scalar("Loss", DQN.loss)

    write_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    # memory = pickle.load(fileopener)

    if training:

        with tf.Session() as sess:
            # Initialize the variables
            # sess.run(tf.global_variables_initializer())



            model_path = Path("./models/model.ckpt.meta")
            if model_path.exists():
                saver.restore(sess, "./models/model.ckpt")
                print('Model Loaded')

            # Initialize the decay rate (that will use to reduce epsilon)
            decay_step = 0

            # Init the game
            game.init()

            for episode in range(total_episodes+1):
                step = 0
                episode_rewards = []
                game.new_episode()

                state = game.get_state().screen_buffer

                # stack pre-processed frames
                state, stacked_frames = stack_frames(stacked_frames, stack_size, state, True)

                while step < max_steps:

                    # show top-down map buffer of the game
                    if map_buffer_allowed:
                        automap = game.get_state().automap_buffer

                        if automap is not None:
                            cv2.imshow('ViZDoom Map Buffer', automap)

                        cv2.waitKey(21)

                    # save the frame
                    if save_image:
                        cv2.imwrite('video' + '/' + str(step).zfill(5) + '.png', game.get_state().screen_buffer)

                    step += 1
                    decay_step += 1

                    # Predict the action and take it
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                                 state, possible_actions, sess=sess, DQN=DQN)

                    # make action and get reward
                    reward = game.make_action(action)

                    done = game.is_episode_finished()
                    episode_rewards.append(reward)

                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros((84, 84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, stack_size, next_state, False)

                        # Set step = max_steps to end the episode
                        step = max_steps

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        memory.add((state, action, reward, next_state, done))

                    else:

                        total_reward = np.sum(episode_rewards)

                        # Get the next state
                        next_state = game.get_state().screen_buffer

                        # Stack the frame of the next_state
                        next_state, stacked_frames = stack_frames(stacked_frames, stack_size, next_state, False)

                        # Add experience to memory
                        memory.add((state, action, reward, next_state, done))

                        state = next_state

                    # LEARNING PART
                    # Obtain random mini-batch from memory

                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    # Get Q values for next_state
                    Qs_next_state = sess.run(DQN.output, feed_dict={DQN.inputs_: next_states_mb})

                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])

                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([DQN.loss, DQN.optimizer],
                                       feed_dict={DQN.inputs_: states_mb,
                                                  DQN.target_Q: targets_mb,
                                                  DQN.actions_: actions_mb})



               # Write TF Summaries
                    # summary = sess.run(write_op, feed_dict={DQN.inputs_: states_mb,
                    #                                         DQN.target_Q: targets_mb,
                    #                                         DQN.actions_: actions_mb})
                    # writer.add_summary(summary, episode)
                    # writer.flush()

                # Save model every 5 episodes
                # pickle.dump(memory, filesaver)
                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_probability))
                if episode % 20 == 0 and episode != 0:
                    saver.save(sess, "./models/model.ckpt")

                    print("Model Saved")
                sess.graph.finalize()

    # test result
    if not training:
        with tf.Session() as sess:

            game, possible_actions = create_environment()

            # Load the model
            saver.restore(sess, "./models/model.ckpt")
            game.init()
            for i in range(100):

                done = False

                game.new_episode()

                state = game.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, stack_size, state, True)

                while not game.is_episode_finished():
                    # Take the biggest Q value (= the best action)
                    Qs = sess.run(DQN.output, feed_dict={DQN.inputs_: state.reshape((1, *state.shape))})

                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)
                    action = possible_actions[int(choice)]

                    game.make_action(action)
                    done = game.is_episode_finished()
                    score = game.get_total_reward()

                    if done:
                        break

                    else:
                        next_state = game.get_state().screen_buffer
                        next_state, stacked_frames = stack_frames(stacked_frames, stack_size, next_state, False)
                        state = next_state

                score = game.get_total_reward()
                print("Score: ", score)
            game.close()


if __name__ == '__main__':
    main()

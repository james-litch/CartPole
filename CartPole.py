import gym
import math
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential

# training parameters.

num_episodes = 100000
num_winning_steps = 195
max_env_steps = None

discount_factor = 1.0  # discount factors.
exploration_factor = 1.0  # exploration factors
min_exploration = 0.01
exploration_decay = 0.955
learning_rate = 0.01  # learning factors
learning_decay = 0.01

batch_size = 64

# environment parameters.

memory = deque(maxlen=10000)
# step 1
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]  # this is equal to 4.

if max_env_steps is not None:
    env._max_episode_steps = max_env_steps

# defining model - step 2

model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))  # environment has 4 parameters that describes environment.
model.add(Dense(48, activation='relu'))
model.add(Dense(2, activation='relu'))  # 2 outputs for force to left and right.
model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=learning_decay))

# loading model
model = load_model("DQN.p5")
print(model.summary())
print("loading file")


# useful functions.

def completion_check(mean_score, episode):
    # if the mean score > winning score save the model, output and return true.
    if mean_score >= num_winning_steps and episode >= 100:
        print("Ran {} episodes. Solved after {} trials".format(episode, episode / 100))
        model.save("DQN.p5")
        return True
    else:
        return False


# stores previous decisions.
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def pick_action(state, exploration_factor):
    # if the exploration factor is high the agents is likely to pick a random action.
    # Else the model predicts what action to pick given the current sate
    return env.action_space.sample() if np.random.random() <= exploration_factor else np.argmax(model.predict(state))


def get_exploration_factor(time):
    # the exploration factor decreases at a faster rate the more the agent explores.
    return max(min_exploration, min(exploration_factor, 1.0 - math.log10((time + 1) * exploration_decay)))


def preprocess_state(state):
    # makes sure array is in the right import format by transposing it into a 1 x (state size )
    return np.reshape(state, [1, state_size])


def replay(batch_size, exploration_factor):
    x_batch, y_batch = [], []
    # random sample from memory
    memory_sample = random.sample(memory, min(len(memory), batch_size))

    for state, action, reward, next_state, done in memory_sample:
        y_target = model.predict(state)  # what model is trying to predict.
        # rewarded if correct target to be predicting.
        y_target[0][action] = reward if done else reward + discount_factor * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    # use actions to train model - Step 3
    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    if exploration_factor > min_exploration:
        exploration_factor *= exploration_decay


# main function that trains the model.

def run():
    scores = deque(maxlen=100)  # stores the past 100 episodes scores.

    for episode in range(num_episodes):
        state = preprocess_state(env.reset())
        done = False
        current_time = 0  # starts at a time stamp of 0
        while not done:
            # picks an action using the given state and exploration factor.
            action = pick_action(state, get_exploration_factor(episode))
            next_state, reward, done, _ = env.step(action)
            env.render()  # responsible for game visuals.
            # gets the next state ready.
            next_state = preprocess_state(next_state)
            # stores information to memory.
            remember(state, action, reward, next_state, done)
            state = next_state
            # adds 1 to score for that episode, this also acts as the reward for the agent.
            current_time += 1

        scores.append(current_time)
        mean_score = np.mean(scores)

        if episode % 100 == 0 and episode > 0:  # if current episode is a multiple of 100 anc not 0.
            print("[Trial {}] - Average survival time over the past 100 episodes was {} seconds".format(episode / 100,
                                                                                                        mean_score))
            if completion_check(mean_score, episode):  # if completion check returns true then program will end.
                break

        replay(batch_size, get_exploration_factor(episode))  # this trains the model based on results.


run()

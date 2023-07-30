import gymnasium as gym
from collections import deque
import random
import numpy as np
import gymnasium as gym
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import cv2

class DQNAgent:
    def __init__(self, epsilon = 1.0):
        """"
        # This constructor initializes the class with an exploration rate (epsilon), sets up the actions possible
        # in the action space, creates a memory buffer, sets a discount rate (gamma), sets up an epsilon decay
        # rate for adjusting exploration, sets a learning rate, and builds the deep learning model.

        # Define the action space for the car race environment, with varying combinations of steering, acceleration, and brake

        # action space Structure
        #       (Steering Wheel, speed, Break)
        # Range       -1-1       0-1   0-1

        """
        self.action_space    = [(-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
                                (-1, 1,   0), (0, 1,   0), (1, 1,   0),
                                (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
                                (-1, 0,   0), (0, 0,   0), (1, 0,   0)]
        
        self.memory          = deque(maxlen=5000) # Set up memory with a maximum size of 5000 for the agent's experiences
        self.gamma           = 0.95 # Discount rate
        self.epsilon         = epsilon # Exploration rate
        self.epsilon_min     = 0.1
        self.epsilon_decay   = 0.9999
        self.learning_rate   = 0.001

        # Create two instances of the model, one for predicting the Q-values, and another as the target model
        # The target model is used to stabilize learning, by providing a fixed set of weights for calculating target values as seen in the Deep Q-Network paper
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.update_target_model()  # Copy weights from the model to the target model

    def build_model(self):
        # Define the deep learning model architecture - a CNN in this case (for visual tasks like the Box2D or Atari games)
        # Model structure: Two convolutional layers, followed by flattening and dense layers
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        # Function to copy the weights from the model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def action(self, state):
        # Function to select an action, either randomly (exploration) or the one with maximum predicted Q-value (exploitation)
        if np.random.rand() > self.epsilon:
            # Get action with max value
            action_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            action_index = np.argmax(action_values[0])
        else:
            # Get random action
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def load(self, name):
        # Function to load weights into the model from a file, then copy them to the target model (for playing with trained weights)
        self.model.load_weights(name)
        self.update_target_model()

def process_state_image(state):
    """
    This function processes the input state (an image) by converting it to grayscale and normalizing the pixel values.
    The purpose of this preprocessing step is to simplify the input without losing too much information.
    
    :param state: A 3-channel color image representing a game state.
    :return: A grayscale image with normalized pixel values.
    """
    
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)    # Convert color image to grayscale
    state = state.astype(float)                        # Convert data to float
    state /= 255.0                                     # Normalize
    return state

def generate_state_frame_stack_from_queue(deque):
    """
    A Function that takes a deque containing multiple frames of the game state and 
    converts it into a numpy array, then transposes the array to move the 'stack' dimension to the channel dimension. 
    This way, the stack of frames can be processed by a Convolutional Neural Network as separate channels.
    
    :param deque: A deque object containing multiple frames of the game state.
    :return: A numpy array where the stack dimension has been moved to the channel dimension.
    """
    frame_stack = np.array(deque)    
    return np.transpose(frame_stack, (1, 2, 0))


# Specify the saved model to load
MODEL = 'model/trial_10.h5'

def run_game():
    env = gym.make('CarRacing-v2', render_mode="human") # Create the gym environment and the agent for the CarRacing-v2 env.
    agent = DQNAgent(epsilon=0) # Epsilon 0 allows deterministic action selection
    agent.load(MODEL) # load the pre-trained model

    # Initialize the state of the environment and process the initial image
    init_state = env.reset()
    init_state = process_state_image(init_state[0])

    total_reward = 0
    state_frame_stack_queue = deque([init_state] * 3, maxlen=3)
    time_frame_counter = 1
    
    # Game loop, this renders it and executes the actions given
    while True:
        # Render the environment on screen
        env.render()

        current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
        action = agent.action(current_state_frame_stack)
        next_state, reward, done, info, _ = env.step(action) # Take a step in the environment using the selected action
        
        # Print the selected action
        print(f"Selected action at step {time_frame_counter}: {action}")

        total_reward += reward

        # Process the image of the next state and add it to the deque
        next_state = process_state_image(next_state)
        state_frame_stack_queue.append(next_state)
   
        # If the game is done, print the total frames and reward, then break the loop (as seen in the training)
        if done:
            print('Time Frames: {}, Total Rewards: {:.2}'.format(time_frame_counter, float(total_reward)))
            break
        time_frame_counter += 1

    env.close()

if __name__ == '__main__':
    run_game()
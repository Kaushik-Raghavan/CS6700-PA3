import gym
import sys
import copy
import random

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

class DQN:

    REPLAY_MEMORY_SIZE = 10000 			# number of tuples in experience replay
    EPSILON = 0.5 						# epsilon of epsilon-greedy exploation     # Default 1.0
    EPSILON_DECAY = 0.99 				# exponential decay multiplier for epsilon
    MIN_EPS_VALUE = 0.05					# Stop decaying epsilon if epsilon is below this value    # Default 0.0
    CONSTANT_STEP_SIZE = 20 				# number of steps for which epsilon should be maintained constant before decaying  Default 1
    HIDDEN1_SIZE = 256 					# size of hidden layer 1		Default 128
    HIDDEN2_SIZE = 256					# size of hidden layer 2		Default 128
    #HIDDEN3_SIZE = 2 					# size of hidden layer 3    	Default 128
    EPISODES_NUM = 2000 				# number of episodes to train on. Ideally shouldn't take longer than 2000
    MAX_STEPS = 200 					# maximum number of steps in an episode
    LEARNING_RATE = 0.0001 				# learning rate and other parameters for SGD/RMSProp/Adam   Default 0.00025 //given in DQN paper
    MINIBATCH_SIZE = 10 				# size of minibatch sampled from the experience replay
    DISCOUNT_FACTOR = 0.9 				# MDP's gamma
    TARGET_UPDATE_FREQ = 100 			# number of steps (not episodes) after which to update the target networks   # Default 100
    LOG_DIR = './logs/logs1' 					# directory wherein logging takes place
    LAMBDA = 0.001						# Regularization weight



    # Create and initialize the environment
    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
        self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)
        self.gamma = self.DISCOUNT_FACTOR
        self.eps = self.EPSILON

    # Create the Q-network
    def initialize_network(self):

        # placeholder for the state-space input to the q-network
        self.x = tf.placeholder(tf.float32, [None, self.input_size])

        ############################################################
        # Q-network.
        #############################################################
        self.weights = {
                'W1': tf.Variable(np.random.normal(0, 0.01, size = (self.input_size, self.HIDDEN1_SIZE)), dtype = tf.float32, name = 'weight1'),
                'b1': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN1_SIZE)), dtype = tf.float32, name = 'bias1'),
                'W2': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN1_SIZE, self.HIDDEN2_SIZE)), dtype = tf.float32, name = 'weight2'),
                'b2': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN2_SIZE)), dtype = tf.float32, name = 'bias2'),
                'W3': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN2_SIZE, self.output_size)), dtype = tf.float32, name = 'weight3'),
                'b3': tf.Variable(np.random.normal(0, 0.01, size = (self.output_size)), dtype = tf.float32, name = 'bias3'),
                #'W4': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN3_SIZE, self.output_size)), dtype = tf.float32, name = 'weight4'),
                #'b4': tf.Variable(np.random.normal(0, 0.01, size = (self.output_size)), dtype = tf.float32, name = 'bias4'),
                }

        H1 = tf.matmul(self.x, self.weights['W1']) + self.weights['b1']
        H1 = tf.nn.relu(H1)
        H2 = tf.matmul(H1, self.weights['W2']) + self.weights['b2']
        H2 = tf.nn.relu(H2)
        #H3 = tf.matmul(H2, self.weights['W3']) + self.weights['b3']
        #H3 = tf.nn.relu(H3)
        self.Q = tf.matmul(H2, self.weights['W3']) + self.weights['b3']

        self.target_weights = {
                'W1': tf.Variable(np.random.normal(0, 0.01, size = (self.input_size, self.HIDDEN1_SIZE)), dtype = tf.float32, name = 'target_weight1'),
                'b1': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN1_SIZE)), dtype = tf.float32, name = 'target_bias1'),
                'W2': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN1_SIZE, self.HIDDEN2_SIZE)), dtype = tf.float32,  name = 'target_weight2'),
                'b2': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN2_SIZE)), dtype = tf.float32, name = 'target_bias2'),
                'W3': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN2_SIZE, self.output_size)), dtype = tf.float32, name = 'target_weight3'),
                'b3': tf.Variable(np.random.normal(0, 0.01, size = (self.output_size)), dtype = tf.float32, name = 'target_bias3'),
                #'W4': tf.Variable(np.random.normal(0, 0.01, size = (self.HIDDEN3_SIZE, self.output_size)), dtype = tf.float32, name = 'target_weight4'),
                #'b4': tf.Variable(np.random.normal(0, 0.01, size = (self.output_size)), dtype = tf.float32, name = 'target_bias4'),
                }

        Ht1 = tf.matmul(self.x, self.target_weights['W1']) + self.target_weights['b1']
        Ht1 = tf.nn.relu(Ht1)
        Ht2 = tf.matmul(Ht1, self.target_weights['W2']) + self.target_weights['b2']
        Ht2 = tf.nn.relu(Ht2)
        #Ht3 = tf.matmul(Ht2, self.target_weights['W3']) + self.target_weights['b3']
        #Ht3 = tf.nn.relu(Ht3)
        self.Qt = tf.matmul(Ht2, self.target_weights['W3']) + self.target_weights['b3']


        ############################################################
        # Next, compute the loss.
        #
        # First, compute the q-values. Note that you need to calculate these
        # for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
        #
        # Next, compute the l2 loss between these estimated q-values and
        # the target (which is computed using the frozen target network)
        #
        ############################################################
        self.curr_a = tf.placeholder(tf.int32, [None], name = 'curr_a')
        self.one_hot_curr_a = tf.one_hot(self.curr_a, self.output_size, 1.0, 0.0, name = 'one_hot_curr_a')
        self.Qvalues = tf.reduce_sum(tf.multiply(self.Q, self.one_hot_curr_a, name = 'one_hot_q'), reduction_indices = [1])

        self.target = tf.placeholder(tf.float32, [None], name = 'target')
        self.error = self.target - self.Qvalues
        self.loss = (tf.reduce_mean(tf.square(self.error)) +
                    self.LAMBDA * (tf.nn.l2_loss(self.weights['W1']) + tf.nn.l2_loss(self.weights['W2']) +
                    tf.nn.l2_loss(self.weights['W3']))) #tf.nn.l2_loss(self.weights['W4']) +

        ############################################################
        # Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam.
        #
        # For instance:
        # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        #
        ############################################################
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.LEARNING_RATE)
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.train_op = optimizer.minimize(self.loss, global_step = global_step)

        ############################################################

    def train(self, episodes_num = None, render = True):
        self.env.reset()
        # Initialize summary for TensorBoard
        summary_writer = tf.summary.FileWriter(self.LOG_DIR)
        summary = tf.Summary()

        # Initialize the TF session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        #summary_writer.add_graph(self.session.graph)


        ############################################################
        # Initialize other variables (like the replay memory)
        ############################################################
        replay_buffer = []
        total_steps = 0;

        ############################################################
        # Initialize return variables
        ############################################################
        last100_episodes = []
        avg_steps = []

        ############################################################
        # Main training loop
        #
        # In each episode,
        #	pick the action for the given state,
        #	perform a 'step' in the environment to get the reward and next state,
        #	update the replay buffer,
        #	sample a random minibatch from the replay buffer,
        # 	perform Q-learning,
        #	update the target network, if required.
        #
        ############################################################
        for episode in range(self.EPISODES_NUM):

            curr_state = self.env.reset()

            ############################################################
            # Episode-specific initializations go here.
            ############################################################
            episode_length = 0
            episode_cost = 0.0
            episode_reward = 0.0
            ############################################################

            while True:

                ################################################################
                # Pick the next action using epsilon greedy and and execute it
                ################################################################
                curr_a = self.get_action(curr_state)
                if (total_steps % self.CONSTANT_STEP_SIZE == 0 and self.eps > self.MIN_EPS_VALUE):  # Decay epsilon after every CONSTANT_STEP_SIZE episodes
                    self.eps *= self.EPSILON_DECAY

                ############################################################
                # Step in the environment.
                ############################################################
                next_state, reward, done, _ = self.env.step(curr_a)
                episode_reward += reward
                episode_length += 1
                total_steps += 1

                if (done):
                    if (episode_length < self.MAX_STEPS): reward = reward
                if (render):
                    self.env.render()

                reward -= abs(next_state[0])
                curr_experience = [curr_state, curr_a, reward, next_state, done]

                ############################################################
                # Update the (limited) replay buffer.
                #
                # Note : when the replay buffer is full, you'll need to
                # remove an entry to accommodate a new one.
                ############################################################
                if len(replay_buffer) == self.REPLAY_MEMORY_SIZE:
                    replay_buffer[total_steps % self.REPLAY_MEMORY_SIZE] = curr_experience.copy()
                else:
                    if (len(replay_buffer) == 0):
                        replay_buffer = [curr_experience.copy()]
                    else:
                        replay_buffer.append(curr_experience.copy())

                ############################################################
                # Sample a random minibatch and perform Q-learning (fetch max Q at s')
                #
                # Remember, the target (r + gamma * max Q) is computed
                # with the help of the target network.
                # Compute this target and pass it to the network for computing
                # and minimizing the loss with the current estimates
                #
                ############################################################
                if (len(replay_buffer) > self.MINIBATCH_SIZE):
                    experiences = np.array(random.sample(replay_buffer, self.MINIBATCH_SIZE))
                    #experiences[0] = curr_experience.copy()

                    # Converting objects array to float matrix of nxt_state, where each row corresponds to a state
                    current_states = np.array([list(exp) for exp in experiences[:, 0]])
                    current_actions = np.array([act for act in experiences[:, 1]])  # indices of actions chosen in the experiences
                    rewards = np.array([rwd for rwd in experiences[:, 2]])
                    next_states = np.array([list(exp) for exp in experiences[:, 3]])
                    terminal = np.array([d for d in experiences[:, 4]])

                    q_i = np.max(self.session.run(self.Qt, feed_dict = {self.x: next_states}), axis = 1)
                    y_i = np.where(terminal, rewards, rewards + self.gamma * q_i)

                    _, cost = self.session.run([self.train_op, self.loss], feed_dict = {self.x: current_states, self.target: y_i, self.curr_a: current_actions})
                    episode_cost += cost


                    ############################################################
                    # Update target weights.
                    #
                    # Something along the lines of:
                    # if total_steps % self.TARGET_UPDATE_FREQ == 0:
                    # 	target_weights = self.session.run(self.weights)
                    ############################################################
                    if total_steps % self.TARGET_UPDATE_FREQ == 0:
                        print ("Reseting target network")
                        for key in self.weights:
                            #self.target_weights[key] = self.weights[key]
                            self.session.run(tf.assign(self.target_weights[key], tf.identity(self.weights[key])))
                            #assert np.all(self.session.run(self.target_weights[key]) == self.session.run(self.weights[key])), "Nope. Not equal\n"

                ############################################################
                # Break out of the loop if the episode ends
                #
                # Something like:
                # if done or (episode_length == self.MAX_STEPS):
                # 	break
                ############################################################
                curr_state = next_state
                if (done):
                    break


            if (len(last100_episodes) < 100):
                last100_episodes.append(episode_length)
            else:
                last100_episodes[episode % 100] = episode_length
            mean = np.mean(last100_episodes)
            avg_steps.append(mean)
            ############################################################
            # Logging.
            #
            # Very important. This is what gives an idea of how good the current
            # experiment is, and if one should terminate and re-run with new parameters
            # The earlier you learn how to read and visualize experiment logs quickly,
            # the faster you'll be able to prototype and learn.
            ############################################################
            print("Training: Episode = %d, Global step = %d, Length = %d, Average steps = %0.2f" % (episode, total_steps, episode_length, mean))
            summary.value.add(tag = "Average total reward (number of steps per episode)", simple_value = mean)
            summary_writer.add_summary(summary, episode)


            if (np.min(last100_episodes) >= 195 and episodes_num == None):
                print ("Task solved. Reward is >= 195 for last 100 episodes")
                break

        return avg_steps, last100_episodes


    def get_action(self, curr_state):
        #if (np.random.binomial(1, self.eps) == 1):
            if (random.random() < self.eps):
                return np.random.binomial(1, 0.5)
            Qvals = self.session.run(self.Q, feed_dict = {self.x: curr_state.reshape(1, curr_state.shape[0])})
            return Qvals.argmax()

    def terminal_state(self, curr_state, episode_length):
        return (abs(curr_state[0]) > 2.4 or abs(curr_state[2] > 12.0 or episode_length >= self.MAX_STEPS))

    def clipped_error(self, error):
        return tf.where(tf.abs(error) < 1, 0.5 * (error ** 2), tf.abs(error) - 0.5)   # Huber loss


    # Simple function to visually 'test' a policy
    def playPolicy(self):

        done = False
        steps = 0
        state = self.env.reset()

        # we assume the CartPole task to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:
            self.env.render()
            q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
            action = q_vals.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1

        return steps



if __name__ == "__main__":
    # Create and initialize the model
        dqn = DQN('CartPole-v0')
        dqn.initialize_network()

        print("\nStarting training...\n")
        avg_rewards, _ = dqn.train(render = True)
        print("\nFinished training...\nCheck out some demonstrations\n")

        np.savetxt("AverageRewards100-100_try2.csv", avg_rewards)
        """
        plt.figure(1)
        plt.plot(np.arange(len(avg_rewards)), avg_rewards)
        plt.xlabel("Episodes (till completion)")
        plt.ylabel("Average steps (reward)")
        plt.title("Avg steps vs episodes")
        plt.show()
        """

        # Visualize the learned behaviour for a few episodes
        results = []
        for i in range(50):
                episode_length = dqn.playPolicy()
                print("Test steps = ", episode_length)
                results.append(episode_length)
        print("Mean steps = ", sum(results) / len(results))

        print("\nFinished.")
        print("\nCiao, and hasta la vista...\n")

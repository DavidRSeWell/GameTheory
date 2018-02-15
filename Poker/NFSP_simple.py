'''
This will be an implementation of the Neural fictitious self-play with fitted Q learning
as implemented by Silver ,Heinrich 2017

Simplified version is that we use a DQN model for the Q(s,a) network with no target network
and for our policy network we just use a simple counting function as seen in Heinrich 2015 XFP paper
'''

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:

            return random.randrange(self.action_size)

        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:

                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class AKQPlayer(object):

    def __init__(self,name,info_tree,starting_stack):

        self.name = name

        self.info_tree = info_tree

        self.policy = {}

        self.out_of_tree = False

        self.current_hand = None

        self.starting_stack = starting_stack

class NFSPSimple:

    def __init__(self,tree):

        self.tree = tree

        self.rl_replay = []

        self.actions = ["bet","check","call"]

        self.q_parameters = None

        self.anticipatory_paramter = None

        self.player1 = None

        self.player2 = None

        self.deck = [0,1,2]

    def deal_hand(self):

        return random.choice(self.deck)

    def init(self):
        '''
        initialize the dqn network with single hidden layer
        :return:
        '''

    def get_new_state(self, s, a):

        node_children = s.children

        for child in node_children:

            if child.action == a:

                return child

            else:
                continue

        # we should not reach this line of code
        # the function should always be able to return a new state

        raise Exception("get_new_state was not able to find a child with the given action")

    def get_policy(self):
        '''
        selects either an epsilon greedy policy with prob 1 - ant
        select current policy using the policy network with prob ( ant )
        :return:
        '''
        return

    def get_reward(self,s):

        '''

                Takes in a leaf node and returns the reward to each player
                :param s:
                :return:

                '''

        r = {"SB": 0, "BB": 0}

        villian, hero = self.get_hero_villian(s)

        villian_cip, hero_cip = self.get_hero_villian_cip(s)

        current_pot = 1.0 + s.SB_cip + s.BB_cip

        action_type = list(s.action.keys())[0]

        if action_type == "fold":
            # the parent folded so the current player gets the pot
            r[villian.name] = villian.starting_stack - villian_cip

            r[hero.name] = current_pot + (hero.starting_stack - hero_cip)


        elif action_type == "check":

            # evaluate winner
            if (hero.current_hand < villian.current_hand):
                # SB wins
                r[hero.name] = current_pot + (hero.starting_stack - hero_cip)

                r[villian.name] = villian.starting_stack - villian_cip

            else:

                r[villian.name] = current_pot + (villian.starting_stack - villian_cip)

                r[hero.name] = hero.starting_stack - hero_cip


        elif action_type == "call":  # same as check?

            # evaluate winner
            if (hero.current_hand < villian.current_hand):
                # SB wins
                r[hero.name] = current_pot + (hero.starting_stack - hero_cip)

                r[villian.name] = villian.starting_stack - villian_cip

            else:

                r[villian.name] = current_pot + (villian.starting_stack - villian_cip)

                r[hero.name] = hero.starting_stack - hero_cip

        return r

    def get_state_from_node(self,s,player,action):

        '''
        Takes in the node and current hand of the player and
        return the array representation of the current state

        state = players x raises x actions + card_vector
        :param s:
        :return:
        '''

        curr_player = np.array([1,0])

        if s.player == "BB":

            curr_player = np.array([0,1])

        action_matrix = np.zeros((3,3))  # raises x actions matrix

        action_matrix[1][1] = 1

        action_history = []

        curr_node = s

        while True:

            if curr_node.node_index == 0:
                break

            if curr_node.node_index in [2,3]:

                a_index = self.actions.index(curr_node.action)

                action_matrix[1][a_index] = 1

            elif curr_node.node_index in [4,5]:

                a_index = self.actions.index(curr_node.action)

                action_matrix[2][a_index] = 1

            action_history.insert(0,curr_node.action)

            curr_node = s.parent

        action_flattened = action_matrix.flatten()

        hand_array = np.zeros(3)

        curr_hand_index = self.deck.index(player.current_hand)

        hand_array[curr_hand_index] = 1

        state = np.concatenate((curr_player,action_flattened))

        state = np.concatenate((state,hand_array))

        return state

    def execute_action(self,action,state):
        '''
        Take in the current state and return the reward and the next state
        :param action:
        :param state:
        :return:
        '''

    def simulate(self,s,policy,i):

        '''
        Used like in mcts to allow an episode of the game to be played out
        :param s:
        :return:
        '''


        if s.is_leaf == True:

            reward = self.get_reward(s)

            return reward

        current_player = s.player

        action = policy.select(s,current_player,)

        next_state = self.get_new_state(s, action)[0]

        r = self.simulate(next_state, i)

        rl_transition = (s,action,r,next_state)

        self.rl_replay.append()

        self.update(s, r, i)

        return r

    def run(self,iterations):

        for i in range(iterations):

            policy = self.get_policy()

            self.deck = [0,1,2]  # reshuffle the cards yo

            self.player1.out_of_tree = False

            self.player2.out_of_tree = False

            # deals cards to each player

            sb_card = self.deal_hand()

            self.player1.current_hand = sb_card

            self.deck.remove(sb_card)

            bb_card = self.deal_hand()

            self.player2.current_hand = bb_card

            s0 = self.tree.get_root()

            self.simulate(s0,policy, i)





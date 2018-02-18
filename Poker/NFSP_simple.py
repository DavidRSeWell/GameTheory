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
        self.learning_rate = 0.1
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

    def act(self, state,valid_actions):

        if np.random.rand() <= self.epsilon:

            random_valid_action = valid_actions[random.randrange(len(valid_actions))]

            return (random_valid_action,'random')

        act_values = self.model.predict(state)

        for i in range(self.action_size):
            if i not in valid_actions:
                act_values[0][i] = -100

        return (np.argmax(act_values[0]),'max')  # returns action

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

        self.actions = ["bet","check","call","fold"]

        self.sb_DQN = None

        self.q_parameters = None

        self.deck = [0, 1, 2]

        self.behaviour_policy = np.zeros((len(tree.nodes),len(self.deck),len(self.actions)))

        self.count_lookup = {'SB':np.zeros((6,3)),'BB':np.zeros((6,3))}

        self.anticipatory_paramter = 0.1

        self.player1 = None

        self.player2 = None

        self.policy = None

        self.batch_size = 8

        self.init()

    def deal_hand(self):

        return random.choice(self.deck)

    def init(self):

        '''
        initialize the dqn network with single hidden layer
        :return:
        '''

        self.player1 = AKQPlayer(name="SB", info_tree=None, starting_stack=1)

        self.player2 = AKQPlayer(name="BB", info_tree=None, starting_stack=1)

        sb_dqn = DQNAgent(17,4)

        bb_dqn = DQNAgent(17,4)

        self.sb_DQN = sb_dqn

        self.bb_DQN = bb_dqn

        # init the behaviour policy s.t. the player is equally likely to take an action
        # with every hand in every state
        for node_index in [1,2]:
            for hand in self.deck:
                for action in self.get_possible_action(self.tree.nodes[node_index]):
                    self.behaviour_policy[node_index][hand][action] = 0.5

        print('done init')

    def get_hero_villian(self,s):

        '''
        Take in the name of the owner of the current node and returns hero,villian
        :param current_player:
        :return: []
        '''

        if s.parent.player != "SB":

            return [self.player1,self.player2]

        else:

            return [self.player2,self.player1]

    def get_hero_villian_cip(self, s):

        if s.parent.player == "SB":
            return [s.SB_cip, s.BB_cip]

        if s.parent.player == "BB":
            return [s.BB_cip, s.SB_cip]

    def get_new_state(self, s, a):

        node_children = s.children

        for child in node_children:

            try:

                if list(child.action.keys())[0] == self.actions[a]:

                    return child

                else:
                    continue

            except Exception as e:
                print(e)

        # we should not reach this line of code
        # the function should always be able to return a new state

        raise Exception("get_new_state was not able to find a child with the given action")

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

    def get_state_from_node(self,s,player):

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

        action_matrix = np.zeros((3,4))  # raises x actions matrix

        action_matrix[1][1] = 1

        action_history = []

        curr_node = s

        curr_action = list(curr_node.action.keys())[0]

        while True:

            if curr_node.node_index == 1:
                break

            if curr_node.node_index in [2,3]:

                a_index = self.actions.index(curr_action)

                action_matrix[1][a_index] = 1

            elif curr_node.node_index in [4,5]:

                a_index = self.actions.index(curr_action)

                action_matrix[2][a_index] = 1

            action_history.insert(0,curr_action)

            curr_node = curr_node.parent

        action_flattened = action_matrix.flatten()

        hand_array = np.zeros(3)

        #curr_hand_index = self.deck.index(player.current_hand)

        hand_array[player.current_hand] = 1

        state = np.concatenate((curr_player,action_flattened))

        state = np.concatenate((state,hand_array))

        state = np.reshape(state,[1,17])

        return state

    def get_possible_action(self,s):

        return [self.actions.index(list(child.action.keys())[0]) for child in s.children]

    def set_policy(self):

        '''
        selects either an epsilon greedy policy with prob 1 - ant
        select current policy using the policy network with prob ( ant )
        :return:
        '''

        rand = np.random.random()

        if rand < self.anticipatory_paramter:

            self.policy = "current"

        else:
            self.policy = "e-greedy"

    def select_action(self,s,player):
        '''
        Uses the currently set policy to
        select and return an action
        :return:
        '''

        actions = self.get_possible_action(s)

        if self.policy == "current":

            possible_actions = self.behaviour_policy[s.node_index][player.current_hand].copy()

            for i in range(len(actions)):
                if i not in actions:
                    possible_actions[i] = -100

            return np.argmax(possible_actions)

        else:
            state_vector = self.get_state_from_node(s,player)

            if s.player == "SB":

                dqn_action,type = self.sb_DQN.act(state_vector,actions)

            if s.player == "BB":

                dqn_action,type = self.bb_DQN.act(state_vector,actions)

            if type == 'max':
                self.update(s,dqn_action)

            return dqn_action

    def store_transition(self,current_player,s,action,r,next_state):

        current_state_vector = self.get_state_from_node(s,current_player)

        next_state_vector = self.get_state_from_node(next_state,current_player)

        if s.player == "SB":

            self.sb_DQN.remember(current_state_vector,action,r,next_state_vector,s.is_leaf)

        elif s.player == "BB":

            self.bb_DQN.remember(current_state_vector,action,r,next_state_vector,s.is_leaf)

        else:
            print("Error incorrect player defined on node: store_transition")

    def execute_action(self,action,state):
        '''
        Take in the current state and return the reward and the next state
        :param action:
        :param state:
        :return:
        '''
        return

    def simulate(self,s,i):

        '''
        Used like in mcts to allow an episode of the game to be played out
        :param s:
        :return:
        '''

        if s.is_leaf == True:

            reward = self.get_reward(s)

            #action = self.actions.index(list(s.action.keys())[0])

            #self.update(s,action)

            return reward

        current_player = self.player1 if s.player == "SB" else self.player2

        #current_state_vector = self.get_state_from_node(s)

        action = self.select_action(s,current_player)

        next_state = self.get_new_state(s, action)

        r = self.simulate(next_state, i)

        self.store_transition(current_player,s,action,r[s.player],next_state)

        #self.update(s,next_state,action)

        return r

    def update(self,s,action):

        # first update behaviour strategy

        current_player = self.player1 if s.player == "SB" else self.player2

        next_state = self.get_new_state(s, action)

        self.count_lookup[s.player][s.node_index][current_player.current_hand] += 1

        self.count_lookup[s.player][next_state.node_index][current_player.current_hand] += 1

        current_s_count = self.count_lookup[s.player][s.node_index][current_player.current_hand]

        for child in s.children:

            for hand in self.deck:

                child_count = self.count_lookup[s.player][child.node_index][hand]

                action_index = self.actions.index(list(child.action.keys())[0])

                self.behaviour_policy[s.node_index][current_player.current_hand][action_index] = child_count / current_s_count


        #action_index = self.actions.index(list(action.keys())[0])

        # update - N(s,a) = N(s,a) + policy(a)

        #policy_w = self.behaviour_policy[s.node_index][current_player.current_hand][action]

        #self.behaviour_policy[s.node_index][current_player.current_hand][action] = \

        #   (self.behaviour_policy[s.node_index][current_player.current_hand][action] + 1) / s.count

        # next update Q(s,q)

    def run(self,iterations):

        for i in range(iterations):

            self.set_policy()

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

            s1 = s0.children[0] # start at BB because it is a force check for SB

            self.simulate(s1, i)

            if len(self.sb_DQN.memory) > self.batch_size:
                self.sb_DQN.replay(self.batch_size)

            if len(self.bb_DQN.memory) > self.batch_size:
                self.bb_DQN.replay(self.batch_size)

        return self.behaviour_policy





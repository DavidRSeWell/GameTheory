'''
    MCTS solution to the AKQ game
'''

'''
    Software design:

        Player:

            Props:

                Range: {A,Q,K}

                Chips: number of big blinds

        Tree:

            Props:

                Root:

                Struct: - The current structure of the tree. { a: {b,c}, b: {c,d} ...

            Actions:

                addNode

                getNode

                getChildren

        Nodes:

            Properties:

                Pot: Size of the current pot

                Player: Player whos action it is

                Actions: Possible actions



        Solver:

            Props:

                Type: - Type of solver it is. MCTS ect...

                Tree: - The current tree

                Strategy - The current strategy implemented on the current tree

'''

import random
import numpy as np
from Util.node import InfoNode,PokerNode
from Util.tree import InfoTree
#from RL.MCTS.Model import MCTS

class ExtensiveFormMCTS(object):
    '''
    Outline of what a basic mcts algorith looks like for games of hidden info
    as outlined by David Silver and heinrich
    '''

    def __init__(self):

        pass

    def search(game):

        '''
            While within budget
                Sample initial game state
                simulate(s_o)
            end

            return policy
        '''
        pass

    def rollout(s):
        '''
            takes in a state
            gets action based off of a rollout policy - i.e random actions, ect...
            new state s' from G(s,a) - transition simulator
            return simulate(s')
        '''
        pass

    def simulate(s):
        '''
            Takes in a state

            if state.terminal == True:
                return reward

            Player = player(s)
            if Player.out_of_tree == True:
                return rollout(s)
            InfoState = information_function(s) maps state to info state
            if InfoState not in PlayerTree:
                Expand(PlayerTree,InfoState)
                a = rollout_policy
                Player.out_of_tree = True
            else:
                a = select(InfoState)
            s' = G(s,a)
            r = simulate(s')
            update(InfoState,a,r)
            return r
        '''


        pass

    def select_uct(u_i):
        '''
            select action that maximizes
            Q(u,a) + c sqrt( log(N(u))/N(u,a) )

        '''
        pass

    def update(u_i, a, r):
        '''
        N(u_i) += 1
        N(u,a) += 1
        Q(u,a) += (r - Q(u,a)) / N(u,a)
        '''
        pass

class AKQPlayer(object):

    def __init__(self,name,info_tree,starting_stack):

        self.name = name

        self.info_tree = info_tree

        self.policy = {}

        self.out_of_tree = False

        self.current_hand = None

        self.starting_stack = starting_stack

class AKQGameState(object):

    '''
        class used for simulating a simple AKQ poker game
        The game state needs to deal random cards to each player
        and to track the button
    '''

    def __init__(self,game_tree):

        self.player1 = None

        self.player2 = None

        self.iter_count = 0

        self.deck = [3,2,1]

        self.game_tree = game_tree # the full game tree for for the information trees to reference

        self.replay_data = []

        self.adaptive_constant  = 0.8

        self.behavior_policy = {
            "SB": {},
            "BB":{}
        }

        self.init_game()

    def init_game(self):

        SB_tree = InfoTree() # info tree

        chance_node = PokerNode(player="chance",SB_cip=0.0,BB_cip=0.0)

        SB_tree.set_root(chance_node)

        BB_tree = InfoTree() # info tree

        game_root = self.game_tree.get_root()

        BB_root_node = PokerNode(game_root.player,SB_cip=0,BB_cip=0)

        BB_tree.set_root(BB_root_node)

        self.player1 = AKQPlayer(name="SB",info_tree=SB_tree,starting_stack=1)

        self.player2 = AKQPlayer(name="BB",info_tree=BB_tree,starting_stack=1)

        self.player1.policy[0] = {}

        self.player2.policy[0] = {}

        for curr_node in self.game_tree.nodes:

            if curr_node.player == 'Leaf':
                continue

            actions = [list(node.action.keys())[0] for node in curr_node.children]

            self.behavior_policy[curr_node.player][curr_node.node_index] = {}

            for action in actions:

                self.behavior_policy[curr_node.player][curr_node.node_index][action] = {'A':0.0,'K':0.0,'Q':0.0}

    def deal_hand(self):

        return random.choice(self.deck)

    def get_hand_string(self,hand):

        if hand == 3:
            return "A"

        elif hand == 2:
            return "K"

        else:
            return "Q"

    def get_hero_villian(self,s):

        '''
        Take in the name of the owner of the current node and returns hero,villian
        :param current_player:
        :return: []
        '''

        if s.parent.player == "SB":

            return [self.player1,self.player2]

        else:

            return [self.player2,self.player1]

    def get_hero_villian_cip(self, s):

        if s.parent.player == "SB":

            return [s.SB_cip, s.BB_cip]

        else:
            return [s.BB_cip, s.SB_cip]

    def get_info_state(self,current_player, s):

        '''

        info state for the AKG game is (actions,hand)
        actions are all previous actions to this node

        For the AKQ I dont think we need to keep track of
        actions so the info stat is only going to have
        the additional hand value

        Append
        :param s:
        :return:

        '''

        if s.parent == None:
            # this is a root node and the parent will be chance

            NewInfoNode = InfoNode(current_player.current_hand, player=s.player,action=s.action, parent=current_player.info_tree.get_root(),
                                   SB_cip=s.SB_cip, BB_cip=s.BB_cip,is_leaf=s.is_leaf)

            return NewInfoNode

        # need to make a copy of the parent cannot pass as reference

        new_parent = PokerNode(s.parent.player,parent=s.parent.parent,SB_cip=s.parent.SB_cip,BB_cip=s.parent.BB_cip,action=s.parent.action)

        NewInfoNode = InfoNode(current_player.current_hand,player=s.player ,action=s.action, parent=new_parent,
                               SB_cip=s.SB_cip, BB_cip=s.BB_cip,is_leaf=s.is_leaf)

        return NewInfoNode

    def get_new_state(self,s,a):

        node_children = s.children

        for child in node_children:

            if child.action == a:

                return child

            else:
                continue

        # we should not reach this line of code
        # the function should always be able to return a new state

        raise Exception("get_new_state was not able to find a child with the given action")

    def get_child_info(self,u_i,a):

        for child in u_i.children:
            if u_i.action == a:
                return child

        # should not reach this location

        raise Exception("g_child_info parent does not have child with action: " + str(a))

    def reward(self,s):

        '''

        Takes in a leaf node and returns the reward to each player
        :param s:
        :return:

        '''

        r = {"SB":0,"BB":0}

        hero, villian = self.get_hero_villian(s)

        hero_cip, villian_cip = self.get_hero_villian_cip(s)

        current_pot = 2.0 + s.SB_cip + s.BB_cip

        action_type = list(s.action.keys())[0]

        if action_type == "fold":
            # the parent folded so the current player gets the pot
            r[hero.name] = hero.starting_stack - hero_cip

            r[villian.name] = current_pot + (villian.starting_stack - villian_cip)


        elif action_type == "check":

            # evaluate winner
            if (hero.current_hand > villian.current_hand):
                # SB wins
                r[hero.name] = current_pot + (hero.starting_stack - hero_cip)

                r[villian.name] = villian.starting_stack - villian_cip

            else:

                r[villian.name] = current_pot + (villian.starting_stack - villian_cip)

                r[hero.name] = hero.starting_stack - hero_cip


        elif action_type == "call": # same as check?

            # evaluate winner
            if (hero.current_hand > villian.current_hand):
                # SB wins
                r[hero.name] = current_pot + (hero.starting_stack - hero_cip)

                r[villian.name] = villian.starting_stack - villian_cip

            else:

                r[villian.name] = current_pot + (villian.starting_stack - villian_cip)

                r[hero.name] = hero.starting_stack - hero_cip

        return r

    def rollout(self,s):

        '''
            takes in a state
            gets action based off of a rollout policy - i.e random actions, ect...
            new state s' from G(s,a) - transition simulator
            return simulate(s')
        '''

        new_action = self.rollout_policy(s)

        new_state = self.get_new_state(s,new_action)

        return self.simulate(new_state) # recursive call

    def rollout_policy(self,s):

        '''

        Get a new action for the player based off the rollout policy
        :param s:
        :return:

        '''

        # for now just going to use a random rollout policy

        #possible_actions = [child.action for child in s.children]

        # just return the child node

        try:

            return random.choice(s.children).action

        except Exception as e:
            print("Error at rollout_policy: " + str(e))

    def select_uct(self,u_i):

        '''
            select action that maximizes

            if random U [0,1] < mu

                Q(u,a) + c sqrt( log(N(u))/N(u,a) )

            else
                policy = N(u,a) / N(u)
                return a ~ p
        '''

        N_U = u_i.visit_count

        if (np.random.random() < self.adaptive_constant):

            if N_U == 0:
                print("Visit count = 0!")

            current_max_action = None

            current_max = -1

            current_player = self.player1 if u_i.player == "SB" else self.player2

            info_policy = current_player.policy[u_i.node_index]

            for action in info_policy.keys():

                child_ev_value = info_policy[action]['ev']

                child_visit_count = info_policy[action]['count']

                score = 0

                if child_visit_count == 0:

                    score = current_max + 1000

                else:
                    score = child_ev_value + 1.5*np.sqrt(np.log(N_U)/child_visit_count)

                if score > current_max:

                    current_max = score

                    current_max_action = action

            if current_max_action == "check" or current_max_action == "fold":
                return {current_max_action: 0}
            else:
                return {current_max_action: 1}

        else:

            current_player = self.player1 if u_i.player == "SB" else self.player2

            action_p = []

            actions = []

            for action in current_player.policy[u_i.node_index].keys():

                n_a_count = current_player.policy[u_i.node_index][action]['count']

                action_p.append(float(n_a_count/N_U))

                actions.append(action)

            choose_action = np.random.choice(actions,1,p=action_p)[0]


            if choose_action == "check" or choose_action == "fold":

                return {choose_action: 0}
            else:
                return {choose_action: 1}

    def simulate(self,s):

        self.iter_count += 1

        '''

            Takes in a state

            if state.terminal == True:
                return reward

            Player = player(s)
            if Player.out_of_tree == True:
                return rollout(s)
            InfoState = information_function(s) maps state to info state
            if InfoState not in PlayerTree:
                Expand(PlayerTree,InfoState)
                a = rollout_policy
                Player.out_of_tree = True
            else:
                a = select(InfoState)
            s' = G(s,a)
            r = simulate(s')
            update(InfoState,a,r)
            return r

        '''

        if s.is_leaf == True:

            return self.reward(s)

        current_player = self.player1 if s.player == "SB" else self.player2

        if current_player.out_of_tree == True:

            return self.rollout(s)

        infostate = self.get_info_state(current_player,s)

        action = None

        action_select_type = "uct"

        if not current_player.info_tree.node_in_tree(infostate):

            current_player.info_tree.add_node(infostate)

            action = self.rollout_policy(s)

            action_select_type = "rollout"

            current_player.out_of_tree = True

            current_player.policy[infostate.node_index] = {}

            for child in s.children:

                new_action = list(child.action.keys())[0]

                current_player.policy[infostate.node_index][new_action] = {}

                current_player.policy[infostate.node_index][new_action]['count'] = 0

                current_player.policy[infostate.node_index][new_action]['ev'] = 0

        else:

            infostate = current_player.info_tree.get_tree_node(infostate)

            action = self.select_uct(infostate)



        next_state = self.get_new_state(s,action)

        ###############
        # REPLAY DATA
        ###############

        action_type = list(action.keys())[0]

        r = self.simulate(next_state)

        replay_data = [current_player.name,s.node_index,self.get_hand_string(current_player.current_hand),action_type,r[current_player.name]]

        self.replay_data.append(replay_data)

        self.update(current_player,s,infostate,action,r)

        return r

    def update(self,current_player,s,u_i, a, r):

        '''
        N(u_i) += 1
        N(u,a) += 1
        Q(u,a) += (r - Q(u,a)) / N(u,a)
        '''

        u_i.visit_count += 1

        player_reward = r[current_player.name]

        action_type = list(a.keys())[0]

        current_player.policy[u_i.node_index][action_type]['count'] += 1

        current_count = current_player.policy[u_i.node_index][action_type]['count']

        current_ev = current_player.policy[u_i.node_index][action_type]['ev']

        update = (player_reward - current_ev)/current_count

        current_player.policy[u_i.node_index][action_type]['ev'] += update

        # update policy simply N(u,a) / N(u)

        string_hand = self.get_hand_string(u_i.player_hand)

        for action in list(current_player.policy[u_i.node_index].keys()):

            N_U_A = current_player.policy[u_i.node_index][action]['count']

            self.behavior_policy[current_player.name][s.node_index][action][string_hand] = N_U_A / float(u_i.visit_count)

    def run(self,num_iterations):

        for i in range(num_iterations):

            #self.iter_count = i

            self.deck = [3,2,1] # reshuffle the cards yo

            self.player1.out_of_tree = False

            self.player2.out_of_tree = False

            # deals cards to each player

            sb_card = self.deal_hand()

            self.player1.current_hand = sb_card

            self.deck.remove(sb_card)

            bb_card = self.deal_hand()

            self.player2.current_hand = bb_card

            s0 = self.game_tree.get_root()

            self.simulate(s0)


        return [self.player1.policy,self.player2.policy]








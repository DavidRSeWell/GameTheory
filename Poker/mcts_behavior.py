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
from Util import akq_util
#from RL.MCTS.Model import MCTS
from Util.node import InfoNode,PokerNode
from Util.tree import InfoTree



AQK_HANDS = ["A","K","Q"]

AQK_HAND_NUMS = [0,1,2]

class MCTSStrategyProfile:

    '''
    A strategy profile is defined as the set of strategies for each player in the game
    for a simple 2 player poker game such as we will use in this example there will just
    be 2 different strategies represented where each strategy itself will be the distribution
    of actions that a player takes at every information state in the game.

    Example:
        For the AKQ game the strategy profile will look something like
        Node 0: {sb: { bet: {A:1.0,K:0.0,Q:0.33}, check: {A:0.0,K:1.0,Q:0.66} } , bb: {.....

        So the small blind has a distribution for each one of his actions at the first node and this
        would continue for each player for every node.

        The distribution of hands for each player in poker is generally referred to his/her range. In the
        literature this is referred to as the players behavioral strategy.

        In this code base I will simply use Range to denote a players distribution of hands at a give node in
        the game tree. This is simpler and is easier to communicate to a more general audience
    '''

    def __init__(self,tree):

        self.tree = tree

        self.ranges = [{'A':1.0,'K':1.0,'Q':1.0} for node in range(self.tree.get_num_nodes())]

        self.policy = {
            "SB":{},
            "BB":{}
        }

        self.sb_starting_range = {'A':1.0,'K':1.0,'Q':1.0}

        self.bb_starting_range = {'A':1.0,'K':1.0,'Q':1.0}

        self.initialize()

    def initialize(self):
        '''
        Set all ranges in Strategy pair
        :return:
        '''
        self.initialize_helper(0, 1.0, 1.0)

        for i in range(len(self.ranges)):

            self.policy["SB"][i] = {}

            self.policy["BB"][i] = {}

            self.policy["SB"][i]["ev"] = [0,0,0]

            self.policy["SB"][i]["count"] = 0

            self.policy["BB"][i]["ev"] = [0,0,0]

            self.policy["BB"][i]["count"] = 0

        print("Done initializing")

    def scale_range(self,range,scale):

        for key in list(range.keys()):

            range[key] *= scale

        return range

    def initialize_helper(self, icurrDecpt, sbScale, bbScale):

        children = self.tree.nodes[icurrDecpt].children

        numChildren = len(children)

        if numChildren == 0:
            return

        if self.tree.nodes[icurrDecpt].player == "SB":

            sbScale /= numChildren

            for iChild in children:

                self.ranges[iChild.node_index] = self.scale_range(self.ranges[iChild.node_index],sbScale)

        elif self.tree.nodes[icurrDecpt].player == "BB":

            bbScale /= numChildren

            for iChild in children:

                self.ranges[iChild.node_index] = self.scale_range(self.ranges[iChild.node_index],bbScale)


        for iChild in children:

            self.initialize_helper(iChild.node_index, sbScale, bbScale)

    def update_strategy_profile(self,br_profile,n):
        '''
        Uses the calculated best response profile to update the current
        strategy profile.
        :param br_profile: list of strategies
        :return: None
        '''

        for player in list(br_profile.keys()):

            for node_index in list(br_profile[player].keys()):

                self.ranges[node_index] = self.update_range_brown(self.ranges[node_index],br_profile[player][node_index],n)

    def update_strategy_profile_player(self,br_profile,player,n):
        '''
        Uses the calculated best response profile to update the current
        strategy profile.
        :param br_profile: list of strategies
        :return: None
        '''

        for node_index in list(br_profile[player].keys()):

            self.ranges[node_index] = self.update_range_basic(self.ranges[node_index],br_profile[player][node_index],n)

    def get_update_fraction(self,r1,hand,n,direction):

        hand = akq_util.get_hand_string(hand)

        fraction = 1 - 1 / (n + 2)

        update_amount = r1[hand] * fraction

        if direction == 1:  # cant increase range frac if were already using 100%

            if r1[hand] == 1:

                return 0 # dont update the hand at all

            elif (r1[hand] + update_amount) > 1:
                return 1 - r1[hand]

            else:
                return update_amount

        if direction == -1:

            if r1[hand] == 0:
                return 0

            elif (r1[hand] - update_amount) < 0:
                return r1[hand]

            else:
                return update_amount

    def update_range_basic(self,r1, hand, update_amount,direction):
        '''

        :param r1:
        :param r2:
        :param n:
        :return:
        '''

        hand = akq_util.get_hand_string(hand)

        r1[hand] +=  update_amount * direction

    def get_starting_range(self,player):

        '''

        :param player:
        :return:
        '''

        if player == "SB":
            return self.sb_starting_range

        elif player == "BB":
            return self.bb_starting_range

        else:
            print("Error incorrect player name in get_starting_range")

    def get_player_starting_stack(self,player):

        if player == "SB":

            return self.tree.sb_starting_stack

        elif player == "BB":

            return self.tree.bb_starting_stack

        else:
            print("Incorrect player name passed to method:get_player_starting_stack")

    def get_recent_range(self,dec_pt,player):

        '''
        Given the current decesion point return the most recent range of a player
        :param player:
        :return:
        '''

        if dec_pt.node_index == 0:

            return self.get_starting_range(player)


        if dec_pt.player == "Leaf":

            if player == dec_pt.parent.player:

                return self.ranges[dec_pt.node_index]

            else:

                return self.ranges[dec_pt.parent.node_index]

        '''curr_index = dec_pt.node_index

        curr_player = dec_pt.player

        while(player != curr_player):

            curr_player = dec_pt.parent.player

            curr_index = dec_pt.parent.node_index

        return self.ranges[curr_index]'''

    def get_player_cip(self,dec_pt,player):

        if player == "SB":
            return dec_pt.SB_cip

        elif player == "BB":
            return dec_pt.BB_cip

        else:
            print("Error incorrect player name in get_player_cip")

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

        self.current_hand = None

        self.starting_stack = starting_stack

class AKQMixedMcts(object):
    '''
    class used for simulating a simple AKQ poker game
    The game state needs to deal random cards to each player
    and to track the button
    '''

    def __init__(self,game_tree,strat_profile):

        self.player1 = None

        self.player2 = None

        self.iter_count = 0

        self.deck = [0,1,2]

        self.game_tree = game_tree # the full game tree for for the information trees to reference

        self.strat_profile = strat_profile

        self.init_game()

    def deal_hand(self):

        return random.choice(self.deck)

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

    def get_hero_villian_cip(self,s):

        if s.parent.player == "SB":

            return [s.SB_cip,s.BB_cip]

        if s.parent.player == "BB":

            return [s.BB_cip,s.SB_cip]

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

        children = []

        for child in node_children:

            if child.action == a:

                children.insert(0,child)

            else:
                children.append(child)

        return children
        # we should not reach this line of code
        # the function should always be able to return a new state

        raise Exception("get_new_state was not able to find a child with the given action")

    def get_child_info(self,u_i,a):

        for child in u_i.children:
            if u_i.action == a:
                return child

        # should not reach this location

        raise Exception("g_child_info parent does not have child with action: " + str(a))

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

    def reward(self,s):

        '''

        Takes in a leaf node and returns the reward to each player
        :param s:
        :return:

        '''

        r = {"SB":0,"BB":0}

        villian,hero = self.get_hero_villian(s)

        villian_cip,hero_cip = self.get_hero_villian_cip(s)

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


        elif action_type == "call": # same as check?

            # evaluate winner
            if (hero.current_hand < villian.current_hand):
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

        new_state = self.get_new_state(s,new_action)[0]

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

    def select_uct(self,u_i,hand):

        '''
            select action that maximizes
            Q(u,a) + c sqrt( log(N(u))/N(u,a) )

        '''

        # Get fraction of the range of the first
        # child in the range of the current node

        if(u_i.node_index == 0):
            return u_i.children[0].action

        child_node = u_i.children[0]

        hand_string = akq_util.get_hand_string(hand)

        p = self.strat_profile.ranges[child_node.node_index][hand_string] # get the fraction of the hand that the potential node contains

        random_select = np.random.random()

        action = child_node.action

        if random_select > p: # if random > p take other action otherwise keep child action

            try:
                action = u_i.children[1].action

            except Exception as e:
                print(e)

        return action

    def simulate(self,s,i):

        self.iter_count += 1

        if s.is_leaf == True:

            return self.reward(s)

        current_player = self.player1 if s.player == "SB" else self.player2

        action = self.select_uct(s,current_player.current_hand)

        next_state = self.get_new_state(s,action)[0]

        r = self.simulate(next_state,i)

        self.update(current_player,s,action,r,i)

        return r

    def update(self,current_player,u_i, a, r,n):

        '''
        N(u_i) += 1
        N(u,a) += 1
        Q(u,a) += (r - Q(u,a)) / N(u,a)
        '''

        if u_i.node_index == 0:
            return

        player_reward = r[current_player.name]

        next_states = self.get_new_state(u_i,a)

        if len(next_states) > 1:

            next_state,other_child = next_states[0],next_states[1]

        else:
            next_state, other_child = next_states[0], None

        next_state_index = next_state.node_index

        self.strat_profile.policy[current_player.name][next_state_index]["count"] += 1

        current_count = self.strat_profile.policy[current_player.name][next_state_index]["count"]

        current_ev = self.strat_profile.policy[current_player.name][next_state_index]["ev"][current_player.current_hand]

        update = (player_reward - current_ev)/current_count

        self.strat_profile.policy[current_player.name][next_state_index]['ev'][current_player.current_hand] += update

        direction = player_reward / np.abs(player_reward) # if player_reward < 0 -> direction = -1

        if player_reward == 0.0:

            direction = -1

        if u_i.node_index == 2:
            print("node 2")

        update_amount = self.strat_profile.get_update_fraction(self.strat_profile.ranges[next_state_index],current_player.current_hand,n,direction)

        self.strat_profile.update_range_basic(self.strat_profile.ranges[next_state_index],current_player.current_hand,update_amount,direction)

        if other_child:

            self.strat_profile.update_range_basic(self.strat_profile.ranges[other_child.node_index],current_player.current_hand,update_amount,-direction)

    def run(self,num_iterations):

        for i in range(num_iterations):

            #self.iter_count = i

            self.deck = [0,1,2] # reshuffle the cards yo

            self.player1.out_of_tree = False

            self.player2.out_of_tree = False

            # deals cards to each player

            sb_card = self.deal_hand()

            self.player1.current_hand = sb_card

            self.deck.remove(sb_card)

            bb_card = self.deal_hand()

            self.player2.current_hand = bb_card

            s0 = self.game_tree.get_root()

            self.simulate(s0,i)


        return self.strat_profile






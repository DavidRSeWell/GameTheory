'''
    We are going ot apply the XFP algorithm to the AKQ game
    XFP: Extensive-form fictitious play algorithm

    Func:
        def FP(tree):
            init strategy
            j = 1
            while in budget do
                b(t +1) = ComputeBR(strategy)
                strategy(t + 1) = Update Avg Strat
                j += 1

            end
            return strategy
        end

        def ComputeBR(strategy)
            recurisve game tree
            return b(t + 1)
        end

        def UpdateAvgStrat(strategy(t), b(t+1))
            Theorem 7 from Heinrich paper
            or something simpler
           return strateg(t + 1)

        end

'''

import numpy as np



AQK_HANDS = ["A","K","Q"]
AQK_HAND_NUMS = [0,1,2]

class StrategyProfile:

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

        self.ev = {
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

            self.ev["SB"][i] = [0,0,0]
            self.ev["BB"][i] = [0,0,0]

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

                self.ranges[node_index] = self.update_range_basic(self.ranges[node_index],br_profile[player][node_index],n)

    def update_range_basic(self,r1, r2, n):
        '''

        :param r1:
        :param r2:
        :param n:
        :return:
        '''
        fraction = 1 - 1 / (n + 2.0)

        for hand in AQK_HANDS:

            r1[hand] =  r1[hand]* (fraction) + (r2[hand]) * (1 - fraction)

        return r1

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

        parent_node_index = dec_pt.parent.node_index

        return self.ranges[parent_node_index]

    def get_player_cip(self,dec_pt,player):

        if player == "SB":
            return dec_pt.SB_cip

        elif player == "BB":
            return dec_pt.BB_cip

        else:
            print("Error incorrect player name in get_player_cip")


##############################
### MAX EV/Strat FUNCTIONS ###
##############################

def get_max_strat(player,strategy_profile):
    '''
    based upon the maximum ev for
    :param player:
    :param strategy_profile:
    :return:
    '''

    result = {}

    if player == "SB":

        get_max_strat_helper(player,strategy_profile,strategy_profile.tree.get_root(),strategy_profile.sb_starting_range,result)

    elif player == "BB":

        get_max_strat_helper(player,strategy_profile,strategy_profile.tree.get_root(),strategy_profile.bb_starting_range,result)

    else:

        print("Invalid player")

    return result

def get_max_strat_helper(player,strategy_profile,current_node,current_range,result):

    current_index = current_node.node_index

    if current_node.player == player:

        for child in current_node.children:

            result[child.node_index] = {"A": 0, "K": 0, "Q": 0}

        for hand in AQK_HAND_NUMS:

            hand_string = get_hand_string(hand)

            if strategy_profile.ranges[current_index][hand_string] > 0: # only look at hand if we have it in current range

                # we need to look at the expected value of each hand for every action
                # the best response strategy will be the one the picks the strategy
                # with the highest expected value

                max_ev = -1

                max_ev_node = None

                for child in current_node.children:

                    child_ev = strategy_profile.ev[player][child.node_index][hand]

                    if child_ev > max_ev:

                        max_ev = child_ev

                        max_ev_node = child

                result[max_ev_node.node_index][hand_string] = strategy_profile.ranges[current_index][hand_string]

    for child in current_node.children:

        get_max_strat_helper(player,strategy_profile,child,current_range,result)

def calc_max_ev(strat_pair,hero,villain):

    calc_max_ev_helper(strat_pair.tree.get_root(),strat_pair,hero,villain)

def calc_max_ev_helper(dec_pt,strat_pair,hero,villain):

    curr_player = dec_pt.player

    if curr_player == "Leaf":

        calc_max_ev_leaf(dec_pt,strat_pair,hero,villain)

    elif curr_player == hero:

        calc_max_hero(dec_pt,strat_pair,hero,villain)

    elif curr_player == villain:

        calc_max_villain(dec_pt,strat_pair,hero,villain)


    else: # for AKQ game there are no nature nodes
        print("Error: incorrect dec pt")

def calc_max_ev_leaf(dec_pt,strat_pair,hero,villain):

    curr_index = dec_pt.node_index

    action = list(dec_pt.action.keys())[0]

    if action == "call" or action == "check":

        # assume the same starting stack for both players for the AKQ game
        # ev = (S - cip) + equity*(pot)

        curr_range = strat_pair.ranges[dec_pt.node_index]

        villain_range = strat_pair.get_recent_range(dec_pt,villain)

        for hand in list(curr_range.keys()):

            hand_number = get_hand_number(hand)

            hand_quity = hand_v_range_equity(hand,villain_range)

            ev = (strat_pair.get_player_starting_stack(hero) - strat_pair.get_player_cip(dec_pt,hero)) + hand_quity*(dec_pt.SB_cip + dec_pt.BB_cip)

            strat_pair.ev[hero][curr_index][hand_number] = ev

    elif action == "fold":

        if dec_pt.parent.player == villain: # villain folded

            # ev = (StartStack + villain cip)

            ev = strat_pair.get_player_starting_stack(hero) + strat_pair.get_player_cip(dec_pt,villain)

            strat_pair.ev[hero][curr_index] = np.ones_like(strat_pair.ev[hero][curr_index])*ev

        elif dec_pt.parent.player == hero: # hero folded

            # ev = (StartStack - hero cip)

            ev = strat_pair.get_player_starting_stack(hero) - strat_pair.get_player_cip(dec_pt,hero)

            strat_pair.ev[hero][curr_index] = np.ones_like(strat_pair.ev[hero][curr_index])*ev

        else:
            print("invalid player name at node: " + str(dec_pt.node_index))

    else:
        print("Invalid action in calc_max_leaf")

def calc_max_hero(dec_pt,strat_pair,hero,villain):
    '''
    The ev at a hero node for max exploit play will just be the ev
    of the action with the highest ev. So get max ev play
    :param tree:
    :param dec_pt:
    :param strat_pair:
    :param hero:
    :param villain:
    :return:
    '''

    curr_index = dec_pt.node_index

    strat_pair.ev[hero][curr_index] = 0

    for child_dec_pt in dec_pt.children:

        calc_max_ev_helper(child_dec_pt,strat_pair,hero,villain)

        strat_pair.ev[hero][curr_index] = np.maximum(strat_pair.ev[hero][curr_index],strat_pair.ev[hero][child_dec_pt.node_index])

def calc_max_villain(dec_pt,strat_pair,hero,villain):

    '''
    The ev for hero at a villain node is (Freq of villain action) * (hero ev vs that action)
    :param tree:
    :param dec_pt:
    :param strat_pair:
    :param hero:
    :param villain:
    :return:
    '''

    curr_index = dec_pt.node_index

    for child_dec_pt in dec_pt.children:

        calc_max_ev_helper(child_dec_pt,strat_pair,hero,villain)


    for hand in AQK_HANDS: # for each hand in the heroes range the villains range will have a diff frequency

        hand_index = get_hand_number(hand)

        total_possible = 0

        villain_frequency = {}

        for child_dec_pt in dec_pt.children:

            # first get the frequency for each action

            villain_range = strat_pair.ranges[child_dec_pt.node_index]

            villain_combos = get_num_hands(hand,villain_range)

            total_possible += villain_combos

            villain_frequency[child_dec_pt.node_index] = villain_combos

        # now get the ev of the hero at that node

        for child_dec_pt in dec_pt.children:

            ev = (strat_pair.ev[hero][child_dec_pt.node_index][hand_index])*villain_frequency[child_dec_pt.node_index]/total_possible

            strat_pair.ev[hero][curr_index][hand_index] = ev


##########################
### AKQ UTIL FUNCTIONS ###
##########################

def hand_v_range_equity(hand,range):
    '''
    For this game the hand first range equity is quite straightforward
    :param hand:
    :param range:
    :return:
    '''

    if hand == "A":

        return 1

    elif hand == "K":

        return range["Q"]

    else:
        # the hand must be a Q and the Q never has any equity
        return 0

def get_num_hands(hero_hand,range):

    '''
    Calculates the number of hands in a range based given
    the hand currently held by the hero

    :param hand:
    :param range:
    :return:
    '''

    num_hands = 0

    for hand in range:
        if hand != hero_hand:
            num_hands += range[hand]

    return num_hands

def get_hand_number(hand):

    if hand == "A":
        return 0

    elif hand == "K":
        return 1

    else:
        return 2

def get_hand_string(hand):

    if hand == 0:
        return "A"

    elif hand == 1:
        return "K"

    else:
        return "Q"

def calculate_best_response(strategy_profile):
    '''
    Looks at the current strategy profile on the object and calculates and
    returns a best response profile
    :return:
    '''

    best_response_profile = {
        "SB": {},
        "BB":{}
    } # set of BR strats for each player

    # calculate max ev for SB
    calc_max_ev(strategy_profile,"SB","BB")

    br_sb_strategy = get_max_strat("SB",strategy_profile)

    best_response_profile["SB"] = br_sb_strategy

    calc_max_ev(strategy_profile,"BB","SB")

    br_bb_strategy = get_max_strat("BB",strategy_profile)

    best_response_profile["BB"] = br_sb_strategy

    return best_response_profile

def FP(tree,n_iter):

    # first intit strategy profile
    strategy_profile = StrategyProfile(tree)

    j = 1
    while(j < n_iter + 1):

        # first find best Response profile
        best_response_profile = calculate_best_response(strategy_profile)

        # Now update each
        strategy_profile.update_strategy_profile(best_response_profile,j)

        j+=1


    return strategy_profile
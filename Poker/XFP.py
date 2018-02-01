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

        self.sb_starting_range = {'A':1.0,'K':1.0,'Q':1.0}

        self.bb_starting_range = {'A':1.0,'K':1.0,'Q':1.0}

        self.initialize()

    def initialize(self):
        '''
        Set all ranges in Strategy pair
        :return:
        '''
        self.initialize_helper(0, 1.0, 1.0)

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

    def calculate_best_respone(self):

        '''
        Looks at the current strategy profile on the object and calculates and
        returns a best response profile
        :return:
        '''

        return {}

    def update_strategy_profile(self,br_profile):
        '''
        Uses the calculated best response profile to update the current
        strategy profile.
        :param br_profile: list of strategies
        :return: None
        '''

def FP(tree,n_iter):

    # first intit strategy profile
    strategy_profile = StrategyProfile(tree)

    while(n_iter > 0):

        # first find best Response profile
        best_response_profile = strategy_profile.calculate_best_respone()
        # Now update each

        n_iter -= 1

def get_max_strat(tree,player,strategy_profile):

    result = {}

    if player == "SB":

        get_max_strat_helper(tree,player,tree.get_root(),strategy_profile.sb_starting_range,result)

    elif player == "BB":

        get_max_strat_helper(tree,player,tree.get_root(),strategy_profile.sb_starting_range,result)

    else:

        print("Invalid player")

    return result

def get_max_strat_helper(tree,player,current_node,current_range,result):


    if current_node.player == player:

        for child in current_node.children:

            # we need to look at the expected value of each hand for every action
            # the best response strategy will be the one the picks the strategy
            # with the highest expected value

            # get_expected_value_at_node() how to do this?

            pass


    else:
        for child in current_node.children:

            get_max_strat_helper(tree,player,child,current_range,result)



def calc_max_ev(tree,strat_pair,hero,villian):


    calc_max_ev_helper(tree,0,strat_pair,hero,villian)



def calc_max_ev_helper(tree,dec_pt,strat_pair,hero,villian):

    curr_player = dec_pt.player

    if dec_pt.is_leaf:
        pass

    elif curr_player == hero:
        pass

    elif curr_player == villian:
        pass

    else: # for AKQ game there are no nature nodes
        print("Error: incorrect dec pt")



def calc_max_leaf(tree,dec_pt,strat_pair,hero,villian):

    if dec_pt.action == "call":
        # ev = (S - cip) + equity*(pot)
        ev = strat_pair.sb
        pass

    elif dec_pt.action == "fold":

        if dec_pt.player == hero: # villian folded
            # ev = (StartStack + villian cip)
            pass

        elif dec_pt.player == villian: # hero folded
            # ev = (StartStack - hero cip)
            pass
        else:
            print("invalid player name at node: " + str(dec_pt.node_index))
    else:
        print("Invalid action in calc_max_leaf")


def calc_max_hero(tree,dec_pt,strat_pair,hero,villian):
    pass


def calc_max_villian(tree,dec_pt,strat_pair,hero,villian):
    pass


def calc_max_(tree,dec_pt,strat_pair,hero,villian):
    pass


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



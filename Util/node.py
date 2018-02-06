
class PokerNode(object):

    '''

        General Tree structure:
                A
                |
              / | |
            A1 A2 A3

        Each node is itself a Tree
        whith attributes player,pot

    '''

    def __init__(self,player,parent=None,action=None,node_index=None,
                 is_leaf=False,SB_cip=None,BB_cip=None):

        self.node_index = node_index

        self.player = player

        self.action = action

        self.children = []

        self.parent = parent

        self.is_leaf = is_leaf

        self.SB_cip = SB_cip

        self.BB_cip = BB_cip

class InfoNode(object):

    '''

    Nodes used in incomplete information games

    '''

    def __init__(self, player_hand,player,action,parent,SB_cip,BB_cip,is_leaf=False):

        self.node_index = None

        self.player_hand = player_hand

        self.player = player

        self.action = action

        self.children = []

        self.parent = parent

        self.visit_count = 0  # number of times the node has been visited in MCTS

        self.current_ev_value = 0  # total value of node for current player

        # self.current_ucb1 = 0 # average ev + 2 * sqrt ( ln (total iterations) / visit_count)

        self.is_leaf = is_leaf

        self.SB_cip = SB_cip # the number of cip for p1

        self.BB_cip = BB_cip # the number of cip for p2






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

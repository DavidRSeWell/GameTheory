import  sys
from Util.node import PokerNode


class GameState(object):

    '''
        A game takes in actions from the environment
        and maintains the tree,nodes and graph based off these
    '''

    def __init__(self,tree,players,graph=None,name='',solver=None):

        self.name = name

        self.players = players

        self.tree = tree

        self.graph = graph

        self.solver = solver

        self.assert_init()

    def assert_init(self):

        assert (len(self.players) > 0)

        print("Finished initializing the game state")

    def assert_leaf(self,current_node,action):

        action_type = list(action.keys())[0]

        if current_node.action == action and action_type == "check":
            return True

        elif action_type == "call":
            return True

        elif action_type == "fold":
            return True

        else:
            return False

    def set_root(self,player,init_SB_cip,init_BB_cip):

        new_node = PokerNode(player,SB_cip=init_SB_cip,BB_cip=init_BB_cip,node_index=0)

        self.tree.set_root(new_node)

    def new_action(self,current_index,player,action):

        '''
        Method that takes in the intended actiont to take
        and adds it to the game state.
        :param player: string
        :param action: dict : {type:'',amount: float}
        :return:
        '''

        current_node = self.tree.get_node(current_index)

        SB_cip,BB_cip = self.get_new_cip(current_node,player,action)

        is_leaf = self.assert_leaf(current_node,action)

        opponent = self.get_opponent(player)

        if is_leaf:

            opponent = "Leaf"

        new_node = PokerNode(opponent,parent=current_node,action=action,SB_cip=SB_cip,BB_cip=BB_cip,is_leaf=is_leaf)

        self.tree.add_node(new_node)

    def get_new_cip(self,node,player,action):

        '''
        Return new pot size based off a new action
        :param pot:
        :param action:
        :return:
        '''

        action_type = list(action.keys())[0]

        amount = list(action.values())[0]

        SB_cip = node.SB_cip

        BB_cip = node.BB_cip

        if action_type in  ("bet","raise","call"):

            try:
                assert amount > 0

                if player == "SB":

                    SB_cip += amount

                else:
                    BB_cip += amount

            except Exception as e:

                print("Error: Bet amount must be > 0")

        return [SB_cip,BB_cip]

    def get_opponent(self,player):

        if self.players[0] == player:

            return self.players[1]

        else:
            return self.players[0]
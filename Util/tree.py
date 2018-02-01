


class Tree:

    def __init__(self):

        self.nodes = []

        self.node_index = 0

    def set_root(self,node):

        '''
        Init the tree with the first player in the list as the
        initial player for the root node
        :return:
        '''

        assert(self.node_index == 0) # this method should only be used for setting initial root

        self.nodes.insert(0,node)

    def add_node(self,node):

        '''
        Adds a node to the current tree
        :param player: string
        :param action: string
        :param amount: float
        :return: None
        '''

        self.node_index += 1

        node.node_index = self.node_index

        self.nodes.append(node) # add the new node to the list of nodes on the tree

        node.parent.children.append(node) # add the node to parents list of children

    def get_root(self):

        return self.nodes[0]

    def get_node(self,index):

        return self.nodes[index]

    def get_nodes(self):

        return self.nodes

    def get_num_nodes(self):

        return len(self.nodes)





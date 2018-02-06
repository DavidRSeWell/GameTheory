'''
    Module containing the Graph class
'''

import graphviz as gv


class TreeGraph(object):

    '''
        Class used for providing a graphical representation of graphs
    '''
    def __init__(self,tree=None,graph=None):

        self.tree = tree

        self.graph = graph


    def graph_from_tree(self,tree):

        '''
                Takes the current tree representation of a tree and
                converts it to a graphiz graph

                :return:
        '''

        # first add root node

        if tree.node_index == 0:

            label = tree.player + ' \\n ' + 'Pot: ' + str(tree.SB_cip + tree.BB_cip)

            self.graph.node(str(tree.node_index),label)

            for child in tree.children:

                label = tree.player + ' \\n ' + 'Pot: ' + str(tree.SB_cip + tree.BB_cip)

                self.graph.node(str(child.node_index),label)

                self.graph.edge(str(child.parent.node_index),str(child.node_index), str(child.action))

                self.graph_from_tree(child)

            #self.graph.edge(str(tree.parent.node_index),str(tree.node_index),str(tree.action))


        else:

            if (len(tree.children) == 0):

                label = tree.player + ' \\n ' + 'Pot: ' + str(tree.SB_cip + tree.BB_cip)

                self.graph.node(str(tree.node_index), label)

                self.graph.edge(str(tree.parent.node_index), str(tree.node_index), str(tree.action))

            else:

                for child in tree.children:

                    label = tree.player + ' \\n ' + 'Pot: ' + str(tree.SB_cip + tree.BB_cip)

                    self.graph.node(str(child.node_index),label)

                    self.graph.edge(str(child.parent.node_index),str(child.node_index), str(child.action))

                    self.graph_from_tree(child)

    def create_graph_from_tree(self):

        '''

            Takes the current tree representation of a tree and
            converts it to a graphiz graph

        :return:
        '''

        tree_nodes = self.tree.get_nodes()

        for node in tree_nodes:

            label = node.player + ' \\n ' + 'Pot: ' + str(node.SB_cip + node.BB_cip)

            self.graph.node(str(node.node_index),label)

            for child in node.children:

                label = child.player + ' \\n ' + 'Pot: ' + str(node.SB_cip + node.BB_cip)

                self.graph.node(str(child.node_index),label)

                self.graph.edge(str(node.node_index),str(child.node_index), str(list(child.action.keys())[0]))

    def create_graph_from_info_tree(self,info_tree):

        '''

            Takes the current tree representation of a tree and
            converts it to a graphiz graph

            :return:
        '''

        info_tree_nodes = info_tree.get_nodes()

        for node in info_tree_nodes:

            label = node.player + ' \\n ' + 'Pot: ' + str(node.SB_cip + node.BB_cip)

            if node.player != 'chance':

                label += 'Hand: ' + node.player_hand

            self.graph.node(str(node.node_index), label)

            for child in node.children:
                label = child.player + ' \\n ' + 'Pot: ' + str(node.SB_cip + node.BB_cip)

                self.graph.node(str(child.node_index), label)

                self.graph.edge(str(node.node_index), str(child.node_index), str(list(child.action.keys())[0]))

    def add_nodes(self,graph, nodes):

        for n in nodes:

            if isinstance(n, tuple):
                graph.node(n[0], **n[1])
            else:
                graph.node(n)

        return graph

    def add_edges(self,graph, edges):

        for e in edges:
            if isinstance(e[0], tuple):
                graph.edge(*e[0], **e[1])
            else:
                graph.edge(*e)

        return graph

from Poker.XFP import StrategyProfile,FP
from Poker.mcts_akq import AKQGameState
from Poker.mcts_behavior import AKQMixedMcts,MCTSStrategyProfile
from Graph.graph_tree import TreeGraph
import graphviz as gv
import pandas as pd
import Util.tree as Tree
import Poker.game as Game

###############################
## Test StrategyProfile: PASS #
###############################
run_test_strategy_prof = 0
if run_test_strategy_prof:

    tree = Tree.Tree()

    players = ["SB", "BB"]

    init_SB_cip = 0.0
    init_BB_cip = 0.0

    akq_game = Game.GameState(tree=tree, players=players, name='akq_game')

    akq_game.set_root(players[0], init_SB_cip, init_BB_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0, player="SB", action={"bet": 1})
    akq_game.new_action(current_index=0, player="SB", action={"check": 0})

    akq_game.new_action(current_index=1, player="BB", action={"call": 1})
    akq_game.new_action(current_index=1, player="BB", action={"fold": 0})

    akq_game.new_action(current_index=2, player="BB", action={"bet": 1})
    akq_game.new_action(current_index=2, player="BB", action={"check": 0})

    akq_game.new_action(current_index=5, player="SB", action={"call": 1})
    akq_game.new_action(current_index=5, player="SB", action={"fold": 0})

    #strategy_profile = StrategyProfile(akq_game.tree)


    print("Done testing Strat profile")

#####################################
## Run AKQ XFP Algo For simple tree #
#####################################
run_akq_xfp_simple = 0
if run_akq_xfp_simple:

    tree = Tree.Tree()

    players = ["SB", "BB"]

    init_SB_cip = 0.0
    init_BB_cip = 0.0

    akq_game = Game.GameState(tree=tree, players=players, name='akq_game')

    akq_game.set_root(players[0], init_SB_cip, init_BB_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0, player="SB", action={"check": 0})

    akq_game.new_action(current_index=1, player="BB", action={"bet": 1})
    akq_game.new_action(current_index=1, player="BB", action={"check": 0})

    akq_game.new_action(current_index=2, player="SB", action={"call": 1})
    akq_game.new_action(current_index=2, player="SB", action={"fold": 0})

    solution_strategies = FP(akq_game.tree,n_iter=1000)

    print("Done running akq xfp")

#####################################
## Run AKQ XFP Algo Longer version  #
#####################################
run_akq_xfp = 0
if run_akq_xfp:

    tree = Tree.Tree()

    players = ["SB", "BB"]

    init_SB_cip = 0.0
    init_BB_cip = 0.0

    akq_game = Game.GameState(tree=tree, players=players, name='akq_game')

    akq_game.set_root(players[0], init_SB_cip, init_BB_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0, player="SB", action={"bet": 1})
    akq_game.new_action(current_index=0, player="SB", action={"check": 0})

    akq_game.new_action(current_index=1, player="BB", action={"call": 1})
    akq_game.new_action(current_index=1, player="BB", action={"fold": 0})

    akq_game.new_action(current_index=2, player="BB", action={"bet": 1})
    akq_game.new_action(current_index=2, player="BB", action={"check": 0})

    akq_game.new_action(current_index=5, player="SB", action={"call": 1})
    akq_game.new_action(current_index=5, player="SB", action={"fold": 0})

    solution_strategies = FP(akq_game.tree,n_iter=100000)

    print("Done running akq xfp")

########################################
## Run MCTS akq tree. Version: regular #
########################################
run_mcts_akq_regular = 1
if run_mcts_akq_regular:

    tree = Tree.Tree()

    players = ["SB", "BB"]

    init_SB_cip = 0.0
    init_BB_cip = 0.0

    akq_game = Game.GameState(tree=tree, players=players, name='akq_game')

    akq_game.set_root(players[0], init_SB_cip, init_BB_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0, player="SB", action={"bet": 1})
    akq_game.new_action(current_index=0, player="SB", action={"check": 0})

    akq_game.new_action(current_index=1, player="BB", action={"call": 1})
    akq_game.new_action(current_index=1, player="BB", action={"fold": 0})

    akq_game.new_action(current_index=2, player="BB", action={"bet": 1})
    akq_game.new_action(current_index=2, player="BB", action={"check": 0})

    akq_game.new_action(current_index=5, player="SB", action={"call": 1})
    akq_game.new_action(current_index=5, player="SB", action={"fold": 0})

    GameState = AKQGameState(tree)

    new_graph = gv.Digraph(format="png")

    AKQGraph = TreeGraph(tree=akq_game.tree, graph=new_graph)

    AKQGraph.create_graph_from_tree()

    AKQGraph.graph.render('/Users/befeltingu/GameTheory/Results/Poker/MCTS/img/akq_extended')

    p1_policy, p2_policy = GameState.run(1000)

    p1_ev_matrix = []

    for node in GameState.player1.info_tree.nodes:

        if node.player == "chance":
            continue

        current_hand = node.player_hand

        policy = p1_policy[node.node_index]

        for action in policy.keys():
            ev = policy[action]['ev']

            p1_ev_matrix.append(
                ['player 1', 'node: ' + str(node.node_index), 'hand:' + str(current_hand), action, 'value: ' + str(ev)])

    ev_df_1 = pd.DataFrame(p1_ev_matrix)

    p2_ev_matrix = []

    for node in GameState.player2.info_tree.nodes:

        if node.player != "p2":
            continue

        current_hand = node.player_hand

        policy = p2_policy[node.node_index]

        for action in policy.keys():
            ev = policy[action]['ev']

            p2_ev_matrix.append(
                ['player 2', 'node: ' + str(node.node_index), 'hand:' + str(current_hand), action, 'value: ' + str(ev)])

    ev_df_2 = pd.DataFrame(p2_ev_matrix)

    ev_df_1.to_csv('/Users/befeltingu/GameTheory/Results/Poker/MCTS/data/ev_1.csv')

    ev_df_2.to_csv('/Users/befeltingu/GameTheory/Results/Poker/MCTS/data/ev_2.csv')

########################################
## Run MCTS mixed. Version: small #
########################################
run_mcts_akq_regular = 0
if run_mcts_akq_regular:

    tree = Tree.Tree()

    players = ["SB", "BB"]

    init_SB_cip = 0.0
    init_BB_cip = 0.0

    akq_game = Game.GameState(tree=tree, players=players, name='akq_game')

    akq_game.set_root(players[0], init_SB_cip, init_BB_cip)

    root = akq_game.tree.get_root()

    akq_game.new_action(current_index=0, player="SB", action={"check": 0})

    akq_game.new_action(current_index=1, player="BB", action={"bet": 1})
    akq_game.new_action(current_index=1, player="BB", action={"check": 0})

    akq_game.new_action(current_index=2, player="SB", action={"call": 1})
    akq_game.new_action(current_index=2, player="SB", action={"fold": 0})

    strategy_profile = MCTSStrategyProfile(tree)

    GameState = AKQMixedMcts(tree,strategy_profile)

    new_graph = gv.Digraph(format="png")

    AKQGraph = TreeGraph(tree=akq_game.tree, graph=new_graph)

    AKQGraph.create_graph_from_tree()

    AKQGraph.graph.render('/Users/befeltingu/GameTheory/Results/Poker/MCTS/img/akq_extended')

    strategy_profile = GameState.run(1000)
    p1_ev_matrix = []

    for node in GameState.player1.info_tree.nodes:

        if node.player == "chance":
            continue

        current_hand = node.player_hand

        policy = p1_policy[node.node_index]

        for action in policy.keys():
            ev = policy[action]['ev']

            p1_ev_matrix.append(
                ['player 1', 'node: ' + str(node.node_index), 'hand:' + str(current_hand), action, 'value: ' + str(ev)])

    ev_df_1 = pd.DataFrame(p1_ev_matrix)

    p2_ev_matrix = []

    for node in GameState.player2.info_tree.nodes:

        if node.player != "p2":
            continue

        current_hand = node.player_hand

        policy = p2_policy[node.node_index]

        for action in policy.keys():
            ev = policy[action]['ev']

            p2_ev_matrix.append(
                ['player 2', 'node: ' + str(node.node_index), 'hand:' + str(current_hand), action, 'value: ' + str(ev)])

    ev_df_2 = pd.DataFrame(p2_ev_matrix)

    ev_df_1.to_csv('/Users/befeltingu/GameTheory/Results/Poker/MCTS/data/ev_1.csv')

    ev_df_2.to_csv('/Users/befeltingu/GameTheory/Results/Poker/MCTS/data/ev_2.csv')


import Util.tree as Tree
import Poker.game as Game
from Poker.XFP import StrategyProfile





##############################
## Test StrategyProfile: PASS #
##############################
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




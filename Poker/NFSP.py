'''
This will be an implementation of the Neural fictitious self-play with fitted Q learning
as implemented by Silver ,Heinrich 2017
'''








class NFSP:

    def __init__(self,tree):

        self.tree = tree

        self.rl_replay = None

        self.sl_replay = None

        self.q_parameters = None

        self.q_target_parameters = None

        self.policy_parameters = None # For the policy network

        self.anticipatory_paramter = None


    def get_policy(self):
        '''
        selects either an epsilon greedy policy with prob 1 - ant
        select current policy using the policy network with prob ( ant )
        :return:
        '''
        return

    def get_reward(self,s):

        '''

                Takes in a leaf node and returns the reward to each player
                :param s:
                :return:

                '''

        r = {"SB": 0, "BB": 0}

        villian, hero = self.get_hero_villian(s)

        villian_cip, hero_cip = self.get_hero_villian_cip(s)

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


        elif action_type == "call":  # same as check?

            # evaluate winner
            if (hero.current_hand < villian.current_hand):
                # SB wins
                r[hero.name] = current_pot + (hero.starting_stack - hero_cip)

                r[villian.name] = villian.starting_stack - villian_cip

            else:

                r[villian.name] = current_pot + (villian.starting_stack - villian_cip)

                r[hero.name] = hero.starting_stack - hero_cip

        return r

    def execute_action(self,action,state):
        '''
        Take in the current state and return the reward and the next state
        :param action:
        :param state:
        :return:
        '''

    def simulate(self,s,policy):

        '''
        Used like in mcts to allow an episode of the game to be played out
        :param s:
        :return:
        '''


        if s.is_leaf == True:

            reward = self.get_reward(s)

            return reward

        current_player = self.player1 if s.player == "SB" else self.player2

        action = self.select_uct(s, current_player.current_hand)

        next_state = self.get_new_state(s, action)[0]

        r = self.simulate(next_state, i)

        self.update(s, r, i)

        return r

    def run(self,iterations):

        policy = self.get_policy()





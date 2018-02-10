
def get_max_ev(range1 ,range2):

    max_ev = -1

    max_action = None

    for action_hero in range1.keys():

        if action_hero == "rock":

            curr_max_ev = range2["scissors"]

            if curr_max_ev > max_ev:

                max_ev = curr_max_ev

                max_action = "rock"

        elif action_hero == "paper":

            curr_max_ev = range2["rock"]

            if curr_max_ev > max_ev:

                max_ev = curr_max_ev

                max_action = "paper"

        else:  # scissors
            curr_max_ev = range2["paper"]

            if curr_max_ev > max_ev:

                max_ev = curr_max_ev

                max_action = "scissors"

    return max_action

def update_range(range_hero, max_play, n):

    fraction = 1 - 1 / (n/100 + 2)

    for action in range_hero.keys():

        if action == max_play:

            range_hero[action] = range_hero[max_play] * (fraction) + (1 - fraction)

        else:
            range_hero[action] = range_hero[action] * (fraction)

    return range_hero

def FictitiousPlay(n_iterations):

    p1 = {"rock": 1.0, "paper": 0.0, "scissors": 0.0}

    p2 = {"rock": 0.0, "paper": 0.0, "scissors": 1.0}

    for n in range(n_iterations):

        player_1_max = get_max_ev(p1, p2)

        p1 = update_range(p1,player_1_max,n)

        player_2_max = get_max_ev(p2, p1)

        p2 = update_range(p2,player_2_max,n)

    return p1, p2




optimal_strategy = FictitiousPlay(10000)

print("Optimal strategy")

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

        return range["Q"]/(range["Q"] + range["A"])
        #return 0.5

    else:
        # the hand must be a Q and the Q never has any equity
        return 0

def get_num_hands(hero_hand,range):

    '''
    Calculates the number of hands in a range based given
    the hand currently held by the hero

    :param hand:
    :param range:
    :return:
    '''

    num_hands = 0

    for hand in range:
        if hand != hero_hand:
            num_hands += range[hand]

    return num_hands

def get_hand_number(hand):

    if hand == "A":
        return 0

    elif hand == "K":
        return 1

    else:
        return 2

def get_hand_string(hand):

    if hand == 0:
        return "A"

    elif hand == 1:
        return "K"

    else:
        return "Q"
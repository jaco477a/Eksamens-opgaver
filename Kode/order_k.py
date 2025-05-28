import numpy as np

#Genereret af ChatGPT

def order_k_rationality(U_row, U_col, k):
    """
    Compute level-k strategy for the row player given a game (U_row vs U_col).
    U_row: row player's payoff matrix (na1 x na2)
    U_col: column player's payoff matrix (na1 x na2)
    level: level of reasoning (0 = random, 1 = best response to 0, etc.)
    Returns: probability distribution over row player's actions
    """
    na1, na2 = U_row.shape

    if k == 0:
        return np.ones(na1) / na1  # uniform random strategy
    
    # Get (k-1)'s strategy of the opponent (column player)
    col_level_k_minus_1 = order_k_rationality(U_col.T, U_row.T, k - 1)
    
    # Expected utility for each row action
    expected_utilities = U_row @ col_level_k_minus_1

    # Best responses: argmax expected utility
    best_actions = np.argwhere(expected_utilities == np.max(expected_utilities)).flatten()

    strategy = np.zeros(na1)
    strategy[best_actions] = 1 / len(best_actions)
    
    return strategy

def level_k_profile(U1, U2, k1, k2):
    """
    Compute the level-k strategy profile for both players.
    """
    s1 = order_k_rationality(U1, U2, k1)
    s2 = order_k_rationality(U2.T, U1.T, k2)
    return s1, s2
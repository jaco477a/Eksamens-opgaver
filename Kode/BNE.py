import numpy as np
import nashpy as nash
import pandas as pd

def compute_full_matrix(U1, U2, p, action_1, action_2):
    # U1 = [u11, u12]
    # U2 = [u21, u22]
    # each uij is an nA1 * nA2-matrix of payoffs
    # p (scalar): probability of type=0

    nA1, nA2 = U1[0].shape

    t1 = np.empty((nA1, nA2 * nA2))
    t2 = np.empty((nA1, nA2 * nA2))

    # player 1 chooses an action without knowing what type 2 is
    for ia1 in range(nA1):  # choice if type 0
        i_col = 0

        # player 2 chooses an action conditional on observing her type
        for a2_1 in range(nA2):  # choice if type 1
            for a2_2 in range(nA2):
                t1[ia1, i_col] = p * U1[0][ia1, a2_1] + (1.0 - p) * U1[1][ia1, a2_2]
                t2[ia1, i_col] = p * U2[0][ia1, a2_1] + (1.0 - p) * U2[1][ia1, a2_2]
                i_col += 1

    col_labels = [
        f"{kv}{kh}"
        for kv in action_1
        for kh in action_2
    ]
    df_U1 = pd.DataFrame(t1, index=action_1, columns=col_labels)
    df_U2 = pd.DataFrame(t2, index=action_2, columns=col_labels)

    return [t1, df_U1], [t2, df_U2]

def compute_bne(U1, U2):
    game = nash.Game(U1, U2)
    BNE = list(game.support_enumeration())
    return BNE



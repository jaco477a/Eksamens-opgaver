#Genereret af ChatGPT

import numpy as np 

def BR_1(U1):
    na1, na2 = U1.shape

    argmax_indices = np.argmax(U1, axis=0)
    
    max_values = U1[argmax_indices, np.arange(na2)]
    
    br = [
        np.where(U1[:, j] == max_values[j])[0].tolist()
        for j in range(na2)
    ]
    
    return br

def nash_eq_brute_force(U1, U2): 
    '''
    Output:
        NE: list of 2-tuples. There is one tuple per equilibrium, so len(NE) is the number of equilibria found. 
    '''
    br1 = BR_1(U1)
    br2 = BR_1(U2.T)
    
    NE = [] 
    for i in range(len(br1)):
        for j in range(len(br2)):
            if i in br2[j] and j in br1[i]:
                NE.append((i,j))
    return NE
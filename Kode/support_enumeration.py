import nashpy as nash

#Hjemmelavet kode

def msne(U1, U2):
    game = nash.Game(U1, U2)
    msne = list(game.support_enumeration())
    
    return msne
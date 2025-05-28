from scipy.optimize import linprog
import numpy as np

#Kode taget fra forelæsning 4 slide 17/23

def solve_by_LP(U):
    num_vars, num_cols = U.shape
    assert num_vars == num_cols  # U must be square

    oo = np.zeros((1, num_vars))
    ii = np.ones((1, num_vars))

    # objective: c = [-1, 0, 0, ..., 0]
    c = np.insert(oo, 0, -1.0)

    # inequality constraints: A*x <= b
    top = np.hstack((ii.T, -1 * U.T))
    bot = np.hstack((oo.T, -1 * np.eye(num_vars)))
    A_ub = np.vstack((top, bot))

    b_ub = np.zeros((1, 2 * num_vars))
    b_ub = np.matrix(b_ub)

    # equality constraints: A*x = b
    A_eq = np.matrix(np.hstack((0, np.ones((num_vars,)))))
    b_eq = 1.0  # just one condition so scalar

    sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    return sol.x[1:]

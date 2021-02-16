import cdd
import numpy as np


def solve_zero_sum_game(A, method="cdd"):
    """
    method in {"cdd"}
    - cdd is an *exact* arithmetic solver
    """

    if method == "cdd":
        ret = solve_with_cdd(A)
    else:
        raise("method ", method, " not recognised")

    return ret


def solve_with_cdd(A, verbose=False): 
    return solve_with_cdd_for_II(-A.T, verbose=verbose)


def solve_with_cdd_for_II(A, verbose=False): 
    """This method finds II's minmax strategy for zero-sum game A"""
    m = A.shape[0] # number of rows
    n = A.shape[1] # number of columns

    A = np.column_stack([[0]*m,-A,[1]*m])

    I = np.eye(n)
    nn = np.column_stack([[0]*n,I,[0]*n])

    # non-negativity constraints
    n1 = [-1] * n
    n1.insert(0,1)
    n1.append(0) # n1 = 1,-1,-1,...,-1,0]
    n2 = [1] * n
    n2.insert(0,-1)
    n2.append(0) # n1 = 1,-1,-1,...,-1,0]

    d = np.vstack([A,nn,n1,n2])

    mat = cdd.Matrix(d.tolist(), number_type='fraction')
    mat.obj_type = cdd.LPObjType.MIN

    d = [0] * (n+1)
    d.append(1) # [0,0,...0,1]
    mat.obj_func = d

    lp = cdd.LinProg(mat)
    lp.solve()
    lp.status == cdd.LPStatusType.OPTIMAL

    # lp.primal_solution uses fractions, and has value as last entry, so that
    # is dropped
    p = [float(val) for val in lp.primal_solution[:-1]]
    u = float(lp.obj_value)

    if verbose:
        print("------ Solved with cdd -------------")
        print("Optimal strategy:", p)
        print("Optimal payoff:", -u)
        print("------------------------------------")

    return p, -u


if __name__ == "__main__":
    # A = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
    # A = np.array([[2, 0],[1, 3], [1.5, 2]])
    A = np.array([
        [2, 0, 7, 1],
        [1, 3, 6, 3],
        [1.5, 2, 3, 1]
    ])

    print(solve_zero_sum_game(A, method="cdd"))

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid

def get_compound_coeff_func(phi=1.0, max_cost=2.0):
    def compund_coeff(x):
        alpha = x[0]
        beta = x[1]
        gamma = x[2]

        # Scale by power
        depth = alpha ** phi
        width = beta ** phi
        resolution = gamma ** phi

        # Compute the cost function
        cost = depth * (width ** 2) * (resolution ** 2)
        return (cost - max_cost) ** 2

    return compund_coeff

def optimize_coefficients(num_coeff=3, cost_func=None, phi=1.0, max_cost=2.0,
                        search_per_coeff=4, save_coeff=True, tol=None):
    """
    Compute the possible values of any number of coefficients given a cost
    function, phi and max cost
    Take into account the search space per coefficient so that the grid search
    does not become prohibitively large

    # Arguments:
        num_coeff: number of coefficients that must be optimized
        cost_func: coefficient cost function that minimized. The function
            can be user defined, in which case its params are numpy vector of 
            length `num_coeff` defined above. It is suggested to use MSE
        phi: The base power of the parameters. Default to 1
        max_cost: The maximum cost of permissable. Default to 2
        search_per_coeff: # of values tried per coefficient, so  the search
            space of size `search_per_coeff` ^ `num_coeff`
        save_coeff: bool, whether to save the resulting coefficients into the
            file `param_coeff.npy` in the current working dir
        tol: float, tolerance  of error in the cost function. Used to select
            candidates which have a cost less than the tolerance

    # Returns:
        A numpy array of shape [search_per_coeff ^ num_coeff, num_coeff]
    """
    phi = float(phi)
    max_cost = float(max_cost)
    search_per_coeff = float(search_per_coeff)

    if cost_func is None:
        cost_func = get_compound_coeff_func(phi=phi, max_cost=max_cost)

    #  perpare inequality constraints
    ineq_contraints = {
        'type': 'ineq',
        'fun': lambda x: x - 1
    }

    # prepare a matrix to store results
    param_range = (search_per_coeff ** num_coeff, num_coeff)
    param_set = np.zeros(param_range)

    grid = {i: np.linspace(1.0, max_cost, num=search_per_coeff)
            for i in range(num_coeff)}
    param_grid  = ParameterGrid(grid)
    for idx, param in enumerate(param_grid):
        # Create a vector for the cost function and minimize using SLSQP
        x0 = np.array([param[i] for i in range(num_coeff)])
        res = minimize(cost_func, x0, method='SLSQP', constraints=ineq_contraints)
        param_set[idx] = res.x

    # Compute a minimum tolerance of the cost function to select in candidate list
    if tol is not None:
        tol = float(tol)
        cost_scores = np.array([cost_func(param) for param in param_set])
        param_set = param_set[np.where(cost_scores <= tol)]

    if save_coeff:
        np.save('param_coeff.npy', param_set)
    return param_set
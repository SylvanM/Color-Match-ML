from sklearn.neural_network import MLPRegressor

"""
Here are the basic setup stuff for the network,
"""

hidden_layer_sizes  = (4,)
activation          = 'tanh'
solver              = 'lbfgs'
alpha               = 0.0001
max_iter            = 200
random_state        = None
tol                 = 1e-4
verbose             = True
max_fun             = 15000

regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
    alpha=0.0001, max_iter=max_iter, random_state=random_state, tol=tol, verbose=verbose, max_fun=max_fun)


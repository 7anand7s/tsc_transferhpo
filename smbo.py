import numpy as np
from scipy.stats import norm


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.05):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(Y_sample)
    imp = mu - mu_sample_opt - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei.flatten()


def propose_location(acquisition, X_sample, Y_sample, gpr, search_space):
    '''
    Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        search_space: The whole search space where acquisition in run

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]

    # Find the best optimum by starting from n_restart different random points.
    aq_func = acquisition(search_space.reshape(-1, dim), X_sample, Y_sample, gpr)

    max_i = np.argmax(aq_func)
    next_loc = search_space[max_i]

    return next_loc.reshape(-1, 1), max_i

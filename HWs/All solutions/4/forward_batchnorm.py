def forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode):
    """
    Performs the forward propagation through a BatchNorm layer.

    Arguments:
    Z -- input, with shape (num_examples, num_features)
    gamma -- vector, BN layer parameter
    beta -- vector, BN layer parameter
    eps -- scalar, BN layer hyperparameter
    beta_avg -- scalar, beta value to use for moving averages
    mode -- boolean, indicating whether used at 'train' or 'test' time

    Returns:
    out -- output, with shape (num_examples, num_features)
    """

    if mode == 'train':
        # TODO: Mean of Z across first dimension
        mu = np.mean(X, axis=0)

        # TODO: Variance of Z across first dimension
        var = np.var(X, axis=0)

        # Take moving average for cache_dict['mu']
        cache_dict['mu'] = beta_avg * cache_dict['mu'] + (1-beta_avg) * mu

        # Take moving average for cache_dict['var']
        cache_dict['var'] = beta_avg * cache_dict['var'] + (1-beta_avg) * var

        X = (X - mu) / np.sqrt(var + eps)
        out = gamma * X + beta

    elif mode == 'test':
        # TODO: Load moving average of mu
        mu = cache_dict['mu']

        # TODO: Load moving average of var
        var = cache_dict['var']

    # TODO: Apply z_norm transformation
    Z_norm = (X - mu) / np.sqrt(var + eps)

    # TODO: Apply gamma and beta transformation to get Z tiled
    out = gamma * X + beta

    return out

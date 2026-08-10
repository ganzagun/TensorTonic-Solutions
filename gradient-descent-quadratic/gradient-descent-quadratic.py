def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """

    for ep in range(steps):
        x0 = x0 - lr*(2*a*x0 + b)

    return x0
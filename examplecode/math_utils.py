"""
Mathematical utility functions and simple numeric algorithms.
"""


def factorial(n):
    """Compute n! iteratively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def gcd(a, b):
    """Compute greatest common divisor using Euclid's algorithm."""
    while b:
        a, b = b, a % b
    return a


def is_prime(n):
    """Check if n is a prime number."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def moving_average(values, window):
    """Calculate the moving average over a sliding window."""
    avgs = []
    for i in range(len(values) - window + 1):
        chunk = values[i:i + window]
        avgs.append(sum(chunk) / window)
    return avgs

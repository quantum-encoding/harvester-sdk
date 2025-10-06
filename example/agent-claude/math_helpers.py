"""
Mathematical helper functions for common computations.
"""


def factorial(n):
    """
    Calculate the factorial of a non-negative integer.

    Args:
        n (int): A non-negative integer

    Returns:
        int: The factorial of n (n!)

    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer

    Examples:
        >>> factorial(0)
        1
        >>> factorial(5)
        120
        >>> factorial(10)
        3628800
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")

    if n == 0 or n == 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def fibonacci(n):
    """
    Calculate the nth Fibonacci number.

    The Fibonacci sequence starts with 0, 1, and each subsequent number
    is the sum of the previous two numbers.

    Args:
        n (int): The position in the Fibonacci sequence (0-indexed)

    Returns:
        int: The nth Fibonacci number

    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
        >>> fibonacci(15)
        610
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")

    if n == 0:
        return 0
    if n == 1:
        return 1

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


if __name__ == "__main__":
    # Test factorial function
    print("Testing factorial function:")
    print(f"factorial(0) = {factorial(0)}")
    print(f"factorial(1) = {factorial(1)}")
    print(f"factorial(5) = {factorial(5)}")
    print(f"factorial(10) = {factorial(10)}")

    print("\nTesting fibonacci function:")
    print(f"fibonacci(0) = {fibonacci(0)}")
    print(f"fibonacci(1) = {fibonacci(1)}")
    print(f"fibonacci(10) = {fibonacci(10)}")
    print(f"fibonacci(15) = {fibonacci(15)}")
    print(f"fibonacci(20) = {fibonacci(20)}")

    # Test error handling
    print("\nTesting error handling:")
    try:
        factorial(-1)
    except ValueError as e:
        print(f"factorial(-1) raised ValueError: {e}")

    try:
        fibonacci(-5)
    except ValueError as e:
        print(f"fibonacci(-5) raised ValueError: {e}")

    print("\nAll tests completed!")

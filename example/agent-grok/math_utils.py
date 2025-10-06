def factorial(n):
    """Calculate the factorial of a non-negative integer n.

    Args:
        n (int): A non-negative integer.

    Returns:
        int: The factorial of n.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def fibonacci(n):
    """Calculate the nth Fibonacci number.

    Args:
        n (int): A non-negative integer.

    Returns:
        int: The nth Fibonacci number.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == "__main__":
    print("Testing factorial:")
    print(f"factorial(0) = {factorial(0)}")
    print(f"factorial(1) = {factorial(1)}")
    print(f"factorial(5) = {factorial(5)}")
    try:
        factorial(-1)
    except ValueError as e:
        print(f"factorial(-1) raised: {e}")

    print("\nTesting fibonacci:")
    print(f"fibonacci(0) = {fibonacci(0)}")
    print(f"fibonacci(1) = {fibonacci(1)}")
    print(f"fibonacci(5) = {fibonacci(5)}")
    print(f"fibonacci(10) = {fibonacci(10)}")
    try:
        fibonacci(-1)
    except ValueError as e:
        print(f"fibonacci(-1) raised: {e}")
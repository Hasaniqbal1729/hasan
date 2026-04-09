def fibonacci(n):
    """Return a list of the first n Fibonacci numbers."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    series = [0, 1]
    while len(series) < n:
        series.append(series[-1] + series[-2])
    return series


def fibonacci_recursive(n):
    """Return the nth Fibonacci number (0-indexed) using recursion."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_generator(limit):
    """Generate Fibonacci numbers up to a given limit."""
    a, b = 0, 1
    while a <= limit:
        yield a
        a, b = b, a + b


if __name__ == "__main__":
    # Print first 10 Fibonacci numbers
    n = 10
    print(f"First {n} Fibonacci numbers:")
    print(fibonacci(n))

    # Using generator up to 100
    print("\nFibonacci numbers up to 100:")
    print(list(fibonacci_generator(100)))

    # Single value via recursion
    print("\n10th Fibonacci number (recursive):", fibonacci_recursive(10))

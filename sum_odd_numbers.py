def sum_first_odd_numbers(n):
    """Return the sum of the first n odd natural numbers."""
    return sum(2 * i - 1 for i in range(1, n + 1))


if __name__ == "__main__":
    n = int(input("Enter the count of first odd numbers to sum: "))
    result = sum_first_odd_numbers(n)
    print(f"Sum of first {n} odd natural numbers = {result}")

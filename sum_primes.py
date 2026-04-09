def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def sum_first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return sum(primes), primes


if __name__ == "__main__":
    total, primes = sum_first_n_primes(10)
    print(f"First 10 prime numbers: {primes}")
    print(f"Sum = {total}")

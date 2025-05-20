"""
Prime Counting Problem (π function)
Input: A positive integer n
Output: The number of prime numbers less than or equal to n

This is a classic algorithmic problem with many optimization opportunities.
Key insights to consider:
- All primes except 2 are odd numbers
- A number's smallest prime factor is always ≤ its square root
- Consider using a boolean array to mark numbers efficiently
- Multiples of small primes can be marked as composite
- You can start marking from the square of each prime
- Once a number is marked as composite, all its multiples are also composite
"""

def prime_count(n: int) -> int:
    """
    Count the number of prime numbers less than or equal to n.
    
    Args:
        n: A positive integer
        
    Returns:
        The number of prime numbers ≤ n
        
    Examples:
        >>> prime_count(10)
        4  # primes are 2, 3, 5, 7
        >>> prime_count(20)
        8  # primes are 2, 3, 5, 7, 11, 13, 17, 19
    """
    # <evolve>
    cnt = 0
    for x in range(2, n + 1):
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                break
        else:
            cnt += 1
    return cnt
    # </evolve>

def helper_function():
    """This is a fixed helper function that won't be evolved."""
    return "I'm a helper!" 
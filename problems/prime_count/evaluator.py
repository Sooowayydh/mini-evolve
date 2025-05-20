"""
Evaluator for the prime counting problem.
Tests correctness and measures performance across different input sizes.
"""

import time
import random
from typing import Dict, Any

# Pre-computed prime counts for verification
# Source: https://oeis.org/A000720
PRIME_COUNTS = {
    10: 4,      # primes: 2, 3, 5, 7
    20: 8,      # primes: 2, 3, 5, 7, 11, 13, 17, 19
    50: 15,     # primes up to 50
    100: 25,    # primes up to 100
    1000: 168,  # primes up to 1000
    10000: 1229 # primes up to 10000
}

def evaluate(candidate_code: str, num_tests: int = 5) -> Dict[str, Any]:
    """
    Evaluate the candidate solution for correctness and performance.
    
    Args:
        candidate_code: The code to evaluate
        num_tests: Number of test cases to run
        
    Returns:
        Dictionary containing pass_rate and speed metrics
    """
    # Create a namespace for the candidate code
    namespace = {}
    
    try:
        # Execute the candidate code
        exec(candidate_code, namespace)
        
        # Get the function from the namespace
        if 'prime_count' not in namespace:
            return {'pass_rate': 0.0, 'speed': float('inf')}
        
        prime_count = namespace['prime_count']
        
        # Test cases with different sizes
        test_cases = [
            # Small numbers (fast verification)
            (10, 4),
            (20, 8),
            (50, 15),
            # Medium numbers (balance of speed and correctness)
            (100, 25),
            (1000, 168),
            # Large numbers (performance testing)
            (10000, 1229),
            # Random numbers for robustness
            (random.randint(100, 1000), None),
            (random.randint(1000, 10000), None)
        ]
        
        # Run tests
        total_time = 0
        passed = 0
        total = 0
        
        for n, expected in test_cases:
            # Skip if we've run enough tests
            if total >= num_tests:
                break
                
            # Measure execution time
            start_time = time.time()
            result = prime_count(n)
            end_time = time.time()
            
            # Add to total time
            total_time += (end_time - start_time)
            
            # Verify result
            if expected is not None:
                if result == expected:
                    passed += 1
            else:
                # For random numbers, verify against our own implementation
                if result == reference_prime_count(n):
                    passed += 1
            
            total += 1
        
        # Calculate metrics
        pass_rate = passed / total if total > 0 else 0.0
        avg_speed = total_time / total if total > 0 else float('inf')
        
        return {
            'pass_rate': pass_rate,
            'speed': avg_speed
        }
        
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return {'pass_rate': 0.0, 'speed': float('inf')}

def reference_prime_count(n: int) -> int:
    """
    Reference implementation for verification.
    Uses a simple sieve of Eratosthenes.
    """
    if n < 2:
        return 0
    
    # Initialize sieve
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return sum(is_prime) 
"""
Evaluator for the QuickSort Problem.
Tests correctness and performance of quicksort implementations.
"""

import random
import time
from typing import Dict, Any

def evaluate(candidate_code: str, num_tests: int = 100) -> Dict[str, Any]:
    """
    Evaluate a quicksort implementation.
    
    Args:
        candidate_code: The code to evaluate (should define a quicksort function)
        num_tests: Number of test cases to run
        
    Returns:
        Dictionary containing:
        - pass_rate: Fraction of tests passed (0.0 to 1.0)
        - speed: Average execution time in seconds
    """
    # Create a namespace for execution
    namespace = {}
    
    try:
        # Execute the candidate code
        exec(candidate_code, namespace)
        
        # Get the quicksort function
        quicksort = namespace.get('quicksort')
        if not quicksort:
            return {'pass_rate': 0.0, 'speed': float('inf')}
        
        # Run test cases
        passed = 0
        total_time = 0.0
        
        for _ in range(num_tests):
            # Generate random array of different sizes
            size = random.randint(10, 1000)
            arr = [random.randint(-1000, 1000) for _ in range(size)]
            
            # Get expected result
            expected = sorted(arr)
            
            try:
                # Time the candidate implementation
                start_time = time.time()
                result = quicksort(arr)
                end_time = time.time()
                
                # Check if result matches expected
                if result == expected:
                    passed += 1
                    total_time += (end_time - start_time)
                    
            except Exception:
                continue
        
        if passed == 0:
            return {'pass_rate': 0.0, 'speed': float('inf')}
        
        return {
            'pass_rate': passed / num_tests,
            'speed': total_time / passed
        }
        
    except Exception:
        return {'pass_rate': 0.0, 'speed': float('inf')} 
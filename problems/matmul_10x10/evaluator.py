"""
Evaluator for the Matrix Multiplication Problem.
Tests correctness and performance of matrix multiplication implementations.
"""

import random
import time
from typing import Dict, Any

def evaluate(candidate_code: str, num_tests: int = 100) -> Dict[str, Any]:
    """
    Evaluate a matrix multiplication implementation.
    
    Args:
        candidate_code: The code to evaluate (should define a matmul function)
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
        
        # Get the matmul function
        matmul = namespace.get('matmul')
        if not matmul:
            return {'pass_rate': 0.0, 'speed': float('inf')}
        
        # Run test cases
        passed = 0
        total_time = 0.0
        
        for _ in range(num_tests):
            # Generate random 10x10 matrices
            a = [[random.uniform(-10, 10) for _ in range(10)] for _ in range(10)]
            b = [[random.uniform(-10, 10) for _ in range(10)] for _ in range(10)]
            
            # Compute reference result
            ref_result = [[sum(a[i][k] * b[k][j] for k in range(10)) 
                          for j in range(10)] for i in range(10)]
            
            try:
                # Time the candidate implementation
                start_time = time.time()
                result = matmul(a, b)
                end_time = time.time()
                
                # Check dimensions
                if len(result) != 10 or any(len(row) != 10 for row in result):
                    continue
                
                # Check correctness (with some tolerance for floating point)
                is_correct = True
                for i in range(10):
                    for j in range(10):
                        if abs(result[i][j] - ref_result[i][j]) > 1e-10:
                            is_correct = False
                            break
                    if not is_correct:
                        break
                
                if is_correct:
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
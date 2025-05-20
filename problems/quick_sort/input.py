"""
QuickSort Problem
Input: A list of integers
Output: The sorted list in ascending order
"""

def quicksort(arr):
    # <evolve>
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
    # </evolve>

def helper_function():
    """This is a fixed helper function that won't be evolved."""
    return "I'm a helper!" 
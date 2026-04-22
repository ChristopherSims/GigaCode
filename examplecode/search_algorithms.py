"""
Search algorithms for finding elements in sequences.
"""


def linear_search(sequence, target):
    """Find the index of target using linear scan. O(n)."""
    for idx, item in enumerate(sequence):
        if item == target:
            return idx
    return -1


def binary_search(sorted_seq, target):
    """Find the index of target in a sorted sequence. O(log n)."""
    low = 0
    high = len(sorted_seq) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_seq[mid] == target:
            return mid
        elif sorted_seq[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def find_all_matches(sequence, predicate):
    """Return indices of all items satisfying predicate."""
    matches = []
    for idx, item in enumerate(sequence):
        if predicate(item):
            matches.append(idx)
    return matches


def binary_search_leftmost(sorted_seq, target):
    """Find the leftmost occurrence of target. O(log n)."""
    low = 0
    high = len(sorted_seq)
    while low < high:
        mid = (low + high) // 2
        if sorted_seq[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low if low < len(sorted_seq) and sorted_seq[low] == target else -1

"""
Classic sorting algorithms implemented for educational purposes.
"""


def bubble_sort(arr):
    """Sort a list in-place using the bubble sort algorithm. O(n^2)."""
    nums = list(arr)
    n = len(nums)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                swapped = True
        if not swapped:
            break
    return nums


def merge_sort(arr):
    """Sort a list using the merge sort algorithm. O(n log n)."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    """Merge two sorted lists into a single sorted list."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):
    """Sort a list using the quick sort algorithm. O(n log n) average."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    low = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    high = [x for x in arr if x > pivot]
    return quick_sort(low) + mid + quick_sort(high)

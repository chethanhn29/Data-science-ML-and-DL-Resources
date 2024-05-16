## different techniques and strategies along with their logic to tackle array-related problems:

### 1. Brute Force
**Approach**: Try all possible solutions to find the answer.
**Logic**: Often involves nested loops to check all possible combinations or permutations.
**Example**: Finding pairs in an array that sum to a specific value.

```python
def find_pairs(arr, target):
    pairs = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == target:
                pairs.append((arr[i], arr[j]))
    return pairs
```

### 2. Sorting
**Approach**: Sort the array and then apply a strategy that benefits from the sorted order.
**Logic**: Sorting helps to efficiently search, eliminate duplicates, or manage order-dependent operations.
**Example**: Two-pointer technique for finding pairs that sum to a specific value.

```python
def find_pairs(arr, target):
    arr.sort()
    left, right = 0, len(arr) - 1
    pairs = []
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            pairs.append((arr[left], arr[right]))
            left += 1
            right -= 1
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return pairs
```

### 3. Hashing
**Approach**: Use a hash table (dictionary) to store and lookup values.
**Logic**: Provides O(1) average time complexity for insertions and lookups.
**Example**: Finding if there exists a pair with a given sum.

```python
def find_pairs(arr, target):
    seen = {}
    pairs = []
    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen[num] = True
    return pairs
```

### 4. Sliding Window
**Approach**: Maintain a window that slides over the array to maintain a range of elements.
**Logic**: Useful for problems involving contiguous subarrays.
**Example**: Finding the maximum sum of a subarray of size `k`.

```python
def max_sum_subarray(arr, k):
    max_sum = current_sum = sum(arr[:k])
    for i in range(k, len(arr)):
        current_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, current_sum)
    return max_sum
```

### 5. Divide and Conquer
**Approach**: Divide the problem into smaller subproblems, solve them recursively, and combine their results.
**Logic**: Often used in sorting algorithms (merge sort, quicksort) and finding the maximum subarray sum (Kadane's algorithm).
**Example**: Merge Sort for sorting an array.

```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr
```

### 6. Dynamic Programming
**Approach**: Break down problems into subproblems, solve them once, and store their solutions.
**Logic**: Avoids recomputation by storing results of overlapping subproblems.
**Example**: Finding the maximum sum of a contiguous subarray (Kadane's algorithm).

```python
def max_subarray_sum(arr):
    max_so_far = arr[0]
    current_max = arr[0]

    for num in arr[1:]:
        current_max = max(num, current_max + num)
        max_so_far = max(max_so_far, current_max)

    return max_so_far
```

### 7. Greedy
**Approach**: Make the locally optimal choice at each step with the hope of finding a global optimum.
**Logic**: Simplifies problem-solving but does not always provide the optimal solution for all problems.
**Example**: Finding the minimum number of platforms required for trains (merge intervals).

```python
def min_platforms(arrivals, departures):
    arrivals.sort()
    departures.sort()
    platform_needed = 1
    max_platforms = 1
    i, j = 1, 0

    while i < len(arrivals) and j < len(departures):
        if arrivals[i] <= departures[j]:
            platform_needed += 1
            i += 1
        elif arrivals[i] > departures[j]:
            platform_needed -= 1
            j += 1
        max_platforms = max(max_platforms, platform_needed)

    return max_platforms
```

### 8. Backtracking
**Approach**: Explore all possible solutions and backtrack to find the correct one.
**Logic**: Used in combinatorial and permutation problems.
**Example**: Generating all permutations of an array.

```python
def permute(arr):
    result = []

    def backtrack(start):
        if start == len(arr):
            result.append(arr[:])
        for i in range(start, len(arr)):
            arr[start], arr[i] = arr[i], arr[start]
            backtrack(start + 1)
            arr[start], arr[i] = arr[i], arr[start]

    backtrack(0)
    return result
```

### 9. Binary Search
**Approach**: Efficiently search a sorted array by repeatedly dividing the search interval in half.
**Logic**: Reduces search time complexity to O(log n).
**Example**: Finding an element in a sorted array.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### 10. Prefix Sum
**Approach**: Precompute the sum of elements to quickly calculate the sum of any subarray.
**Logic**: Useful for range query problems.
**Example**: Finding the sum of elements between two indices in an array.

```python
def prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, left, right):
    return prefix[right + 1] - prefix[left]

arr = [1, 2, 3, 4, 5]
prefix = prefix_sum(arr)
print(range_sum(prefix, 1, 3))  # Output: 9 (2 + 3 + 4)
```
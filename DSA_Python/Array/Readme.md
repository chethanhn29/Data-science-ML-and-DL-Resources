# [Arrays Data Structure](https://www.geeksforgeeks.org/complete-guide-to-arrays-data-structure/?ref=array_lp)
### [Arrays introduction and time complexity of each Basic Operation](https://www.geeksforgeeks.org/python-arrays/)
### [Basic Operations and Questions based on Array data structure](https://www.geeksforgeeks.org/array-data-structure/)

### [Applications, Advantages and Disadvantages of Array](https://www.geeksforgeeks.org/applications-advantages-and-disadvantages-of-array-data-structure/)

### [Array Methods](https://www.geeksforgeeks.org/array-python-set-1-introduction-functions/)

### [Array Coursera video](https://www.coursera.org/lecture/data-structures/arrays-OsBSF)


## Arrays

Arrays are fundamental data structures that store elements of the same data type in contiguous memory locations. Here are the main properties and essential aspects to know about arrays:

- **Ordered Collection:** Arrays maintain elements in a specific order and allow access to elements by their index. The first element is at index 0, the second at index 1, and so on.
  
- **Fixed Size:** Arrays have a fixed size determined at the time of creation. The size typically remains constant throughout the array's lifetime.

- **Contiguous Memory:** Elements in an array are stored in contiguous memory locations, facilitating faster access to elements compared to other data structures like linked lists.

- **Random Access:** Arrays provide constant-time access to elements using their indices, allowing quick retrieval and modification of elements.

- **Homogeneous Elements:** Arrays store elements of the same data type, ensuring uniformity within the array.

- **Static vs. Dynamic Arrays:** Some programming languages offer dynamic arrays that can resize dynamically to accommodate more elements beyond their initial size. Static arrays, on the other hand, have a fixed size that cannot be changed during runtime.

- **Complexity Analysis:**
  - Access: O(1)
  - Search: O(n)
  - Insertion/Deletion (at arbitrary position):
    - Worst case: O(n)
    - Average case: O(n/2)

- **Usage:**
  - Storing and accessing a collection of elements where direct access to elements by index is required.
  - Implementing algorithms that require sequential access to elements.

- **Considerations:**
  - Arrays are suitable for a fixed-size collection of elements with known bounds.
  - They may not be the best choice for frequent insertions and deletions in large arrays due to the shifting of elements.
  - Dynamic arrays may offer more flexibility but can incur memory reallocation overhead.

- **Languages Support:** Arrays are supported in various programming languages such as C, C++, Java, Python, JavaScript, and many others.

![](https://i.stack.imgur.com/Ly4Fp.jpg)


**Comprehensive Revision Notes: Arrays**

**Definition of an Array**:
- An array is a contiguous block of memory, either allocated on the stack or the heap.
- It consists of equally sized elements indexed by contiguous integers.
- Arrays can be one-dimensional or multi-dimensional, containing rows and columns.

**Indexing in Arrays**:
- Array indices typically start from 0 in many programming languages, but other languages may support 1-based indexing or allow specification of the initial index.
- Random access: Arrays offer constant-time access to any element, both for reading and writing.
- Address calculation: The address of a particular array element is computed using simple arithmetic, involving the array address, element size, and index.

**Multi-dimensional Arrays**:
- Multi-dimensional arrays can be simulated in languages that do not directly support them.
- Address calculation involves skipping rows and columns to reach the desired element.
- Arrays can be laid out in memory using row-major ordering or column-major ordering.

**Operations on Arrays**:
- Reading or writing any element in an array is O(1).
- Adding or removing elements at the end of an array is O(1) if there is available space.
- Adding or removing elements at the beginning or middle of an array is O(n) due to the need for shifting elements.

**Summary**:
- Arrays provide constant-time access to elements.
- They consist of contiguous memory blocks with equally sized elements indexed by contiguous integers.
- Operations such as adding or removing elements at the end are efficient, but adding or removing elements elsewhere in the array is costly.

**Additional Notes**:
- Arrays are fundamental data structures used in various algorithms and applications.
- Understanding array indexing, address calculation, and operations is crucial for efficient programming.
- Arrays are suitable for scenarios where random access to elements is required, but they may not be optimal for dynamic resizing or frequent insertions/deletions in the middle.

In summary, arrays offer efficient access to elements and are suitable for scenarios where constant-time access is essential. However, they have limitations in dynamic resizing and inserting/removing elements in the middle, where their performance degrades to linear time. Understanding these characteristics helps in effectively utilizing arrays in programming tasks and algorithm design.

Certainly! Here's a table outlining different time and memory complexities associated with common operations on arrays:

| Operation                 | Time Complexity (Average/Amortized) | Memory Complexity |
|---------------------------|--------------------------------------|-------------------|
| Accessing an element      | O(1)                                 | O(1)              |
| Updating an element       | O(1)                                 | O(1)              |
| Adding element (End)      | O(1)                                 | O(1)              |
| Adding element (Start)    | O(n)                                 | O(1)              |
| Adding element (Middle)   | O(n)                                 | O(1)              |
| Removing element (End)    | O(1)                                 | O(1)              |
| Removing element (Start)  | O(n)                                 | O(1)              |
| Removing element (Middle) | O(n)                                 | O(1)              |

This table summarizes the time and memory complexities associated with various operations on arrays. It's important to note that these complexities represent average or amortized cases and may vary depending on factors such as implementation details, hardware characteristics, and programming language optimizations.


##  Grokking Algorithms in Python 
In Grokking Algorithms, several patterns are discussed to solve algorithmic problems efficiently. While arrays are fundamental data structures, they are often used in conjunction with these patterns to solve various problems. Here are some Grokking Algorithms patterns that can be applied to arrays:

1. **Sliding Window**:
   - Use a fixed-size window to iterate through elements in the array.
   - Suitable for problems requiring optimization over a subarray or substring.
   - Examples: Maximum Sum Subarray, Longest Substring without Repeating Characters.

2. **Two Pointers**:
   - Use two pointers to iterate through the array simultaneously.
   - Often used to solve problems involving searching, comparing, or manipulating elements.
   - Examples: Two Sum, Remove Duplicates from Sorted Array.

3. **Merge Intervals**:
   - Combine overlapping or adjacent intervals in the array.
   - Typically applied to problems involving intervals or ranges.
   - Examples: Merge Intervals, Meeting Rooms II.

4. **In-place Reversal of a Linked List**:
   - Reverse a contiguous subset of elements in the array.
   - Useful for problems requiring reversing or rearranging elements.
   - Examples: Reverse Words in a String, Rotate Image.

5. **Cyclic Sort**:
   - Arrange elements in the array by cyclically sorting them.
   - Applicable when elements range from 1 to N and there are no duplicates.
   - Examples: Find the Missing Number, Find the Duplicate Number.

6. **Top K Elements**:
   - Maintain a collection of top K elements from the array.
   - Useful for finding the largest or smallest elements efficiently.
   - Examples: Kth Largest Element in an Array, Top K Frequent Elements.

7. **Binary Search**:
   - Apply binary search to efficiently locate elements in a sorted array.
   - Ideal for problems requiring searching or partitioning elements.
   - Examples: Binary Search, Search in Rotated Sorted Array.


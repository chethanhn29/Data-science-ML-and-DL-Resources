
## Table of Contents

1. [Python Interview Preparation](#python-interview-preparation)
2. [DSA (Data Structures and Algorithms) Interview Preparation](#dsa-interview-preparation)


## Python Interview Preparation

- [Python Cheatsheet, start with this first](https://leetcode.com/discuss/study-guide/2122306/Python-Cheat-Sheet-for-Leetcode)
- [Resources for Interview preparation for Python](https://drive.google.com/drive/folders/1vXoWHuaYO-f4cIu6OwTyBzpO3Dehl8Ad?usp=drive_link)
- [Python Interview questions](https://www.interviewbit.com/python-interview-questions/)
- [Python Cheat sheet](https://www.interviewbit.com/python-cheat-sheet/)
- [Coding Tips](https://realpython.com/python-coding-interview-tips/)
- [Python Data Structures and Time Complexities by Aman.ai](https://aman.ai/code/data-structures/)
- [ Asymptotic Notations](https://aman.ai/code/asymptotic-notations/)
- [Pandas Funtions](https://aman.ai/primers/pandas/)
- [Python Guide by Aman.ai](https://aman.ai/primers/python/#google_vignette)

## DSA Interview Preparation
- [General tips for DSA problems by Aman.ai](https://aman.ai/code/#general-tips)
- [Understanding Time Complexity](https://towardsdatascience.com/understanding-time-complexity-with-python-examples-2bda6e8158a7)
- [Resources for Interview preparation for DSA](https://drive.google.com/drive/folders/19LTeYCi0K4fnkyX7D1HiDmlWua5PkrWr?usp=drive_link)
- ### [Go through with this link](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/tree/main/Python%20Notebooks/Data_Structructres_and_Algorithms)
- [Data Structures Interview questions](https://www.interviewbit.com/data-structure-interview-questions/)
- [DSA Revision and Notes for each Data structure](https://www.programiz.com/dsa/algorithm)
- [Time Complexity for different Data Structures](https://wiki.python.org/moin/TimeComplexity)
- [Data Structures and Algorithms Implementation in Python](https://github.com/campusx-official/dsa-using-python/tree/main)
- [Common Leetcode questions asked by companies](https://mlengineer.io/common-leetcode-questions-by-categories-532b301130b)



### Time and Space Complexity for Common Data Structures in Python

![Time Complexity for sorting](https://he-s3.s3.amazonaws.com/media/uploads/c950295.png)

| Data Structure   | Insertion | Deletion | Traversal | Search | Sorting | Accessing | Space Complexity | Best at             | Worst at  |
|------------------|-----------|----------|-----------|--------|---------|-----------|------------------|---------------------|------------|
| List             | O(1)      | O(1)     | O(n)      | O(n)   | O(n log n) | O(1)     | O(n)             | Accessing           | Sorting    |
| Tuple            | N/A       | N/A      | O(n)      | O(n)   | N/A     | O(1)      | O(n)             | Accessing           | -          |
| Set              | O(1)      | O(1)     | N/A       | O(1)   | N/A     | N/A       | O(n)             | Search              | Traversal  |
| Dictionary       | O(1)      | O(1)     | N/A       | O(1)   | N/A     | O(1)      | O(n)             | Search              | Traversal  |
| Stack            | O(1)      | O(1)     | O(n)      | O(n)   | N/A     | O(1)      | O(n)             | Insertion, Deletion, Accessing | Traversal  |
| Queue            | O(1)      | O(1)     | O(n)      |

| Data Structure     | Insertion | Deletion | Traversal | Search | Sorting | Accessing | Space Complexity | Best at            | Worst at  |
|---------------------|-----------|----------|-----------|--------|---------|-----------|------------------|--------------------|-----------|
| Singly Linked List | O(1)      | O(1)     | O(n)      | O(n)   | N/A     | O(n)      | O(n)             | Insertion, Deletion, Traversal | Accessing |
| Doubly Linked List | O(1)      | O(1)     | O(n)      | O(n)   | N/A     | O(n)      | O(n)             | Insertion, Deletion, Traversal | Accessing |
| Dynamic Array      | O(1)*     | O(1)*    | O(n)      | O(n)   | O(n log n) | O(1)     | O(n)             | Accessing          | Insertion, Deletion |

Note:
- 'n' represents the number of elements in the data structure.
- For dynamic arrays, amortized time complexities for insertion and deletion are O(1)*, but in worst-case scenarios, they can be O(n).
- In the "Best at" column, I've mentioned the operation at which the data structure performs most efficiently compared to others. In the "Worst at" column, I've mentioned the operation at which the data structure performs less efficiently compared to others.


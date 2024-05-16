class Array:
    def __init__(self, capacity):
        self.size = 0  # Initialize the current size of the array to 0
        self.capacity = capacity  # Maximum capacity of the array
        self.data = [None] * capacity  # Initialize the array with 'capacity' number of elements, all set to None

    # Get Element by Index
    def get(self, index):
        if 0 <= index < self.size:
            return self.data[index]  # Retrieve element at the specified index
        else:
            return "Index out of bounds"  # Return an error message for out-of-bounds index access

    # Set Element by Index
    def set(self, index, value):
        if 0 <= index < self.size:
            self.data[index] = value  # Set the value at the specified index
        else:
            print("Index out of bounds")  # Display an error message for out-of-bounds index access

    # Insert Element at Index
    def insert(self, index, value):
        if index < 0 or index > self.size:
            print("Index out of bounds")  # Display an error message for out-of-bounds index access
            return
        
        if self.size == self.capacity:
            self.resize(2 * self.capacity)  # Resize array if it's at capacity

        # Shift elements to the right to make space for the new value at the specified index
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]

        self.data[index] = value  # Insert the value at the specified index
        self.size += 1  # Increase the size of the array

    # Delete Element at Index
    def delete(self, index):
        if index < 0 or index >= self.size:
            print("Index out of bounds")  # Display an error message for out-of-bounds index access
            return

        # Shift elements to the left to overwrite the value at the specified index
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]

        self.size -= 1  # Decrease the size of the array after deletion
        self.data[self.size] = None  # Set the last element to None

    # Resize the array to a new capacity
    def resize(self, new_capacity):
        new_data = [None] * new_capacity  # Create a new array with the new capacity
        for i in range(self.size):
            new_data[i] = self.data[i]  # Copy elements to the new array
        self.data = new_data  # Replace the old array with the new array
        self.capacity = new_capacity  # Update the capacity

    # Display the array elements up to the current size
    def display(self):
        print(self.data[:self.size])

    # For representation of class object
    def __repr__(self):
        return f"Array data: {self.data[:self.size]}"

# Example usage:
arr = Array(5)  # Create an array of capacity 5
arr.set(0, 10)  # Set value 10 at index 0
arr.set(1, 20)  # Set value 20 at index 1
arr.set(2, 30)  # Set value 30 at index 2
arr.display()   # Output: [10, 20, 30]

arr.insert(1, 15)  # Insert value 15 at index 1
arr.display()      # Output: [10, 15, 20, 30]

arr.delete(2)  # Delete element at index 2
arr.display()  # Output: [10, 15, 30]

print(arr.get(1))  # Output: 15 (Retrieve element at index 1)
print(arr.get(5))  # Output: Index out of bounds (Trying to access out-of-bounds index)

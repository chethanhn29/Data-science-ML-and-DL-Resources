## Generator

Generators in Python are a powerful tool for creating iterators. They allow you to iterate through a sequence of values lazily, meaning they generate items one at a time and only when needed. This can be very efficient for large data sets or streams of data where you don’t want to hold the entire sequence in memory.

### Basic Concepts

1. **Generator Functions**:
   - Defined like a regular function but use the `yield` statement to return values one at a time.
   - When called, they return a generator object without starting execution immediately.
   - Execution resumes from where it left off each time `next()` is called on the generator object.

2. **Generator Expressions**:
   - Similar to list comprehensions but with parentheses instead of square brackets.
   - Produce generator objects.

### Simple Example

Here’s a simple generator function to generate a sequence of numbers:

```python
def simple_generator():
    yield 1
    yield 2
    yield 3

# Using the generator
gen = simple_generator()

print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
print(next(gen))  # Output: 3
# print(next(gen))  # Uncommenting this will raise StopIteration as there are no more items
```

### Using Generators in a Loop

You can also iterate over a generator using a loop:

```python
def simple_generator():
    yield 1
    yield 2
    yield 3

for value in simple_generator():
    print(value)

# Output:
# 1
# 2
# 3
```

### Generator Expressions

Here’s a generator expression example:

```python
gen_expr = (x * x for x in range(5))

for value in gen_expr:
    print(value)

# Output:
# 0
# 1
# 4
# 9
# 16
```

### Advanced Example: Fibonacci Sequence

Let’s create a more complex generator function to produce an infinite sequence of Fibonacci numbers:

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Using the generator
fib = fibonacci()

for _ in range(10):
    print(next(fib))

# Output (first 10 Fibonacci numbers):
# 0
# 1
# 1
# 2
# 3
# 5
# 8
# 13
# 21
# 34
```

### Generators with `send`

Generators can also receive values. This can be done using the `send()` method. Here’s an example:

```python
def echo():
    while True:
        received = yield
        print(f"Received: {received}")

gen = echo()
next(gen)  # Prime the generator

gen.send("Hello")
gen.send("World")

# Output:
# Received: Hello
# Received: World
```

### Generator for File Processing

Generators can be very useful for processing large files line-by-line:

```python
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line

# Using the generator
for line in read_large_file('large_file.txt'):
    print(line.strip())
```

### Summary

- **Generators** are a way to create iterators in a lazy fashion.
- **Generator functions** use `yield` to return values one at a time.
- **Generator expressions** are like list comprehensions but for generators.
- **Advanced usage** includes receiving values with `send()` and handling infinite sequences.

Sure! Here's a comparison table highlighting the key differences between the `yield` and `return` statements in Python:

| Feature                         | `yield`                                           | `return`                                    |
|---------------------------------|---------------------------------------------------|---------------------------------------------|
| **Purpose**                     | Produces a value and pauses the function execution, allowing it to resume later | Ends the function execution and returns a value |
| **Function Type**               | Creates a generator function                      | Creates a regular function                  |
| **Execution**                   | Suspends the function state, allowing iteration   | Terminates the function completely          |
| **Memory Usage**                | More memory efficient for large data sets         | May consume more memory if returning large data sets |
| **Resumability**                | Can resume execution from where it was paused     | Cannot resume; function starts over on each call |
| **Return Value**                | Returns a generator object                        | Returns the specified value or None         |
| **State Preservation**          | Preserves the function's state between yields     | No state preservation between calls         |
| **Iteration**                   | Suitable for producing a sequence of values       | Not suitable for producing sequences        |
| **Use Case**                    | Iterating over large data sets, infinite sequences, lazy evaluation | Single result computation, returning a single value or structure |
| **Example Usage**               | `yield value`                                     | `return value`                              |
| **Complexity**                  | Can handle more complex iteration patterns        | Simpler and more straightforward            |
| **Performance**                 | Can improve performance by avoiding unnecessary computations and memory usage | Performance depends on function complexity and return value size |

### Example Illustrations

#### Using `yield`:

```python
def generate_numbers():
    for i in range(5):
        yield i

gen = generate_numbers()
print(next(gen))  # Output: 0
print(next(gen))  # Output: 1
```

#### Using `return`:

```python
def return_numbers():
    return [i for i in range(5)]

numbers = return_numbers()
print(numbers)  # Output: [0, 1, 2, 3, 4]
```

### Summary

- **`yield`** is used for creating generators which are more memory efficient and allow for complex iteration patterns without holding all data in memory.
- **`return`** is used for ending a function and returning a value, suitable for simpler, non-iterative tasks where the function result is needed immediately.

##  Decorators
Decorators in Python are a powerful and expressive tool for modifying or enhancing the behavior of functions or methods. They are a form of metaprogramming and allow for the augmentation of function behavior without modifying the function's code directly.

### Basic Concepts

1. **Function Decorators**:
   - A decorator is a function that takes another function as an argument and extends or alters its behavior.
   - Decorators are applied using the `@decorator_name` syntax above the function definition.

2. **Class Decorators**:
   - Similarly, decorators can be used to modify or extend the behavior of classes.

### Simple Example

Let's start with a simple decorator that prints a message before and after a function is called:

```python
def simple_decorator(func):
    def wrapper():
        print("Before the function call")
        func()
        print("After the function call")
    return wrapper

@simple_decorator
def say_hello():
    print("Hello!")

say_hello()

# Output:
# Before the function call
# Hello!
# After the function call
```

### How Decorators Work

1. **Define the Decorator Function**:
   - The decorator function takes a function as an argument and returns a new function (the wrapper).

2. **Apply the Decorator**:
   - The `@decorator_name` syntax applies the decorator to the function, which is then replaced by the wrapper function.

### Decorators with Arguments

To create a more versatile decorator that accepts arguments, you need to add another layer of function definitions:

```python
def decorator_with_args(arg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Argument passed to decorator: {arg}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@decorator_with_args("Hello")
def greet(name):
    print(f"Greetings, {name}!")

greet("Alice")

# Output:
# Argument passed to decorator: Hello
# Greetings, Alice!
```

### Preserving Function Metadata

When you use decorators, the metadata of the original function (like its name and docstring) is lost. To preserve it, use `functools.wraps`:

```python
import functools

def simple_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Before the function call")
        result = func(*args, **kwargs)
        print("After the function call")
        return result
    return wrapper

@simple_decorator
def say_hello():
    """This function says hello"""
    print("Hello!")

print(say_hello.__name__)  # Output: say_hello
print(say_hello.__doc__)   # Output: This function says hello
```

### Class Decorators

Decorators can also be used to modify classes. Here’s an example of a class decorator:

```python
def class_decorator(cls):
    cls.extra_attribute = "This is an extra attribute"
    return cls

@class_decorator
class MyClass:
    pass

obj = MyClass()
print(obj.extra_attribute)  # Output: This is an extra attribute
```

### Advanced Example: Timing Function Execution

Let's create a decorator that times the execution of a function:

```python
import time
import functools

def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def long_running_function():
    time.sleep(2)
    print("Function complete")

long_running_function()

# Output:
# Function complete
# Function long_running_function took 2.0001 seconds
```

### Summary

- **Decorators** modify or enhance the behavior of functions or methods.
- **Function Decorators**: Modify functions by wrapping them.
- **Class Decorators**: Modify classes by adding attributes or methods.
- **Preserving Metadata**: Use `functools.wraps` to maintain original function metadata.
- **Advanced Usage**: Include arguments in decorators and apply decorators to classes or methods for more complex behavior modifications.

By understanding these concepts and examples, you can effectively use decorators to write more modular, reusable, and readable code in Python.

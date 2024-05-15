### comprehensive regex cheat sheet with Python code examples for each pattern:

### Regex Basics

| Pattern | Description | Example | Python Code |
|---------|-------------|---------|-------------|
| `.` | Matches any single character except newline | `a.b` matches `acb`, `a1b` | `re.search(r'a.b', 'a1b')` |
| `\d` | Matches any digit, equivalent to `[0-9]` | `\d\d` matches `12`, `99` | `re.search(r'\d\d', '123')` |
| `\D` | Matches any non-digit | `\D` matches `a`, `#` | `re.search(r'\D', 'a1')` |
| `\w` | Matches any word character (alphanumeric + underscore) | `\w\w` matches `ab`, `a1` | `re.search(r'\w\w', 'ab')` |
| `\W` | Matches any non-word character | `\W` matches `!`, ` ` | `re.search(r'\W', '!')` |
| `\s` | Matches any whitespace character (space, tab, newline) | `\s` matches ` `, `\t` | `re.search(r'\s', ' ')` |
| `\S` | Matches any non-whitespace character | `\S` matches `a`, `1` | `re.search(r'\S', 'a ')` |
| `^` | Matches the start of a string | `^a` matches `a` in `apple` | `re.search(r'^a', 'apple')` |
| `$` | Matches the end of a string | `e$` matches `e` in `apple` | `re.search(r'e$', 'apple')` |

### Quantifiers

| Pattern | Description | Example | Python Code |
|---------|-------------|---------|-------------|
| `*` | Matches 0 or more repetitions | `a*` matches `a`, `aaa`, ` ` | `re.search(r'a*', 'aaa')` |
| `+` | Matches 1 or more repetitions | `a+` matches `a`, `aaa` | `re.search(r'a+', 'aaa')` |
| `?` | Matches 0 or 1 repetition | `a?` matches `a`, ` ` | `re.search(r'a?', 'a')` |
| `{n}` | Matches exactly n repetitions | `a{3}` matches `aaa` | `re.search(r'a{3}', 'aaa')` |
| `{n,}` | Matches n or more repetitions | `a{2,}` matches `aa`, `aaa` | `re.search(r'a{2,}', 'aaa')` |
| `{n,m}` | Matches between n and m repetitions | `a{2,3}` matches `aa`, `aaa` | `re.search(r'a{2,3}', 'aaa')` |

### Character Classes

| Pattern | Description | Example | Python Code |
|---------|-------------|---------|-------------|
| `[abc]` | Matches any one of the characters a, b, or c | `[abc]` matches `a`, `b`, `c` | `re.search(r'[abc]', 'b')` |
| `[^abc]` | Matches any character not listed | `[^abc]` matches `d`, `1` | `re.search(r'[^abc]', 'd')` |
| `[a-z]` | Matches any character in the range a-z | `[a-z]` matches `a`, `m`, `z` | `re.search(r'[a-z]', 'm')` |
| `[A-Z]` | Matches any character in the range A-Z | `[A-Z]` matches `A`, `M`, `Z` | `re.search(r'[A-Z]', 'M')` |
| `[0-9]` | Matches any character in the range 0-9 | `[0-9]` matches `0`, `5` | `re.search(r'[0-9]', '5')` |

### Groups and Alternations

| Pattern | Description | Example | Python Code |
|---------|-------------|---------|-------------|
| `(abc)` | Matches the exact sequence abc | `(abc)` matches `abc` | `re.search(r'(abc)', 'abc')` |
| `a|b` | Matches either a or b | `a|b` matches `a`, `b` | `re.search(r'a|b', 'a')` |
| `(a|b)c` | Matches `ac` or `bc` | `(a|b)c` matches `ac`, `bc` | `re.search(r'(a|b)c', 'ac')` |
| `(?:abc)` | Non-capturing group, matches abc | `(?:abc)` matches `abc` | `re.search(r'(?:abc)', 'abc')` |

### Special Sequences

| Pattern | Description | Example | Python Code |
|---------|-------------|---------|-------------|
| `\b` | Matches a word boundary | `\bword\b` matches `word` | `re.search(r'\bword\b', 'word')` |
| `\B` | Matches a non-word boundary | `\Bword\B` matches `swordfish` (not at boundaries) | `re.search(r'\Bword\B', 'swordfish')` |
| `\0` | Matches null character | `\0` | `re.search(r'\0', '\0')` |
| `\n` | Matches newline | `\n` | `re.search(r'\n', '\n')` |
| `\t` | Matches tab | `\t` | `re.search(r'\t', '\t')` |

### Assertions

| Pattern | Description | Example | Python Code |
|---------|-------------|---------|-------------|
| `(?=...)` | Positive lookahead: Asserts that what follows the current position is ... | `a(?=b)` matches `a` in `ab` | `re.search(r'a(?=b)', 'ab')` |
| `(?!...)` | Negative lookahead: Asserts that what follows the current position is not ... | `a(?!b)` matches `a` in `ac` | `re.search(r'a(?!b)', 'ac')` |
| `(?<=...)` | Positive lookbehind: Asserts that what precedes the current position is ... | `(?<=a)b` matches `b` in `ab` | `re.search(r'(?<=a)b', 'ab')` |
| `(?<!...)` | Negative lookbehind: Asserts that what precedes the current position is not ... | `(?<!a)b` matches `b` in `cb` | `re.search(r'(?<!a)b', 'cb')` |

### Examples

1. **Simple Match**: `hello`
   ```python
   import re
   pattern = r"hello"
   print(re.search(pattern, "hello world"))  # Matches "hello"
   ```

2. **Digit Check**: `\d+`
   ```python
   pattern = r"\d+"
   print(re.search(pattern, "There are 123 apples"))  # Matches "123"
   ```

3. **Email Validation**: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
   ```python
   pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
   print(re.search(pattern, "contact@example.com"))  # Matches "contact@example.com"
   ```

4. **Find All Words**: `\b\w+\b`
   ```python
   pattern = r"\b\w+\b"
   print(re.findall(pattern, "This is a test."))  # Matches ["This", "is", "a", "test"]
   ```

5. **Phone Number Validation**: `\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}`
   ```python
   pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
   print(re.search(pattern, "Call me at (123) 456-7890"))  # Matches "(123) 456-7890"
   ```

6. **Non-Greedy Match**: `.*?`
   ```python
   pattern = r"<.*?>"
   print(re.findall(pattern, "<div>Test</div><span>Example</span>"))  # Matches ["<div>", "<span>"]
   ```

### Python Regex Functions

1. **Search**: `re.search()`
   ```python
   import re
   pattern = r'hello'
   match = re.search(pattern, 'hello world')
   print(match.group())  # Output: 'hello'
   ```

2. **Match**: `re.match()`
   ```python
   pattern = r'hello'
   match = re.match(pattern, 'hello world')
   print(match.group())  # Output: 'hello'
   ```

3. **Find All**: `re.findall()`
   ```python
   pattern = r'\d+'
   matches = re.findall(pattern, '123 apples and 456 oranges')
   print

(matches)  # Output: ['123', '456']
   ```

4. **Find Iter**: `re.finditer()`

   ```python
   pattern = r'\d+'
   matches = re.finditer(pattern, '123 apples and 456 oranges')
   for match in matches:
       print(match.group())  # Output: '123' and '456'
   ```

5. **Substitute**: `re.sub()`
   ```python
   pattern = r'apples'
   replacement = 'oranges'
   text = 'I like apples'
   result = re.sub(pattern, replacement, text)
   print(result)  # Output: 'I like oranges'
   ```

6. **Split**: `re.split()`
   ```python
   pattern = r'\s+'
   text = 'Split this sentence into words'
   result = re.split(pattern, text)
   print(result)  # Output: ['Split', 'this', 'sentence', 'into', 'words']
   ```


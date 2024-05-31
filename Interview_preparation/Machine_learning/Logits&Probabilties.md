In machine learning, particularly in classification tasks, "logits" and "probabilities" are related but distinct concepts. Understanding the difference between them is crucial for interpreting model outputs and loss functions.

### Logits

- **Definition**: Logits are the raw, unnormalized scores output by a model's final layer (before any activation function like softmax is applied).
- **Range**: They can take any real value, from negative infinity to positive infinity.
- **Usage**: Logits are often used as inputs to the softmax function to obtain probabilities for multi-class classification tasks.
- **Example**: If a neural network for a classification task outputs logits \([2.0, 1.0, 0.1]\), these are the raw scores before any transformation to probabilities.

### Probabilities

- **Definition**: Probabilities are the normalized scores that indicate the likelihood of each class. They are obtained by applying a softmax function to the logits.
- **Range**: They range from 0 to 1 and sum to 100% (or 1 for each instance).
- **Usage**: Probabilities are used for making predictions (e.g., choosing the class with the highest probability) and for evaluating the model's performance using metrics like cross-entropy loss.
- **Example**: Applying the softmax function to the logits \([2.0, 1.0, 0.1]\) might yield probabilities \([0.65, 0.24, 0.11]\), which sum to 1.

### Mathematical Transformation

The softmax function converts logits to probabilities. Given logits \(\mathbf{z} = [z_1, z_2, \ldots, z_n]\), the softmax function is defined as:

\[ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} \]

This ensures that the resulting probabilities are non-negative and sum to 1.

### Example Calculation

Let's calculate the softmax probabilities for the logits \([2.0, 1.0, 0.1]\):

1. Calculate the exponentials of the logits:
   \[
   e^{2.0} \approx 7.39, \quad e^{1.0} \approx 2.72, \quad e^{0.1} \approx 1.10
   \]

2. Sum the exponentials:
   \[
   7.39 + 2.72 + 1.10 \approx 11.21
   \]

3. Divide each exponential by the sum to get the probabilities:
   \[
   \text{softmax}(2.0) = \frac{7.39}{11.21} \approx 0.66, \quad
   \text{softmax}(1.0) = \frac{2.72}{11.21} \approx 0.24, \quad
   \text{softmax}(0.1) = \frac{1.10}{11.21} \approx 0.10
   \]

So the logits \([2.0, 1.0, 0.1]\) are transformed to probabilities \([0.66, 0.24, 0.10]\).

### Key Points

- **Logits** are raw model outputs that can be any real number. They provide unnormalized scores for each class.
- **Probabilities** are normalized logits obtained by applying the softmax function, making them interpretable as likelihoods of each class.

### Use Cases

- **Logits**: Used during the training phase as inputs to the loss function (e.g., cross-entropy loss often directly uses logits for numerical stability).
- **Probabilities**: Used for prediction and evaluation, providing a clear indication of the model's confidence in each class.

By understanding the difference between logits and probabilities, you can better interpret the outputs of your classification models and correctly implement loss functions and evaluation metrics.

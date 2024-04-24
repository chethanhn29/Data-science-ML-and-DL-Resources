
## [Weight intialization by Deeplearning.ai](https://www.deeplearning.ai/ai-notes/initialization/index.html)
## [How-to-initialize-weights-in-neural-networks](https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/)
### [Blog](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78)
### Why Weight Initialization?
Its main objective is to prevent layer activation outputs from exploding or vanishing gradients during the forward propagation. If either of the problems occurs, loss gradients will either be too large or too small, and the network will take more time to converge if it is even able to do so at all.

If we initialized the weights correctly, then our objective i.e, optimization of loss function will be achieved in the least time otherwise converging to a minimum using gradient descent will be impossible.

### Zero Initialization (Initialized all weights to 0)
**If we initialized all the weights with 0, then what happens is that the derivative wrt loss function is the same for every weight in W[l], thus all weights have the same value in subsequent iterations.** This makes hidden layers symmetric and this process continues for all the n iterations. Thus initialized weights with zero make your network no better than a linear model. **It is important to note that setting biases to 0 will not create any problems as non-zero weights take care of breaking the symmetry and even if bias is 0, the values in every neuron will still be different.**
### Random Intialization
#### Uniform
#### HE
#### Xavier
#### Standard Normal

#### How to find appropriate initialization values
To prevent the gradients of the network’s activations from vanishing or exploding, we will stick to the following rules of thumb:

- The mean of the activations should be zero.
- The variance of the activations should stay the same across every layer.

Parameter initialization is the process of assigning initial values to the parameters (weights and biases) of a neural network before training. Proper initialization is crucial for effective training because it can affect the convergence speed, stability, and generalization performance of the model. Here's why parameter initialization is needed and the types of initialization methods commonly used:

### Why Parameter Initialization is Needed:

1. **Avoiding Vanishing/Exploding Gradients**: Poor initialization can lead to gradients becoming too small (vanishing) or too large (exploding), hindering the learning process.

2. **Promoting Convergence**: Proper initialization can help the model converge faster by providing a good starting point for optimization algorithms.

3. **Improving Generalization**: Well-initialized parameters can help prevent overfitting and improve the model's ability to generalize to unseen data.

### Types of Parameter Initialization:

#### 1. Zero Initialization:
   - **Description**: All weights and biases are initialized to zero.
   - **Usage**: Zero initialization is not commonly used in practice due to its symmetric nature, which can cause all neurons in a layer to compute the same output and hinder learning.

#### 2. Random Initialization:
   - **Description**: Weights and biases are randomly initialized using a distribution such as uniform or normal distribution.
   - **Usage**: Random initialization is widely used as it breaks the symmetry and introduces diversity in parameter values. Commonly used distributions include uniform or normal distributions.

#### 3. Xavier/Glorot Initialization:
   - **Description**: Weights are initialized using a distribution with zero mean and variance inversely proportional to the number of input and output units. Biases are often initialized to zeros.
   - **Usage**: Xavier initialization is effective for sigmoid and tanh activations, helping to keep activations within a reasonable range during forward and backward propagation.

#### 4. He Initialization:
   - **Description**: Similar to Xavier initialization, but the variance is scaled by a factor of 2 for ReLU activations to account for the sparsity of activations.
   - **Usage**: He initialization is commonly used with ReLU and its variants (e.g., Leaky ReLU, Parametric ReLU) to prevent vanishing gradients and accelerate convergence.

### Code Examples in TensorFlow and PyTorch:

#### TensorFlow (using Glorot uniform initialization):
```python
import tensorflow as tf

# Define a dense layer
layer = tf.keras.layers.Dense(units=100, activation='relu')

# Initialize layer parameters using Glorot uniform initialization
initializer = tf.initializers.GlorotUniform()
layer.kernel_initializer = initializer
layer.bias_initializer = tf.zeros_initializer()

# Example of how to use the initialized layer in a model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    layer,
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile and train the model
# (Code for training the model is not shown for brevity)
```

#### PyTorch (using He normal initialization):
```python
import torch
import torch.nn as nn

# Define a neural network module
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 100)  # Input size: 28x28=784
        self.fc2 = nn.Linear(100, 10)

        # Initialize parameters using He normal initialization for ReLU activations
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example of how to use the initialized model
model = MyModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
# (Code for training the model is not shown for brevity)
```

In these examples, we demonstrate how to initialize parameters using Glorot uniform initialization in TensorFlow and He normal initialization in PyTorch for a simple neural network. These initialization methods help promote stable training and improve the convergence of the model during optimization.

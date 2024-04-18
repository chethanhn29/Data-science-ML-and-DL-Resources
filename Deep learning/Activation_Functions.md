# Neural Networks Activation Functions
## [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
## [Activation Functions and their Derivatives – A Quick & Complete Guide](https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/)

## [Everything you need to know about “Activation Functions” in Deep learning models](https://towardsdatascience.com/everything-you-need-to-know-about-activation-functions-in-deep-learning-models-84ba9f82c253)
## Exploding-and-vanishing-gradients-problem
- [Article 1](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
- [Article 2](https://towardsdatascience.com/the-exploding-and-vanishing-gradients-problem-in-time-series-6b87d558d22)
- [Article 3](https://www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/)
- [Article 4](https://archive.is/K0DNg),
- https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11
- [Article 5](https://www.kdnuggets.com/2022/02/vanishing-gradient-problem.html)

## [To find Dead Neurons in Relu in DL Network](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html#Finding-dead-neurons-in-ReLU-networks)

## Table of Contents

1. [Introduction](#introduction)
2. [Activation Functions and their Derivatives](#activation-functions-and-derivatives)
3. [Exploding and Vanishing Gradients Problem](#exploding-and-vanishing-gradients-problem)
4. [Finding Dead Neurons in ReLU Networks](#finding-dead-neurons-in-relu-networks)
5. [Why do we need Activation Functions?](#why-do-we-need-them)
   - [Introducing Non-linearity](#introducing-non-linearity)
   - [Learning Complex Functions](#learning-complex-functions)
   - [Enabling Deep Networks](#enabling-deep-networks)
   - [Gradient Flow and Backpropagation](#gradient-flow-and-backpropagation)
   - [Avoiding Vanishing and Exploding Gradients](#avoiding-vanishing-and-exploding-gradients)
6. [Why Sigmoid and Tanh Can Suffer From Vanishing Gradients](#vanishing-gradients-problem)
7. [Linear and Non-Linear Functions](#linear-and-nonlinear-functions)
8. [Activation Functions in Neural Networks](#activation-functions)
   - [Sigmoid Hidden Layer Activation Function](#sigmoid-hidden-layer)
   - [Tanh Hidden Layer Activation Function](#tanh-hidden-layer)
   - [Activation for Output Layers](#activation-for-output-layers)
   - [Exponential Linear Unit (ELU)](#elu)
   - [Parametric Rectified Linear Unit (PReLU)](#prelu)
   - [Leaky Rectified Linear Unit (Leaky ReLU)](#leaky-relu)
   - [Swish Activation Function](#swish)
   - [GELU (Gaussian Error Linear Unit)](#gelu)
9. [Vanishing Gradient Problem](#vanishing-gradient-problem)
   - [Rectified Linear Units (ReLU)](#relu)


## 1. Introduction <a name="introduction"></a>

Activation functions are crucial components in artificial neural networks. They introduce non-linearity into the model, allowing neural networks to learn complex patterns and make them capable of approximating a wide range of functions.
![](https://images.deepai.org/django-summernote/2019-03-15/446c7799-5959-4555-9e75-411820e15d16.png)

A feed-forward neural network with linear activation and any number of hidden layers is equivalent to just a linear neural neural network with no hidden layer. For example lets consider the neural network in figure with two hidden layers and no activation enter image description here

![](https://i.stack.imgur.com/KEFNX.png)

- y = h2 * W3 + b3 
- = (h1 * W2 + b2) * W3 + b3
- = h1 * W2 * W3 + b2 * W3 + b3 
- = (x * W1 + b1) * W2 * W3 + b2 * W3 + b3 
- = x * W1 * W2 * W3 + b1 * W2 * W3 + b2 * W3 + b3 
-  = x * W' + b'
  
We can do the last step because combination of several linear transformation can be replaced with one transformation and combination of several bias term is just a single bias. The outcome is same even if we add some linear activation.

So we could replace this neural net with a single layer neural net.This can be extended to n layers.` This indicates adding layers doesn't increase the approximation power of a linear neural net at all. We need non-linear activation functions to approximate non-linear functions and most real world problems are highly complex and non-linear`. In fact when the activation function is non-linear, then a two-layer neural network with sufficiently large number of hidden units can be proven to be a universal function approximator.

### [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*p_hyqAtyI8pbt2kEl6siOQ.png)
## 2. Activation Functions and their Derivatives <a name="activation-functions-and-derivatives"></a>

[Activation Functions and their Derivatives – A Quick & Complete Guide](https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/)

This article provides a comprehensive guide to various activation functions and their derivatives, essential for understanding how gradients are computed during backpropagation.

## 3. Exploding and Vanishing Gradients Problem <a name="exploding-and-vanishing-gradients-problem"></a>

### Articles:
- [The Vanishing Gradient Problem](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
- [The Exploding and Vanishing Gradients Problem in Time Series](https://towardsdatascience.com/the-exploding-and-vanishing-gradients-problem-in-time-series-6b87d558d22)
- [The Challenge of Vanishing and Exploding Gradients in Deep Neural Networks](https://www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/)
- [Vanishing Gradient Problem](https://archive.is/K0DNg)
- [Vanishing Gradient Problem](https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11)
- [Vanishing Gradient Problem](https://www.kdnuggets.com/2022/02/vanishing-gradient-problem.html)

These articles discuss the challenges posed by exploding and vanishing gradients in deep neural networks and various strategies to overcome these problems.

## 4. Finding Dead Neurons in ReLU Networks <a name="finding-dead-neurons-in-relu-networks"></a>

[To find Dead Neurons in ReLU in DL Network](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html#Finding-dead-neurons-in-ReLU-networks)

This resource explains how to identify "dead" neurons in ReLU networks, which fail to activate during training, and offers solutions to address this issue.

## 5. Why do we need Activation Functions? <a name="why-do-we-need-them"></a>

- Activation functions are crucial components in artificial neural networks. They introduce non-linearity into the model, allowing neural networks to learn complex patterns and make them capable of approximating a wide range of functions. Here's why activation functions are needed in neural networks:
- Introducing Non-linearity: Activation functions add non-linearity, essential for modeling real-world non-linear relationships in data.
- Modeling Complex Patterns: They enable neural networks to capture intricate data patterns.
- Universal Function Approximators: Activation functions make networks capable of approximating any continuous function given enough neurons.
- Learning Non-linear Transformations: They allow networks to learn valuable data transformations.
- Facilitating Decision Boundaries: Activation functions define complex class boundaries in classification tasks.
- Learning Hierarchical Features: Hidden layers use them to learn hierarchical, abstract features.
- Stabilizing Training: They prevent issues like vanishing/exploding gradients, improving training stability.
- Enhancing Expressiveness: Different activation functions offer various characteristics, enhancing network capabilities for specific tasks.


### 5.1 Introducing Non-linearity <a name="introducing-non-linearity"></a>

- Without activation functions, the entire neural network would behave like a linear model. A linear combination of linear functions remains linear, and this limits the power of the network to model complex, non-linear relationships in data.
- Activation functions introduce non-linearity by applying a mathematical operation to the weighted sum of inputs. This non-linearity enables neural networks to approximate more complex and intricate patterns in data.
### 5.2 Learning Complex Functions <a name="learning-complex-functions"></a>

- Many real-world problems involve highly non-linear relationships. Activation functions enable neural networks to learn and represent these complex functions effectively.
- For example, in image classification tasks, the relationship between pixel values in an image and the class of an object is often highly non-linear. Activation functions allow neural networks to capture these intricate relationships.
### 5.3 Enabling Deep Networks <a name="enabling-deep-networks"></a>

- Activation functions are particularly important in deep neural networks, which have multiple hidden layers. Without non-linear activation functions, stacking multiple linear layers would still result in a linear model.
- Deep networks with non-linear activation functions can learn hierarchical features, allowing them to model abstract and hierarchical representations of data.
### 5.4 Gradient Flow and Backpropagation <a name="gradient-flow-and-backpropagation"></a>

- Activation functions play a crucial role in backpropagation, which is the training algorithm for neural networks. They determine how errors are propagated backward through the network during training.
- The derivative of the activation function is used to compute gradients, which guide the update of neural network weights through gradient descent optimization. Different activation functions have different gradient properties that can impact training stability and convergence.
### 5.5 Avoiding Vanishing and Exploding Gradients <a name="avoiding-vanishing-and-exploding-gradients"></a>

- Some activation functions, such as the sigmoid and hyperbolic tangent (tanh), can suffer from vanishing gradients, which slow down training in deep networks. Other activation functions, like ReLU (Rectified Linear Unit), were designed to mitigate this problem.
- Activation functions like ReLU also help prevent exploding gradients, where gradients become too large, causing instability during training.

Common activation functions used in neural networks include:

- **Sigmoid**: S-shaped curve, good for binary classification tasks.
- **Hyperbolic Tangent (tanh)**: Similar to sigmoid but with output values in the range [-1, 1].
- **ReLU (Rectified Linear Unit)**: Piecewise linear, widely used in deep learning due to its effectiveness and computational efficiency.
- **Leaky ReLU**: A variant of ReLU that allows a small gradient for negative inputs to address the "dying ReLU" problem.
- **Softmax**: Used in the output layer for multi-class classification to convert raw scores into class probabilities.

In summary, activation functions are essential in neural networks because they enable the modeling of complex, non-linear relationships in data, facilitate the training of deep networks, and help prevent issues like vanishing and exploding gradients during training. The choice of activation function depends on the specific problem and network architecture.

## 6. Why Sigmoid and Tanh Can Suffer From Vanishing Gradients <a name="vanishing-gradients-problem"></a>

Sigmoid and hyperbolic tangent (tanh) activation functions can suffer from the vanishing gradient problem, especially when used in deep neural networks. The vanishing gradient problem occurs during the training of neural networks when the gradients of the activation function become extremely small, causing the weights to update very slowly or not at all. This phenomenon can hinder the training process for several reasons:

### Saturation of the Activation Functions

- Both the sigmoid and tanh activation functions saturate when their inputs are very large or very small. In the sigmoid function, the output approaches 0 or 1, and in the tanh function, it approaches -1 or 1.
- When the activations are saturated, their gradients become close to zero. This means that during backpropagation, the gradients are small, leading to small weight updates. If the gradients are consistently close to zero, the network learns very slowly or not at all.

### Layer Stacking in Deep Networks

- In deep neural networks with many layers, the gradients can compound as they propagate backward through the layers. If the activation functions are sigmoid or tanh, the gradients can become exponentially smaller as they move from the output layer to the input layer.
- As a result, the early layers in the network receive extremely small gradients, making it difficult for them to learn meaningful features. This is often referred to as the "vanishing gradient" problem.

### Initialization Sensitivity

- The choice of initialization for network weights can exacerbate the vanishing gradient problem with sigmoid and tanh activations. If weights are initialized in a way that pushes the activations toward the saturated regions of these functions, it can lead to slow convergence or getting stuck in training.

To mitigate the vanishing gradient problem, researchers have developed alternative activation functions that have become more popular in deep learning, such as the Rectified Linear Unit (ReLU) and its variants. ReLU doesn't saturate for positive inputs, which helps prevent the vanishing gradient issue. Leaky ReLU and Parametric ReLU (PReLU) variants were also introduced to address the "dying ReLU" problem, where neurons could become inactive during training.

In summary, while sigmoid and tanh activation functions can be useful in certain contexts, they tend to suffer from vanishing gradient problems in deep neural networks due to their saturation properties. This can slow down or hinder the training process, which is why alternative activation functions like ReLU and its variants are often preferred for deep learning tasks.


## 2. Linear and Non-Linear Functions <a name="linear-and-nonlinear-functions"></a>

![Linear and Non-Linear Functions](https://images.deepai.org/django-summernote/2019-03-15/446c7799-5959-4555-9e75-411820e15d16.png)

A feed-forward neural network with linear activation and any number of hidden layers is equivalent to just a linear neural neural network with no hidden layer. This indicates adding layers doesn't increase the approximation power of a linear neural net at all. We need non-linear activation functions to approximate non-linear functions, as most real-world problems are highly complex and non-linear.

## 3. Activation Functions in Neural Networks <a name="activation-functions"></a>

Activation functions introduce non-linearity to the neural network, enabling it to learn complex relationships in the data.

### 3.1 Sigmoid Hidden Layer Activation Function <a name="sigmoid-hidden-layer"></a>

- **How it works:** The sigmoid activation function outputs values in the range 0 to 1.
- **Advantages:** Introduces non-linearity, good for binary classification.
- **Disadvantages:** Prone to the vanishing gradient problem.
- **Mathematical Expression:** 1 / (1 + e^-x)

![Sigmoid Activation Function](https://machinelearningmastery.com/wp-content/uploads/2021/08/sigmoid.png)

### 3.2 Tanh Hidden Layer Activation Function <a name="tanh-hidden-layer"></a>

- **How it works:** The tanh activation function outputs values in the range -1 to 1.
- **Advantages:** Zero-centered, stronger non-linearity compared to sigmoid.
- **Disadvantages:** Still prone to the vanishing gradient problem.
- **Mathematical Expression:** (e^x – e^-x) / (e^x + e^-x)

![Tanh Activation Function](https://miro.medium.com/v2/resize:fit:900/0*FIFkhXuir7JO0Utc.jpg)

### 3.3 Activation for Output Layers <a name="activation-for-output-layers"></a>

The output layer activation function depends on the task:

1. **Linear Activation Function:** Used for regression tasks.
   - **Advantages:** Directly outputs values without restriction.
   - **Disadvantages:** Limited representation for complex data.
   - **Mathematical Expression:** f(x) = x

![Activation Functions for Output Layers](https://machinelearningmastery.com/wp-content/uploads/2020/12/How-to-Choose-an-Output-Layer-Activation-Function.png)

2. **Logistic (Sigmoid):** Used for binary classification tasks.
   - **Advantages:** Outputs probabilities between 0 and 1.
   - **Disadvantages:** Prone to the vanishing gradient problem.
   - **Mathematical Expression:** f(x) = 1 / (1 + exp(-x))

3. **Softmax:** Used for multi-class classification tasks.
   - **Advantages:** Outputs probabilities for each class, ensuring they sum to 1.
   - **Disadvantages:** Prone to the vanishing gradient problem.
   - **Mathematical Expression:** exp(x) / sum(exp(x))

### 3.4 Exponential Linear Unit (ELU) <a name="elu"></a>

- **How it works:** ELU returns the input if it's positive; otherwise, it returns an exponential function of the input minus 1.
- **Advantages:** Smoother gradient for negative values, which can help with convergence.
- **Disadvantages:** Computationally more expensive than ReLU.
- **Mathematical Expression:** 
  - f(x) = x, if x ≥ 0
  - f(x) = α * (exp(x) - 1), if x < 0 (where α is a hyperparameter)

### 3.5 Parametric Rectified Linear Unit (PReLU) <a name="prelu"></a>

- **How it works:** PReLU is similar to ReLU but with a learnable parameter for negative input values.
- **Advantages:** Learns the optimal slope for negative values, potentially improving performance.
- **Disadvantages:** Increases model complexity and training time.
- **Mathematical Expression:** 
  - f(x) = x, if x ≥ 0
  - f(x) = α * x, if x < 0 (where α is a learnable parameter)

### 3.6 Leaky Rectified Linear Unit (Leaky ReLU) <a name="leaky-relu"></a>

- **How it works:** Leaky ReLU is similar to ReLU but with a small slope for negative input values instead of setting them to zero.
- **Advantages:** Prevents the dying ReLU problem, maintains some gradient flow for negative values.
- **Disadvantages:** May introduce some computational overhead.
- **Mathematical Expression:** 
  - f(x) = x, if x ≥ 0
  - f(x) = α * x, if x < 0 (where α is a small constant, e.g., 0.01)

### 3.7 Swish Activation Function <a name="swish"></a>

- **How it works:** Swish is a smooth, non-monotonic activation function that performs well in deep neural networks.
- **Advantages:** Smooth gradient, performs well in deep networks.
- **Disadvantages:** Computationally more expensive than ReLU.
- **Mathematical Expression:** 
  - f(x) = x * sigmoid(x)

### 3.8 GELU (Gaussian Error Linear Unit) <a name="gelu"></a>

- **How it works:** GELU is a smooth approximation of the ReLU function and has been shown to perform well in various deep

 learning tasks.
- **Advantages:** Smooth approximation of ReLU, performs well in practice.
- **Disadvantages:** Computationally more expensive than ReLU.
- **Mathematical Expression:** 
  - f(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))

## 4. Vanishing Gradient Problem <a name="vanishing-gradient-problem"></a>

The vanishing gradient problem occurs when gradients become extremely small during backpropagation, preventing the network from learning effectively.

### 4.1 Rectified Linear Units (ReLU) <a name="relu"></a>

- **How it works:** ReLU sets negative values to zero, providing faster training compared to sigmoid and tanh. However, it can suffer from the dying ReLU problem.
- **Advantages:** Fast training, sparse activation, avoids vanishing gradient.
- **Disadvantages:** Can suffer from the dying ReLU problem.
- **Mathematical Expression:** 
  - f(x) = max(0, x)

![Vanishing Gradient Problem](https://www.kdnuggets.com/wp-content/uploads/jacob-vanishing-gradient-7.jpg)

The derivative of a ReLU function is defined as 1 for inputs that are greater than zero and 0 for inputs that are negative. The graph shared below indicates the derivative of a ReLU function

![ReLU Derivative](https://www.kdnuggets.com/wp-content/uploads/jacob-vanishing-gradient-8.png)

If the ReLU function is used for activation in a neural network in place of a sigmoid function, the value of the partial derivative of the loss function will be having values of 0 or 1 which prevents the gradient from vanishing. The use of ReLU function thus prevents the gradient from vanishing. The problem with the use of ReLU is when the gradient has a value of 0. In such cases, the node is considered as a dead node since the old and new values of the weights remain the same. This situation can be avoided by the use of a leaky ReLU function which prevents the gradient from falling to the zero value.

Another technique to avoid the vanishing gradient problem is weight initialization. This is the process of assigning initial values to the weights in the neural network so that during back propagation, the weights never vanish.























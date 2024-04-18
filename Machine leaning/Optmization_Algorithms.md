# Table of Contents

1. [Gradient Descent](#gradient-descent)
   - [How does Gradient Descent work?](#how-gradient-descent-works)
   - [Types of Gradient Descent](#types-gradient-descent)
   - [Challenges with the Gradient Descent](#challenges-gradient-descent)
2. [Gradient Descent](#gradient-descent)
   - [Vanilla Gradient Descent](#vanilla-gradient-descent)
   - [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent)
   - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
   - [Momentum Gradient Descent](#momentum-gradient-descent)
   - [Nesterov Accelerated Gradient (NAG)](#nesterov-accelerated-gradient)
   - [Adagrad](#adagrad)
   - [RMSprop](#rmsprop)
   - [Adam](#adam)


### [krish naik Blog](https://krishnaik.in/2022/03/28/understanding-all-optimizers-in-deep-learning/)
### [A Comprehensive Guide on Optimizers in Deep Learning](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/)
### [Medium article](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6),[Article 2](https://medium.com/mlearning-ai/optimizers-in-deep-learning-7bf81fed78a0)

 
## 1. Gradient Descent <a name="gradient-descent"></a>

Gradient Descent is defined as one of the most commonly used iterative optimization algorithms of machine learning to train the machine learning and deep learning models. It helps in finding the local minimum of a function.

The best way to define the local minimum or local maximum of a function using gradient descent is as follows:

- If we move towards a negative gradient or away from the gradient of the function at the current point, it will give the local minimum of that function.
- Whenever we move towards a positive gradient or towards the gradient of the function at the current point, we will get the local maximum of that function.

The main objective of using a gradient descent algorithm is to minimize the cost function using iteration. To achieve this goal, it performs two steps iteratively:

1. Calculates the first-order derivative of the function to compute the gradient or slope of that function.
2. Move away from the direction of the gradient, which means slope increased from the current point by alpha times, where Alpha is defined as Learning Rate. It is a tuning parameter in the optimization process which helps to decide the length of the steps.

### Articles to know more about Gradient Descent Algorithms

- [Guidelines for selecting Best Opyimizer](https://datascience.stackexchange.com/questions/10523/guidelines-for-selecting-an-optimizer-for-training-neural-networks)
- [Discussion](https://www.quora.com/Why-are-neural-networks-still-being-trained-using-SGD-when-ADAM-has-outperformed-others-training-methods)
- [Overview of all Optimizer](https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008)
- [Is it possible to train a neural network without backpropagation?](https://stats.stackexchange.com/questions/235862/is-it-possible-to-train-a-neural-network-without-backpropagation)
- [Why is it important to include a bias correction term for the Adam optimizer for Deep Learning?](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)
- [What's the effect of scaling a loss function in deep learning for different optimizers?](https://stats.stackexchange.com/questions/346299/whats-the-effect-of-scaling-a-loss-function-in-deep-learning)
- [Guide to learn Learning Rate in Pytorch](https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863)

![Gradient Descent Image](https://static.javatpoint.com/tutorial/machine-learning/images/gradient-descent-in-machine-learning1.png)

### How does Gradient Descent work? <a name="how-gradient-descent-works"></a>

The starting point is used to evaluate the performance as it is considered just as an arbitrary point. At this starting point, we will derive the first derivative or slope and then use a tangent line to calculate the steepness of this slope. Further, this slope will inform the updates to the parameters (weights and bias).

The slope becomes steeper at the starting point or arbitrary point, but whenever new parameters are generated, then steepness gradually reduces, and at the lowest point, it approaches the lowest point, which is called a point of convergence.

The main objective of gradient descent is to minimize the cost function or the error between expected and actual. To minimize the cost function, two data points are required:


Gradient descent is an optimization algorithm used to minimize the cost or loss function of a machine learning model. It iteratively adjusts the model's parameters (weights and biases) in the direction of steepest descent to find the optimal values.

**Here's a simple explanation of gradient descent:**

- Imagine you are at the top of a mountain and want to reach the bottom as quickly as possible.
- You take a step in the steepest downhill direction.
- After each step, you reassess your position and take another step in the new steepest downhill direction.
- You continue this process until you reach the bottom of the mountain, which corresponds to the minimum of the cost or loss function.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/350px-Gradient_descent.svg.png)

In the image above, the mountain represents the cost function landscape, and the arrows indicate the direction of the steepest descent (negative gradient). The goal is to reach the global minimum, which corresponds to the lowest point on the mountain.


### Challenges with the Gradient Descent <a name="challenges-gradient-descent"></a>

1. **Local Minima and Saddle Point and Global Minima**:
For convex problems, gradient descent can find the global minimum easily, while for non-convex problems, it is sometimes difficult to find the global minimum, where the machine learning models achieve the best results.

The point at which a function takes the minimum value is called global minima. However, when the goal is to minimize the function and solved using optimization algorithms such as gradient descent, it may so happen that function may appear to have a minimum value at different points. Those several points which appear to be minima but are not the point where the function actually takes the minimum value are called local minima. Machine learning algorithms such as gradient descent algorithms may get stuck in local minima during the training of the models. Gradient descent is able to find local minima most of the time and not global minima because the gradient does not point in the direction of the steepest descent. Current techniques to find global minima either require extremely high iteration counts or a large number of random restarts for good performance. Global optimization problems can also be quite difficult when high loss barriers exist between local minima.

![Local vs Global Minima](https://vitalflux.com/wp-content/uploads/2020/09/local-minima-vs-global-minima-1.png)

![Local vs Global Minima Animation](https://vitalflux.com/wp-content/uploads/2020/10/local_minima_vs_global_minima.gif)



# Types of Gradient Descent <a name="types-gradient-descent"></a>


## 1. Gradient Descent <a name="gradient-descent"></a>

Gradient Descent is a fundamental optimization algorithm used in machine learning and deep learning to minimize the cost function and update the parameters of the model iteratively.

### 1.1 Vanilla Gradient Descent <a name="vanilla-gradient-descent"></a>

**How it works:**
Vanilla Gradient Descent computes the gradient of the cost function with respect to the model parameters for the entire training dataset. It then updates the parameters by taking a step proportional to the negative of the gradient.

**Advantages:**
- Easy to implement and understand.
- Converges to the global minimum for convex and strictly convex functions.

**Disadvantages:**
- Computationally expensive for large datasets.
- Prone to get stuck in local minima for non-convex functions.

### 11.1. **Batch Gradient Descent (BGD):**
   - It computes the gradient of the cost function with respect to all training examples in the dataset and updates the parameters accordingly. BGD can be computationally expensive for large datasets.
   -  Use Case: Well-suited for small to medium-sized datasets.
   - Disadvantages: Computationally expensive for large datasets.
     
### 1.2 Stochastic Gradient Descent (SGD) <a name="stochastic-gradient-descent"></a>
   -  It randomly selects one training example at a time, computes the gradient based on that example, and updates the parameters. SGD is faster than BGD but has more fluctuation in the convergence.
   - Use Case: Ideal for large datasets and online learning scenarios.
   - Disadvantages: Prone to high variance and slower convergence.

**How it works:**
Stochastic Gradient Descent updates the model parameters for each training example individually. It computes the gradient of the cost function for a single training example and updates the parameters accordingly.

**Advantages:**
- Faster convergence, especially for large datasets.
- Escapes local minima more easily due to frequent updates.

**Disadvantages:**
- Highly sensitive to the learning rate.
- Oscillates around the minimum, which can slow down convergence.

### 1.3 Mini-Batch Gradient Descent <a name="mini-batch-gradient-descent"></a>
   -  It is a compromise between BGD and SGD. It randomly selects a small batch of training examples, computes the gradient based on that batch, and updates the parameters. It strikes a balance between efficiency and stability.
   - Use Case: Provides a balance between BGD and SGD, suitable for medium-sized datasets.
   - Disadvantages: Requires tuning the batch size and can be affected by noise in small batches.
**How it works:**
Mini-Batch Gradient Descent is a compromise between Vanilla GD and SGD. It divides the training dataset into small batches and computes the gradient of the cost function for each batch. It then updates the parameters based on the average gradient of the batch.

**Advantages:**
- More stable convergence compared to SGD.
- Utilizes vectorized operations for efficient computation.

**Disadvantages:**
- Requires tuning of batch size.
- Still sensitive to learning rate.

### 1.4 Momentum Gradient Descent <a name="momentum-gradient-descent"></a>
   - It introduces momentum by adding a fraction of the previous parameter update to the current update. It helps accelerate convergence, especially in the presence of noisy or sparse gradients.
   - Use Case: Effective for accelerating convergence, especially in the presence of noisy or sparse gradients.
   - Disadvantages: Requires tuning the momentum hyperparameter and can overshoot the minimum in certain cases.
**How it works:**
Momentum Gradient Descent adds a momentum term to the parameter update. It accumulates the gradients of past iterations and uses this accumulated gradient to update the parameters, which helps to dampen oscillations and speed up convergence.

**Advantages:**
- Helps to overcome local minima and saddle points.
- Faster convergence compared to Vanilla GD.

**Disadvantages:**
- Requires tuning of momentum parameter.
- May overshoot the minimum in some cases.

### 1.5 Nesterov Accelerated Gradient (NAG) <a name="nesterov-accelerated-gradient"></a>

**How it works:**
Nesterov Accelerated Gradient (NAG) is an improvement over Momentum GD. It computes the gradient of the cost function with respect to the parameters, but instead of using the current parameters for the gradient computation, it uses the parameters updated by the momentum term.

**Advantages:**
- Faster convergence compared to Momentum GD.
- More precise updates, especially in the vicinity of the minimum.

**Disadvantages:**
- Requires tuning of momentum parameter.
- Slightly more computationally expensive than Momentum GD.

### 1.6 Adagrad <a name="adagrad"></a>
 **AdaGrad (Adaptive Gradient Algorithm):**
**How it works:**
Adagrad adapts the learning rate for each parameter based on the historical gradients. It scales down the learning rate for parameters with frequent updates and scales up the learning rate for parameters with infrequent updates.
   -  It adapts the learning rate individually for each parameter based on the historical squared gradients. It provides larger updates for infrequent parameters and smaller updates for frequent ones.
   -  Use Case: Suitable for sparse datasets or problems with varying feature scales.
   -  Disadvantages: The learning rate decays over time, which can lead to premature convergence.
**Advantages:**
- Automatically adjusts learning rates.
- Suitable for sparse data.

**Disadvantages:**
- Learning rate decay can become very small over time, leading to slow convergence.
- Accumulation of squared gradients may lead to numerical instability.


### 1.7 RMSprop <a name="rmsprop"></a>

**How it works:**
RMSprop is an improvement over Adagrad that addresses its drawback of the rapidly decaying learning rate. Instead of accumulating all past squared gradients, RMSprop uses a moving average of squared gradients.

**Advantages:**
- More stable learning rates compared to Adagrad.
- Efficient for training deep neural networks.

**Disadvantages:**
- Requires tuning of hyperparameters.
- May still suffer from slow convergence in some cases.

### 1.8 Adam <a name="adam"></a>

 **Adam (Adaptive Moment Estimation):**
   - It combines the benefits of momentum-based methods and RMSprop. It utilizes both first-moment (mean) and second-moment (uncentered variance) estimates to adaptively adjust the learning rate.
   -  Use Case: Widely used and suitable for various scenarios, especially for deep learning.
   - Disadvantages: Requires tuning multiple hyperparameters and can exhibit high memory usage.
   - 
**How it works:**
Adam (Adaptive Moment Estimation) combines the ideas of Momentum GD and RMSprop. It computes adaptive learning rates for each parameter as well as an exponentially decaying average of past gradients.

**Advantages:**
- Converges quickly and efficiently.
- Suitable for a wide range of optimization problems.

**Disadvantages:**
- Requires tuning of hyperparameters.
- May exhibit erratic behavior in some cases.


![Screenshot_2023-12-10-00-09-06-85_254de13a4bc8758c9908fff1f73e3725](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/e18b8587-6b05-4365-898c-a2be54004557)

# Optimizers in Machine Learning and Deep Learning

This document provides an overview of various optimization algorithms commonly used in machine learning (ML) and deep learning (DL) applications, along with their characteristics, advantages, disadvantages, and recommended use cases.

## Optimizer Table

| Optimizer        | Process                                 | Weight Update                        | Learning Rate                          | Advantages                            | Disadvantages                          | When to Use                                                     |
|------------------|-----------------------------------------|--------------------------------------|----------------------------------------|---------------------------------------|----------------------------------------|-----------------------------------------------------------------|
| Gradient Descent | Iteratively adjusts parameters to minimize the loss function | \( \theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t) \) | Fixed \( \alpha \)                    | Simple, easy to understand and implement | Prone to get stuck in local minima, slow convergence on complex surfaces | When exploring baseline optimization methods or for small-scale problems |
| Stochastic Gradient Descent (SGD) | An extension of GD using random samples (mini-batches) from the dataset | Similar to GD but with random mini-batch gradients | Fixed \( \alpha \)                    | Faster convergence, handles large datasets | Oscillates around the minimum, might converge to a suboptimal solution | Large datasets, general-purpose optimization |
| Mini-Batch Gradient Descent | Balances efficiency using mini-batches for parameter updates | Similar to SGD but uses mini-batches | Fixed or adaptive with schedules      | Balance between efficiency and convergence | Need to tune batch size, learning rate, might get stuck in saddle points | Commonly used in DL due to better convergence than full-batch GD |
| Adam (Adaptive Moment Estimation) | Combines RMSprop and Momentum methods, uses adaptive learning rates | Adjusts parameters based on past gradients | Adaptive learning rates per parameter  | Fast convergence, adaptive learning rates | Requires tuning of hyperparameters, memory-intensive | Widely used in DL for various architectures and datasets |
| RMSprop          | Adaptive learning rate method             | Utilizes moving average of squared gradients | Adaptive learning rate per parameter   | Effective in non-stationary environments | Hyperparameter sensitivity, may perform poorly in some cases | Suitable for RNNs and LSTMs in sequential data tasks |
| Adagrad          | Adapts learning rates based on parameter frequencies | Divides learning rate by the root sum of squared gradients | Automatically reduces learning rates   | Well-suited for sparse data, reduces learning rates adaptively | Accumulation of squared gradients causes diminishing updates | Sparse data tasks, NLP problems with sparse features |
| AdaDelta         | Extension of Adagrad addressing diminishing learning rates | Uses a more stable update rule       | Adaptive update rule without explicit learning rate | No need for manual tuning of learning rate, more stable | Computationally intensive due to maintaining parameter state | Suitable for large-scale problems with sparse data |
| Adamax           | Variant of Adam based on infinity norm    | Adjusts parameters based on norm of gradients | Adaptive learning rate, less memory intensive | Stable behavior, less memory intensive | May converge to suboptimal solutions in some cases | Suitable for problems in DL with sparse gradients |
| Nadam (Nesterov-accelerated Adam) | Adam with Nesterov momentum              | Similar to Adam but with Nesterov term | Adaptive learning rates based on Adam  | Faster convergence, less sensitive to learning rates | Requires fine-tuning of hyperparameters | Commonly used in DL for various architectures and datasets |

This table summarizes the key aspects of various optimization algorithms used in ML and DL, providing insights into their processes, advantages, disadvantages, and suitable scenarios for their application.


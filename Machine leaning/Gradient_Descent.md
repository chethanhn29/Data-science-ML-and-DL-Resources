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

## 1. Gradient Descent <a name="gradient-descent"></a>

Gradient Descent is defined as one of the most commonly used iterative optimization algorithms of machine learning to train the machine learning and deep learning models. It helps in finding the local minimum of a function.

The best way to define the local minimum or local maximum of a function using gradient descent is as follows:

- If we move towards a negative gradient or away from the gradient of the function at the current point, it will give the local minimum of that function.
- Whenever we move towards a positive gradient or towards the gradient of the function at the current point, we will get the local maximum of that function.

The main objective of using a gradient descent algorithm is to minimize the cost function using iteration. To achieve this goal, it performs two steps iteratively:

1. Calculates the first-order derivative of the function to compute the gradient or slope of that function.
2. Move away from the direction of the gradient, which means slope increased from the current point by alpha times, where Alpha is defined as Learning Rate. It is a tuning parameter in the optimization process which helps to decide the length of the steps.

![Gradient Descent Image](https://static.javatpoint.com/tutorial/machine-learning/images/gradient-descent-in-machine-learning1.png)

### How does Gradient Descent work? <a name="how-gradient-descent-works"></a>

The starting point is used to evaluate the performance as it is considered just as an arbitrary point. At this starting point, we will derive the first derivative or slope and then use a tangent line to calculate the steepness of this slope. Further, this slope will inform the updates to the parameters (weights and bias).

The slope becomes steeper at the starting point or arbitrary point, but whenever new parameters are generated, then steepness gradually reduces, and at the lowest point, it approaches the lowest point, which is called a point of convergence.

The main objective of gradient descent is to minimize the cost function or the error between expected and actual. To minimize the cost function, two data points are required:




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

### 1.2 Stochastic Gradient Descent (SGD) <a name="stochastic-gradient-descent"></a>

**How it works:**
Stochastic Gradient Descent updates the model parameters for each training example individually. It computes the gradient of the cost function for a single training example and updates the parameters accordingly.

**Advantages:**
- Faster convergence, especially for large datasets.
- Escapes local minima more easily due to frequent updates.

**Disadvantages:**
- Highly sensitive to the learning rate.
- Oscillates around the minimum, which can slow down convergence.

### 1.3 Mini-Batch Gradient Descent <a name="mini-batch-gradient-descent"></a>

**How it works:**
Mini-Batch Gradient Descent is a compromise between Vanilla GD and SGD. It divides the training dataset into small batches and computes the gradient of the cost function for each batch. It then updates the parameters based on the average gradient of the batch.

**Advantages:**
- More stable convergence compared to SGD.
- Utilizes vectorized operations for efficient computation.

**Disadvantages:**
- Requires tuning of batch size.
- Still sensitive to learning rate.

### 1.4 Momentum Gradient Descent <a name="momentum-gradient-descent"></a>

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

**How it works:**
Adagrad adapts the learning rate for each parameter based on the historical gradients. It scales down the learning rate for parameters with frequent updates and scales up the learning rate for parameters with infrequent updates.

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

**How it works:**
Adam (Adaptive Moment Estimation) combines the ideas of Momentum GD and RMSprop. It computes adaptive learning rates for each parameter as well as an exponentially decaying average of past gradients.

**Advantages:**
- Converges quickly and efficiently.
- Suitable for a wide range of optimization problems.

**Disadvantages:**
- Requires tuning of hyperparameters.
- May exhibit erratic behavior in some cases.

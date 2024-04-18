# Table of Contents

1. [Gradient Descent](#gradient-descent)
   - [How does Gradient Descent work?](#how-gradient-descent-works)
   - [Types of Gradient Descent](#types-gradient-descent)
   - [Challenges with the Gradient Descent](#challenges-gradient-descent)

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

### Types of Gradient Descent <a name="types-gradient-descent"></a>

Based on the error in various training models, the Gradient Descent learning algorithm can be divided into Batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

### Challenges with the Gradient Descent <a name="challenges-gradient-descent"></a>

1. **Local Minima and Saddle Point and Global Minima**:
For convex problems, gradient descent can find the global minimum easily, while for non-convex problems, it is sometimes difficult to find the global minimum, where the machine learning models achieve the best results.

The point at which a function takes the minimum value is called global minima. However, when the goal is to minimize the function and solved using optimization algorithms such as gradient descent, it may so happen that function may appear to have a minimum value at different points. Those several points which appear to be minima but are not the point where the function actually takes the minimum value are called local minima. Machine learning algorithms such as gradient descent algorithms may get stuck in local minima during the training of the models. Gradient descent is able to find local minima most of the time and not global minima because the gradient does not point in the direction of the steepest descent. Current techniques to find global minima either require extremely high iteration counts or a large number of random restarts for good performance. Global optimization problems can also be quite difficult when high loss barriers exist between local minima.

![Local vs Global Minima](https://vitalflux.com/wp-content/uploads/2020/09/local-minima-vs-global-minima-1.png)

![Local vs Global Minima Animation](https://vitalflux.com/wp-content/uploads/2020/10/local_minima_vs_global_minima.gif)

---

With everything organized and a table of contents added, it should be easier for readers to navigate through the document.

## Gradient Descent Algorithms

### [krish naik Blog](https://krishnaik.in/2022/03/28/understanding-all-optimizers-in-deep-learning/)
### [A Comprehensive Guide on Optimizers in Deep Learning](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/)
### [Medium article](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6),[Article 2](https://medium.com/mlearning-ai/optimizers-in-deep-learning-7bf81fed78a0)

Gradient descent is an optimization algorithm used to minimize the cost or loss function of a machine learning model. It iteratively adjusts the model's parameters (weights and biases) in the direction of steepest descent to find the optimal values.

Here's a simple explanation of gradient descent:

Imagine you are at the top of a mountain and want to reach the bottom as quickly as possible.
You take a step in the steepest downhill direction.
After each step, you reassess your position and take another step in the new steepest downhill direction.
You continue this process until you reach the bottom of the mountain, which corresponds to the minimum of the cost or loss function.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/350px-Gradient_descent.svg.png)

In the image above, the mountain represents the cost function landscape, and the arrows indicate the direction of the steepest descent (negative gradient). The goal is to reach the global minimum, which corresponds to the lowest point on the mountain.

### Types of Majorly Used Gradient Descent Algorithms

- [Guidelines for selecting Best Opyimizer](https://datascience.stackexchange.com/questions/10523/guidelines-for-selecting-an-optimizer-for-training-neural-networks)
- [Discussion](https://www.quora.com/Why-are-neural-networks-still-being-trained-using-SGD-when-ADAM-has-outperformed-others-training-methods)
- [Overview of all Optimizer](https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008)
- [Is it possible to train a neural network without backpropagation?](https://stats.stackexchange.com/questions/235862/is-it-possible-to-train-a-neural-network-without-backpropagation)
- [Why is it important to include a bias correction term for the Adam optimizer for Deep Learning?](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for)
- [What's the effect of scaling a loss function in deep learning for different optimizers?](https://stats.stackexchange.com/questions/346299/whats-the-effect-of-scaling-a-loss-function-in-deep-learning)
- [Guide to learn Learning Rate in Pytorch](https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863)


1. **Batch Gradient Descent (BGD):**
   - It computes the gradient of the cost function with respect to all training examples in the dataset and updates the parameters accordingly. BGD can be computationally expensive for large datasets.
   -  Use Case: Well-suited for small to medium-sized datasets.
   - Disadvantages: Computationally expensive for large datasets.


2. **Stochastic Gradient Descent (SGD):**
   -  It randomly selects one training example at a time, computes the gradient based on that example, and updates the parameters. SGD is faster than BGD but has more fluctuation in the convergence.
   - Use Case: Ideal for large datasets and online learning scenarios.
   - Disadvantages: Prone to high variance and slower convergence.

3. **Mini-Batch Gradient Descent:**
   -  It is a compromise between BGD and SGD. It randomly selects a small batch of training examples, computes the gradient based on that batch, and updates the parameters. It strikes a balance between efficiency and stability.
   - Use Case: Provides a balance between BGD and SGD, suitable for medium-sized datasets.
   - Disadvantages: Requires tuning the batch size and can be affected by noise in small batches.

4. **Momentum-based Gradient Descent:**
   - It introduces momentum by adding a fraction of the previous parameter update to the current update. It helps accelerate convergence, especially in the presence of noisy or sparse gradients.
   - Use Case: Effective for accelerating convergence, especially in the presence of noisy or sparse gradients.
   - Disadvantages: Requires tuning the momentum hyperparameter and can overshoot the minimum in certain cases.

5. **Nesterov Accelerated Gradient (NAG):**
   - It improves upon momentum-based gradient descent by adjusting the gradient calculation. NAG calculates the gradient not at the current parameters but at the expected future parameters, resulting in faster convergence.
   - Use Case: Improves upon momentum-based gradient descent, especially for optimizing deep neural networks.
   - Disadvantages: Sensitive to the choice of the learning rate and may not significantly outperform other methods in all cases.

6. **AdaGrad (Adaptive Gradient Algorithm):**
   -  It adapts the learning rate individually for each parameter based on the historical squared gradients. It provides larger updates for infrequent parameters and smaller updates for frequent ones.
   -  Use Case: Suitable for sparse datasets or problems with varying feature scales.
   -  Disadvantages: The learning rate decays over time, which can lead to premature convergence.

7. **RMSprop (Root Mean Square Propagation):**
   -  It addresses the diminishing learning rate issue of AdaGrad by introducing a decay factor. It maintains a moving average of squared gradients to adjust the learning rate adaptively.
   - Use Case: Effective for dealing with the diminishing learning rate issue of AdaGrad.
   - Disadvantages: Requires tuning the learning rate and can accumulate gradients too quickly in deep networks.

8. **Adam (Adaptive Moment Estimation):**
   - It combines the benefits of momentum-based methods and RMSprop. It utilizes both first-moment (mean) and second-moment (uncentered variance) estimates to adaptively adjust the learning rate.
   -  Use Case: Widely used and suitable for various scenarios, especially for deep learning.
   - Disadvantages: Requires tuning multiple hyperparameters and can exhibit high memory usage.

9. **AdaDelta (Adaptive Delta):**
   - Use Case: Addresses the accumulating squared gradients issue of AdaGrad.
   - Disadvantages: Requires tuning hyperparameters and can exhibit slow convergence.
   -  It extends the idea of AdaGrad and RMSprop by addressing the accumulating squared gradients issue. It uses a running average of squared gradients to adaptively adjust the learning rate.


10. **Adamax:**
    - Use Case: Provides better stability for models with sparse gradients, often used in deep learning.
    - Disadvantages: May require tuning the hyperparameters for optimal performance.
    - Adamax: It is a variant of Adam that replaces the second-moment estimation with the infinity norm of the gradients. It provides better stability for models with sparse gradients.


11. **Nadam (Nesterov-accelerated Adaptive Moment Estimation):**
    - Use Case: Combines NAG with the adaptive learning rates of Adam, often used in deep learning.
    - Disadvantages: Requires tuning hyperparameters and can exhibit slow convergence for some problems.
    - Nadam (Nesterov-accelerated Adaptive Moment Estimation): It combines NAG with Adam. It utilizes the NAG update rule along with the adaptive learning rates of Adam.


12. **Adagrad-Dense:**
    - Use Case: Improves AdaGrad's performance on dense datasets.
    - Disadvantages: May not provide significant benefits on sparse datasets and requires tuning hyperparameters.
    - Adagrad-Dense: It is an extension of AdaGrad that improves its performance on dense datasets by incorporating a bias correction term.


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


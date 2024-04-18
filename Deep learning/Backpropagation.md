## [Back Propagation Vs Gradient Descent](https://www.analyticsvidhya.com/blog/2023/01/gradient-descent-vs-backpropagation-whats-the-difference/#:~:text=To%20put%20it%20plainly%2C%20gradient,direction%20in%20the%20neural%20network.)
## [Back propagation](https://www.analyticsvidhya.com/blog/2022/01/introduction-to-the-neural-network-model-glossary-and-backpropagation/)


# Personal-Collection-of-Resources-to-learn Back Propagation

for Maths behind Back Propagation See at the end of this File and [Watch](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) 

- [Stanford Notes for Neural Networks](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Neuron/index.html) ,
- [Andrew Ng Notes for Neural Networks](https://github.com/ashishpatel26/Andrew-NG-Notes), 
- Deep Understanding Of Neural Networks,Feed Forward,Gradient Descent,Backpropagation and Maths Behind it [Watch](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

**To learn more, I highly recommend the book by Michael Nielsen**
- [http://neuralnetworksanddeeplearning....](http://neuralnetworksanddeeplearning.com/)The book walks through the code behind the example in these videos, which you can find here: 
- [https://github.com/mnielsen/neural-ne...](https://github.com/mnielsen/neural-networks-and-deep-learning)
- Also check out Chris Olah's blog: [http://colah.github.io/](http://colah.github.io/)His post on Neural networks and topology is particular beautiful, but honestly all of the stuff there is great.
- And if you like that, you'll love the publications at distill:[https://distill.pub/](https://distill.pub/)

## Mathematics

1. [Backward Propagation](https://stackoverflow.com/questions/9023404/how-does-a-back-propagation-training-algorithm-work),
2. Deep Dive Into Back Propagation [Part1](https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=11s),[Part2](https://www.youtube.com/watch?v=tIeHLnjs5U8)
3. [Gradient Descent](https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/)
4. [ Back Propagation vs Gradient Descent](https://www.analyticsvidhya.com/blog/2023/01/gradient-descent-vs-backpropagation-whats-the-difference/)
5. [Optimisation Techniques to Reduce Loss Function](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/),


##  Choose a neural network architecture.
See this Articles to Know more
-  [Keras input explanation: input_shape, units, batch_size, dim, etc](https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc)  
-  [How to determine input shape in keras?](https://datascience.stackexchange.com/questions/53609/how-to-determine-input-shape-in-keras)
-  [Determining The Right Batch size for Neural Networks](https://medium.com/data-science-365/determining-the-right-batch-size-for-a-neural-network-to-get-better-and-faster-results-7a8662830f15)
-  [Guide to Learning Rate Schedulers in PyTorch](https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863)
-  [To Improve the accuracy of model](https://stackoverflow.com/questions/59278771/super-low-accuracy-for-neural-network-model)
-  [DropOut Technique to Prevent Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
-  [What does the hidden layer in a neural network compute?](https://stats.stackexchange.com/questions/63152/what-does-the-hidden-layer-in-a-neural-network-compute?rq=1)
-  [How to get weights Parameters in Tensorflow](https://www.gcptutorials.com/post/how-to-get-weights-of-layers-in-tensorflow)
-  [To get Weights and Bias from model](https://stackoverflow.com/questions/56855107/how-do-i-get-weights-and-biases-from-my-model)
-  If you want to get weights and biases of all layers, you can simply use [Link](https://stackoverflow.com/questions/43715047/how-do-i-get-the-weights-of-a-layer-in-keras?rq=3):
```python
	for layer in model.layers: print(layer.get_config(), layer.get_weights())
```
	This will print all information that's relevant.

	If you want the weights directly returned as numpy arrays, you can use:
```python
	first_layer_weights = model.layers[0].get_weights()[0]
	first_layer_biases  = model.layers[0].get_weights()[1]
	second_layer_weights = model.layers[1].get_weights()[0]
	second_layer_biases  = model.layers[1].get_weights()[1]
````
## Number of Bias in Neural Networks 

![](https://miro.medium.com/v2/resize:fit:640/0*rVZEEOvKGHsbxIFw.)

- Each node in the hidden layers or in the output layer of a feed-forward neural network has its own bias term.
-  (The input layer has no parameters whatsoever.) At least, that's how it works in TensorFlow. To be sure, I constructed your two neural networks in TensorFlow as follows:
```Python
model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation = 'softmax', input_shape = (5,))])

model2 =  tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation = 'relu', input_shape = (5,)),
        tf.keras.layers.Dense(3,activation = 'softmax')])
```
**Here is the summary of these two models that TensorFlow provides:
**
![](https://i.stack.imgur.com/KpqkC.png)

**In model1**	
	- We can see that in the First model1 , it has Input shape of data is 5 and for 1 layer 1 Bias term so total  6  Parameters in the Input layer 
	- In the Second Dense layer we can see that it has 4 Nodes so  Total Prameters in model1=(6 * 4)=24 Parameters as same as in the model1 Summary.
	
**In model2**
	- The second model has 24 parameters in the hidden layer (counted the same way as above) In Between The input layer and first Dense Layer
	-  Now 1st Dense Layer act as Input layer and it has 4 Nodes( 1 Weights per each Node) and 1 Bias term should be add per each layer so total 5 Nodes in 	   1st dense layer====>>>> Total Nodes in 1st Dense Layer now=5 Nodes Or  Parameters(Including Bias).
	- The 2nd Dense Layer has 3 Nodes  so, from the first Dense Layer 5 Parameters will transfer into 3 nodes of there are 3 nodes in the output layer. so 		 total  15 parameters in the output layer. Bias terst 2nd Dense Layer So Total Parameters in 2nd Layer is (5 * 3)=15 Parameters
	- Total Parameters=15+24=39 Parametersnd 

##### Following are 7 key steps for training a neural network.

- Pick a neural network architecture. This implies that you shall be pondering primarily upon the connectivity patterns of the neural network including some of the following aspects:
- Number of input nodes: The way to identify number of input nodes is identify the number of features.
- Number of hidden layers: The default is to use the single or one hidden layer. This is the most common practice.
- Number of nodes in each of the hidden layers: In case of using multiple hidden layers, the best practice is to use same number of nodes in each hidden layer. In general practice, the number of hidden units is taken as comparable number to that of number of input nodes. That means one could take either the same number of hidden nodes as input nodes or maybe twice or thrice the number of input nodes.
- Number of output nodes: The way to identify number of output nodes is to identify the number of output classes you want the neural network to process.

- Random Initialization of Weights: The weights are randomly intialized to value in between 0 and 1, or rather, very close to zero.
- Implementation of forward propagation algorithm to calculate hypothesis function for a set on input vector for any of the hidden layer.
- Implementation of cost function for optimizing parameter values. One may recall that cost function would help determine how well the neural network fits the training data.
- Implementation of back propagation algorithm to compute the error vector related with each of the nodes.
-Use gradient checking method to compare the gradient calculated using partial derivatives of cost function using back propagation and using numerical estimate of cost function gradient. The gradient checking method is used to validate if the implementation of backpropagation method is correct.
- Use gradient descent or advanced optimization technique with back propagation to try and minimize the cost function as a function of parameters or weights.
- 
- **Backpropagation** is an algorithm that calculates the gradient of a loss function with respect to the parameters of a neural network. The gradient is a measure of how much the loss function will change if the parameters are changed. Backpropagation is used to find the optimal parameters for a neural network by iteratively updating the parameters in the direction of the negative gradient.
- **Gradient descent** is an optimization algorithm that uses the gradient of a function to find the minimum of the function. Gradient descent is used in neural networks to find the optimal parameters by iteratively updating the parameters in the direction of the negative gradient.
- **Optimization algorithms** are a broad class of algorithms that are used to find the optimal solution to a problem. There are many different optimization algorithms, and the best algorithm to use depends on the specific problem. In the context of neural networks, some common optimization algorithms include stochastic gradient descent (SGD), mini-batch SGD, and Adam.
The order in which these concepts are used when training neural networks is as follows:

	- First, the neural network is initialized with random parameters.
	- Next, a training dataset is used to calculate the loss function.
	- The gradient of the loss function is calculated using backpropagation.
	- The parameters of the neural network are updated using an optimization algorithm.
	- Steps 2-4 are repeated until the loss function converges to a minimum value.
  
  ###  Maths behind Backward Propagation 
  
![IMG20230531092912 (1)](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/7a962fc1-f1ae-4454-a118-bbd11dc6b2f7)


![2 (1)](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/7279d6a4-8c89-43bd-8c6a-bc9c4aff4156)

![3 (1)](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/42f251ad-1d64-4a85-a403-3be6592f51ac)

![IMG20230531095239](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/9c2a8f6f-76e6-4898-afe6-e41f5a1a08c6)


![5 (2)](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/1ed747f6-e036-4930-a515-b206bef3e763)

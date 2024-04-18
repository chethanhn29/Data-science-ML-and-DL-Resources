# Personal Collection of Resources to Learn Deep Learning

## Best Deep Learning Interview Preparation Videos
- [Deep learning Interview Prep Video by Freecodecamp](https://www.youtube.com/watch?v=BAregq0sdyY) - Highly recommended for interview preparation.
- [DS Interview Questions](https://www.youtube.com/watch?v=dBvjBwga8pU) - Useful for Data Science interview preparation.

## Quick Revision Resources
- [Aman.ai Primer for AI](https://aman.ai/primers/ai/) - Comprehensive resource for quick revision covering various AI topics.

## Stanford University Resources
- [Stanford Notes on Neural Networks](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Neuron/index.html) - Excellent resource for understanding neural networks.
- [Dive into Deep Learning Book by Stanford University](https://c.d2l.ai/gtc2020/) - Comprehensive book for deep learning.
- [Dive into Deep Learning in 1 Day by Stanford University](https://c.d2l.ai/odsc2019/) - Condensed version of deep learning concepts in a single day.
- [Stanford University Courses Books](https://courses.d2l.ai/) - Collection of books for various Stanford courses.
- [Deep Learning Practical by Uvadlc](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html) - Practical tutorials for deep learning.

## Deep Learning Study Materials
- [Dive Deep Into Deep learning Book By Stanford](https://d2l.ai/d2l-en.pdf) - In-depth book on deep learning concepts.
- [Deep Learning.ai Summary](https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master) - Summarized notes from DeepLearning.ai courses.

## Additional Learning Resources
- [Andrew Ng Notes](https://github.com/chethanhn29/Andrew-NG-Notes) - Detailed notes on neural networks, feedforward, gradient descent, and backpropagation by Andrew Ng.
- [ML Notes by dair-ai](https://github.com/dair-ai/ML-Course-Notes) - Machine Learning course notes.
- [Machine and Deep Learning Course](https://github.com/chethanhn29/ML-DL-Notes) - Course materials covering machine and deep learning.

## Supplementary Materials and References
- Book by Michael Nielsen: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Recommended for a deeper understanding of neural networks.
- Chris Olah's Blog: [Colah's Blog](http://colah.github.io/) - Particularly beautiful post on neural networks and topology.
- [Distill Publications](https://distill.pub/) - Offers high-quality publications on various deep learning topics.


## Deep learning
1. What is Neuron
2. What is Perceptron
3. Feature Scaling 
	- [Scaling to imporve Model Stability](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)
)
4. Split the Data Train,Validation,Test Data 
5. Input ,Hidden,Output Layer
6. [Choose a neural network architecture](#choose)
	-  preprocessing layer With input layer
	-  [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html),[paper](https://arxiv.org/pdf/1502.01852.pdf)
	-  Input Layer(input shape and its parameters) 
	-  Hidden Layers
		- Number of Hidden Layers
		- Number of Nodes in Each layer
		- Activation Function
		- Kernel intializer in each Hidden Layer
		- To add bias or Notin each hidden layer
	-  Dropout Technique,[how to apply](https://stackoverflow.com/questions/40879504/how-to-apply-drop-out-in-tensorflow-to-improve-the-accuracy-of-neural-network?rq=2)
	-  Output layers
		- Nodes of output Layer
		- Acticvation Funtion
		- Probability Function to get the Probabilities of each Class and By using Argmax() or Softmax function we can get the 			  high probablity Class.
	-  Model Summary
	- Model Compiling 
		- optimizer,[Parameter optimization in neural networks](https://www.deeplearning.ai/ai-notes/optimization/index.html)
			- [optimizers.schedules](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules),[ML mastery](https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/),[Book](https://d2l.ai/chapter_optimization/lr-scheduler.html)
				- Constant learning rate: This is the simplest type of optimizer schedule, and it simply keeps the 					  learning rate constant throughout training. This can be a good choice for simple models, but it can lead 				      to overfitting for more complex models.
				- Decaying learning rate: This type of optimizer schedule starts with a high learning rate and then 					  decreases it over time. This can help the model to converge more quickly and to avoid overfitting.
				- Warming up: This type of optimizer schedule starts with a very low learning rate and then gradually 					 increases it over time. This can help the model to avoid getting stuck in local minima.
		- lr_schedule 
			- initial_learning_rate: The initial learning rate.
			- decay_steps: The number of steps after which the learning rate will be decayed.
			- decay_rate: The rate at which the learning rate will be decayed.
			- staircase: A boolean value that determines whether the learning rate should be decayed in steps or smoothly.
			- [DON’T DECAY THE LEARNING RATE,INCREASE THE BATCH SIZE](https://arxiv.org/pdf/1711.00489.pdf)
			- [closing thegeneralization gap in large batch training of neuralnetworks](https://proceedings.neurips.cc/paper_files/paper/2017/file/a5e0ff62be0b08456fc7f1e88812af3d-Paper.pdf)
		- loss Funtion
	- Fit the Data to the Model
		- Epcochs
		- Verbose
		- time
		- Get the Accuray of Training data and Validation data , in that way decide The model has more bias or more vairance or 		has both high Bias and more variance In the That Way we get the idea of Model is Overfitting or Underfitting or Right 			Model.
	- Evaluate the Model for validation and Test data and get the Accuracy of model
	- Get the Hostory Keys like Val_loss,accuracy,epoch from the history object of model and plot all visulaisations 
	- and plot the Predicted output vs Actual Output to get the view of how model is fitting the data
7. [Parameters affect the model Architecture](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/tree/main/Neural%20Network%20Architecture)
8. Forward Propagation-feed forward Neural network
9. Weights and Bias
10. why do we need weights and Bias
11. Weight and bias intialisation and its types
12. Summation funtion for Wights and  Bias
13. Train the Neural Network
14. Activation Function in Hidden Layers,
[why Activation Function has Limited range](https://datascience.stackexchange.com/questions/62881/why-activation-functions-used-in-neural-networks-generally-have-limited-range?rq=1),
[Why ReLu](https://datascience.stackexchange.com/questions/90114/why-is-the-dying-relu-problem-not-present-in-most-modern-deep-learning-archite?rq=1),
[Dyling Relu](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks?noredirect=1&lq=1),
[ReLu Effectiveness](https://towardsdatascience.com/relu-rules-lets-understand-why-its-popularity-remains-unshaken-ccfe952fc5b1),
[ReLu](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) ,
[SELU](https://towardsdatascience.com/gentle-introduction-to-selus-b19943068cd9),
[ReLu vs SELU](https://datascience.stackexchange.com/questions/102724/why-deep-learning-models-still-use-relu-instead-of-selu-as-their-activation-fun?rq=1)
15. Vanishing Gradient Problem,[Krish Explanation](https://www.youtube.com/watch?v=JIWXbzRXk1I&list=PLZoTAELRMXVPGU70ZGsckrMdr0FteeRUi&index=10),[Blog](https://www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/)
16. Activation Function in Output Layer,
[Activation fun In Regression](https://datascience.stackexchange.com/questions/47751/what-activation-function-should-i-use-for-a-specific-regression-problem).
18. Evaluate the Neural network Using Loss Function 
19. [Backward Propagation](https://stackoverflow.com/questions/9023404/how-does-a-back-propagation-training-algorithm-work),Deep Dive Into Back Propagation [Part1](https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=11s),[Part2](https://www.youtube.com/watch?v=tIeHLnjs5U8)
20. [Gradient Descent](https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/),
[ Back Propagation vs Gradient Descent](https://www.analyticsvidhya.com/blog/2023/01/gradient-descent-vs-backpropagation-whats-the-difference/)
22. [Optimisation Techniques to Reduce Loss Function](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/),
[Overview of Optimisation Methods](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/tree/main/Optimization%20Algorithms%20or%20Methods)
23.[Use Weight Regularization to Reduce Overfitting of Deep Learning Models](https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/),[from tensorflow](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
24. [Hyperparameter Tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)
	-	[In Machine learning](https://www.javatpoint.com/hyperparameters-in-machine-learning#:~:text=Hyperparameters%20in%20Machine%20learning%20are,learning%20process%20of%20the%20model.),
[[kaggle](https://www.kaggle.com/code/faressayah/hyperparameter-optimization-for-machine-learning)],
[[1](https://www.analyticsvidhya.com/blog/2022/02/a-comprehensive-guide-on-hyperparameter-tuning-and-its-techniques/)],
[[2](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/)],
[[3](https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/)]
	-	[In Deep learning](https://www.analyticsvidhya.com/blog/2021/05/tuning-the-hyperparameters-and-layers-of-neural-network-deep-learning/)
[[1](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)]
[[For Pytorch Models](https://machinelearningmastery.com/how-to-grid-search-hyperparameters-for-pytorch-models/)]
22. Get the Performance of the model using Evlualtion Metrics
23. validation dataset,Cross Validation

## Articles TO know More about machine learning and Deep learning
 - [What is the relationship between the accuracy and the loss in deep learning?](https://stats.stackexchange.com/questions/365778/what-should-i-do-when-my-neural-network-doesnt-generalize-well/365806#365806)
 - [Use Weight Regularization to Reduce Overfitting of Deep Learning Models](https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/)
 - Regularization does NOT minimize loss functions. Regularization usually increases loss function, but often offers better generalization.
 - [Why is accuracy not the best measure for assessing classification models?](https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models)
 - [What should I do when my neural network doesn't generalize well?](https://stats.stackexchange.com/questions/365778/what-should-i-do-when-my-neural-network-doesnt-generalize-well/365806#365806)
 - [Dropout Published paper](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer)
 - [What should I do when my neural network doesn't learn?](https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn)
 - [Training loss increases with time [duplicate]](https://stats.stackexchange.com/questions/324896/training-loss-increases-with-time?noredirect=1&lq=1)
 - 
## 4. Choose a neural network architecture.
See this Articles to Know more
-  [Keras input explanation: input_shape, units, batch_size, dim, etc](https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc)  
-  [How to determine input shape in keras?](https://datascience.stackexchange.com/questions/53609/how-to-determine-input-shape-in-keras)
-  [Determining The Right Batch size for Neural Networks](https://medium.com/data-science-365/determining-the-right-batch-size-for-a-neural-network-to-get-better-and-faster-results-7a8662830f15)
-  [Guide to Learning Rate Schedulers in PyTorch](https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863)
-  [To Improve the accuracy of model](https://stackoverflow.com/questions/59278771/super-low-accuracy-for-neural-network-model)
-  [DropOut Technique to Prevent Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
-  [What does the hidden layer in a neural network compute?](https://stats.stackexchange.com/questions/63152/what-does-the-hidden-layer-in-a-neural-network-compute?rq=1)
-  [ How to Calculate the Number of Weights and Bias in Neural Network ](https://stats.stackexchange.com/questions/296981/formula-for-number-of-weights-in-neural-network#:~:text=You%20can%20find%20the%20number,layers%20and%20the%20output%20layer.)
-  -  If you want to get weights and biases of all layers, you can simply use [Link](https://stackoverflow.com/questions/43715047/how-do-i-get-the-weights-of-a-layer-in-keras?rq=3):
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


### Number of Hidden Layers
- [Paper to Choose Hidden Layers](http://ijettjournal.org/volume-3/issue-6/IJETT-V3I6P206.pdf)
- [criteria to choose No  and Size of Hidden layers](https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde/10568938#10568938), and refer [This article](https://stackoverflow.com/questions/9436209/how-to-choose-number-of-hidden-layers-and-nodes-in-neural-network?noredirect=1&lq=1)
-  [No of Hidden layers](https://stackoverflow.com/questions/35520587/how-to-determine-the-number-of-layers-and-nodes-of-a-neural-network?rq=1) and Refer [This Article](https://stackoverflow.com/questions/3345079/estimating-the-number-of-neurons-and-number-of-layers-of-an-artificial-neural-ne?noredirect=1&lq=1)
-  [How to get weights Parameters in Tensorflow](https://www.gcptutorials.com/post/how-to-get-weights-of-layers-in-tensorflow)
-  [To get Weights and Bias from model](https://stackoverflow.com/questions/56855107/how-do-i-get-weights-and-biases-from-my-model)

There is no one-size-fits-all answer to this question, as the number of neurons and number of hidden layers required for a neural network will vary depending on the specific problem being solved and the size and complexity of the dataset. However, there are a few general guidelines that can be followed:

- The number of neurons in the input layer should be equal to the number of features in the dataset. This is because each neuron in the input layer represents a single feature, and the input layer is responsible for taking the data and converting it into a form that can be processed by the rest of the neural network.
- The number of neurons in the hidden layer(s) should be between the number of neurons in the input layer and the number of neurons in the output layer. The number of neurons in the hidden layer(s) can be determined by experimenting with different values and seeing what works best for the specific problem.
- The number of hidden layers should be kept to a minimum. Adding more hidden layers can improve the performance of the neural network, but it can also make the neural network more difficult to train.

**In general, it is best to start with a small number of neurons and a small number of hidden layers, and then experiment with different values to see what works best for the specific problem.**

Here are some additional tips for choosing the number of neurons and number of hidden layers:

	- Consider the complexity of the problem being solved. More complex problems will require more neurons and more hidden layers.
	- Consider the size of the dataset. Larger datasets will require more neurons and more hidden layers.
	- Use a validation set to evaluate the performance of the neural network. The validation set should not be used to train the 		  neural network, but it should be used to evaluate the performance of the neural network after each epoch.
	- Use a cross-validation set to select the best hyperparameters. The cross-validation set should not be used to train the neural 	   network, but it should be used to select the best hyperparameters, such as the number of neurons and number of hidden layers.
	
By following these guidelines, you can choose the number of neurons and number of hidden layers that will give you the best performance for your neural network.

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

### Flatten Layer:
- Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. or The purpose of a flatten layer is to convert a multi-dimensional tensor into a one-dimensional tensor.  
- **This is useful in situations where the next layer in the neural network expects a one-dimensional input. For example, a dense layer expects a one-dimensional input, so a flatten layer is often used to convert the output of a convolutional layer into a format that can be fed into a dense layer.**
- We flatten the output of the convolutional layers to create a single long feature vector. 
- In some architectures, e.g. CNN an image is better processed by a neural network if it is in 1D form rather than 2D.
**- Here are some situations where you might use a flatten layer:
**
	- After a convolutional layer
	- After a pooling layer
	- Before a dense layer**
	
![](https://cdn.discuss.boardinfinity.com/optimized/2X/1/1f1bf9539699c880b33f978e724f803ef8197f6f_2_690x324.png)
	
```Python3
from keras.layers import Flatten

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
```
![](https://i.stack.imgur.com/rx3X4.png)

```Python3
model.summary()
Layer (type)                     Output Shape          Param #
================================================================
vgg16 (Model)                    (None, 4, 4, 512)     14714688
________________________________________________________________
flatten_1 (Flatten)              (None, 8192)          0
________________________________________________________________
dense_1 (Dense)                  (None, 256)           2097408
________________________________________________________________
dense_2 (Dense)                  (None, 1)             257
```
##### Note: Here we can see the effect of flatten layer , which flattens the output shape of(4*4*512) to one dimesion array (8192)

### What is Dense Layer 
- Dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer, thus called as dense
- A dense layer is a type of neural network layer that connects all of the output units from one layer to all of the input units of the next layer. 
- Dense layers are typically used in the final layers of a neural network, where they are used to make predictions or classifications. However, they can also be used in the middle layers of a neural network to help learn more complex relationships between the input and output data.
- Dense layers are typically used in the following situations:

	- When the input data is a vector of features.
	- When the output data is a vector of labels.
	- When the model needs to learn complex relationships between features.
```Python3
#I feed a 514 dimensional real-valued input to a Sequential model in Keras. My model is constructed in following way :
 predictivemodel = Sequential()
    predictivemodel.add(Dense(514, input_dim=514, W_regularizer=WeightRegularizer(l1=0.000001,l2=0.000001), init='normal'))
    predictivemodel.add(Dense(257, W_regularizer=WeightRegularizer(l1=0.000001,l2=0.000001), init='normal'))
    predictivemodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
 ```
    
    
When I print `model.summary()` I get following result:

```Python3

    Layer (type)    Output Shape  Param #     Connected to                   
================================================================
dense_1 (Dense) (None, 514)   264710      dense_input_1[0][0]              
________________________________________________________________
activation_1    (None, 514)   0           dense_1[0][0]                    
________________________________________________________________
dense_2 (Dense) (None, 257)   132355      activation_1[0][0]               
================================================================
Total params: 397065
________________________________________________________________ 

```

- **For the dense_1 layer , number of params is 264710. This is obtained as : 514 (input values) * 514 (neurons in the first layer) + 514 (bias values)**

- **For dense_2 layer, number of params is 132355. This is obtained as : 514 (input values) * 257 (neurons in the second layer) + 257 (bias values for neurons in the second layer)**

### Normalisation Layer  and Standarize layer
- Normalization layers and standardization layers are layers in neural networks that are used to normalize the inputs to the network. This can help to improve the performance of the network by making the inputs more consistent.

Normalization is a technique that is used to transform a set of data into a standard format. This can be done by subtracting the mean from each value and then dividing by the standard deviation. Normalization is often used in machine learning to improve the performance of models.
- In general, normalization layers are typically placed after the input layer and before each hidden layer. This is because normalization helps to stabilize the learning process by ensuring that the activations of each layer are on a similar scale. This can help to prevent the model from overfitting to the training data and can also help to improve the generalization performance of the model.

Standardization is a type of normalization where the data is scaled to have a mean of 0 and a standard deviation of 1. Standardization is often used in machine learning because it can help to improve the convergence of models.

Normalization layers and standardization layers are layers in neural networks that are used to normalize the inputs to the network. This can help to improve the performance of the network by making the inputs more consistent.

![](https://d33wubrfki0l68.cloudfront.net/c8f1f7a886548f82234f8a3b06faeecfbb88c657/42d49/images/layer-normalization.png)
    
![](https://theaisummer.com/static/ac89fbcf1c115f07ae68af695c28c4a0/ee604/normalization.png)

## A simpler way to understand what the bias is:
it is somehow similar to the constant b of a linear function

y = ax + b

It allows you to move the line up and down to fit the prediction with the data better.

Without b, the line always goes through the origin (0, 0) and you may get a poorer fit.

The bias is not an NN term. It's a generic algebra term to consider.

Y = M*X + C (straight line equation)

Now if C(Bias) = 0 then, the line will always pass through the origin, i.e. (0,0), and depends on only one parameter, i.e. M, which is the slope so we have less things to play with.
If we ignore the bias, many inputs may end up being represented by a lot of the same weights (i.e. the learnt weights mostly occur close to the origin (0,0). The model would then be limited to poorer quantities of good weights, instead of the many many more good weights it could better learn with bias. (Where poorly learnt weights lead to poorer guesses or a decrease in the neural net’s guessing power)


# Bias in Machine Learning

## Importance of Bias
### 1. Control Over Regression Line
   - In regression, bias allows for greater control over the line of best fit.
   - Without bias, the line only passes through the origin, limiting control over fitting.

### 2. Incorporating Prior Knowledge
   - Bias enables the incorporation of prior knowledge or assumptions into the model.
   - For instance, in a house price prediction model, the bias term can represent the value of a house with zero bedrooms, accounting for its inherent worth.

### 3. Preventing Overfitting
   - Bias can help prevent overfitting by introducing noise into the model.
   - Overfitting occurs when the model learns the training data too well but fails to generalize to new data.

### 4. Enhancing Model Accuracy
   - By introducing bias, the model can capture more complex relationships between input and output data.
   - This leads to improved accuracy, especially in scenarios with non-linear relationships.

### 5. Speeding Up Training
   - Bias aids in speeding up the training process by assisting the model in converging on a solution more quickly.

## Visual Representation
![Regression Line with and without Bias](https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2017-06-29-at-5.34.44-PM.png)
*Left panel: Regression line without bias. Right panel: Regression line with bias.*

![Bias in Machine Learning](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRF1osE7Ez3z_YE-9Dnyl0623qqrm7xVPQEPGDlXBDc9lPhrbhb7UYiCiPF3fa8IYodNA&usqp=CAU)
*Visual representation highlighting the significance of bias in machine learning.*

## Errors in Machine learning 
In machine learning, an error is a measure of how accurately an algorithm can make predictions for the previously unknown dataset. On the basis of these errors, the machine learning model is selected that can perform best on the particular dataset. There are mainly two types of errors in machine learning, which are:

Reducible errors: These errors can be reduced to improve the model accuracy. Such errors can further be classified into bias and Variance.
and Noice is the irreducable error.

### What is Bias?
- In general, a machine learning model analyses the data, find patterns in it and make predictions. While training, the model learns these patterns in the dataset and applies them to test data for prediction.
- 
- **While making predictions, a difference occurs between prediction values made by the model and actual values/expected values, and this difference is known as bias errors or Errors due to bias. **
- It can be defined as an inability of machine learning algorithms such as Linear Regression to capture the true relationship between the data points. Each algorithm begins with some amount of bias because bias occurs from assumptions in the model, which makes the target function simple to learn. A model has either:

- **Low Bias**: A low bias model will make fewer assumptions about the form of the target function.-- Good Model
- **High Bias**: A model with a high bias makes more assumptions, and the model becomes unable to capture the important features of our dataset. A high bias model also cannot perform well on new data.

###### Algorithms which has low and high bias
-	Some examples of machine learning algorithms with low bias are Decision Trees, k-Nearest Neighbours and Support Vector Machines. 
-	At the same time, an algorithm with high bias is Linear Regression, Linear Discriminant Analysis and Logistic Regression.

#### Ways to reduce High Bias:
High bias mainly occurs due to a much simple model. Below are some ways to reduce the high bias:

-Increase the input features as the model is underfitted.
-Decrease the regularization term.
-Use more complex models, such as including some polynomial features.
### What is Variance

The variance would specify the amount of variation in the prediction if the different training data was used.

- **Low variance** means there is a small variation in the prediction of the target function with changes in the training data set. 
- **High variance** shows a large variation in the prediction of the target function with changes in the training dataset.
- A model that shows high variance learns a lot and perform well with the training dataset, and does not generalize well with the unseen dataset. As a result, such a model gives good results with the training dataset but shows high error rates on the test dataset.
- Since, with high variance, the model learns too much from the dataset, it leads to overfitting of the model. A model with high variance has the below problems:

-	A high variance model leads to overfitting.
-	Increase model complexities.

- Some examples of machine learning algorithms with low variance are, Linear Regression, Logistic Regression, and Linear discriminant analysis.
- At the same time, algorithms with high variance are decision tree, Support Vector Machine, and K-nearest neighbours.

**Ways to Reduce High Variance:**
- Reduce the input features or number of parameters as a model is overfitted.
- Do not use a much complex model.
- Increase the training data.
- Increase the Regularization term.


![](https://vitalflux.com/wp-content/uploads/2020/12/Bias-Variance-Intuition-1024x551.png)

## Bias and Variance and its Tradeoff

![](https://static.javatpoint.com/tutorial/machine-learning/images/bias-and-variance-in-machine-learning4.png)

![](https://static.javatpoint.com/tutorial/machine-learning/images/bias-and-variance-in-machine-learning6.png)

For an accurate prediction of the model, algorithms need a low variance and low bias. But this is not possible because bias and variance are related to each other:

-	If we decrease the variance, it will increase the bias.
-	If we decrease the bias, it will increase the variance.

**1. Low-Bias, Low-Variance:**

The combination of low bias and low variance shows an ideal machine learning model. However, it is not possible practically.

**2. Low-Bias, High-Variance:**

With low bias and high variance, model predictions are inconsistent and accurate on average. This case occurs when the model learns with a large number of parameters and hence leads to an overfitting

**3. High-Bias, Low-Variance:**

With High bias and low variance, predictions are consistent but inaccurate on average. This case occurs when a model does not learn well with the training dataset or uses few numbers of the parameter. It leads to underfitting problems in the model.

**4. High-Bias, High-Variance:** 

With high bias and high variance, predictions are inconsistent and also inaccurate on average.

![](https://i.stack.imgur.com/6Y87l.png)

### Gradient Descent
Gradient Descent is defined as one of the most commonly used iterative optimization algorithms of machine learning to train the machine learning and deep learning models. It helps in finding the local minimum of a function.

The best way to define the local minimum or local maximum of a function using gradient descent is as follows:

- If we move towards a negative gradient or away from the gradient of the function at the current point, it will give the local minimum of that function.
- Whenever we move towards a positive gradient or towards the gradient of the function at the current point, we will get the local maximum of that function.

- The main objective of using a gradient descent algorithm is to minimize the cost function using iteration. To achieve this goal, it performs two steps iteratively:

-	Calculates the first-order derivative of the function to compute the gradient or slope of that function.
-	Move away from the direction of the gradient, which means slope increased from the current point by alpha times, where Alpha is defined as Learning Rate. -
-	It is a tuning parameter in the optimization process which helps to decide the length of the steps


![](https://static.javatpoint.com/tutorial/machine-learning/images/gradient-descent-in-machine-learning1.png)

##### How does Gradient Descent work?

![](https://static.javatpoint.com/tutorial/machine-learning/images/gradient-descent-in-machine-learning2.png)

The starting point(shown in above fig.) is used to evaluate the performance as it is considered just as an arbitrary point. At this starting point, we will derive the first derivative or slope and then use a tangent line to calculate the steepness of this slope. Further, this slope will inform the updates to the parameters (weights and bias).

The slope becomes steeper at the starting point or arbitrary point, but whenever new parameters are generated, then steepness gradually reduces, and at the lowest point, it approaches the lowest point, which is called a point of convergence.

The main objective of gradient descent is to minimize the cost function or the error between expected and actual. To minimize the cost function, two data points are required:

###### Direction & Learning Rate
These two factors are used to determine the partial derivative calculation of future iteration and allow it to the point of convergence or local minimum or global minimum. Let's discuss learning rate factors in brief;
- Learning Rate:
It is defined as the step size taken to reach the minimum or lowest point. This is typically a small value that is evaluated and updated based on the behavior of the cost function. If the learning rate is high, it results in larger steps but also leads to risks of overshooting the minimum. At the same time, a low learning rate shows the small step sizes, which compromises overall efficiency but gives the advantage of more precision.

![](https://static.javatpoint.com/tutorial/machine-learning/images/gradient-descent-in-machine-learning3.png)

#### Types of Gradient Descent
Based on the error in various training models, the Gradient Descent learning algorithm can be divided into Batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

#### Challenges with the Gradient Descent
****1.Local Minima and Saddle Point and Global minima:**
For convex problems, gradient descent can find the global minimum easily, while for non-convex problems, it is sometimes difficult to find the global minimum, where the machine learning models achieve the best results.

- The point at which a function takes the minimum value is called **global minima**. 
- However, when the goal is to minimize the function and solved using optimization algorithms such as gradient descent, it may so happen that function may appear to have a minimum value at different points. 
- Those several points which appear to be minima but are not the point where the function actually takes the minimum value are called local minima. 
- Machine learning algorithms such as gradient descent algorithms may get stuck in local minima during the training of the models. 
- Gradient descent is able to find local minima most of the time and not global minima because the gradient does not point in the direction of the steepest descent. 
- Current techniques to find global minima either require extremely high iteration counts or a large number of random restarts for good performance. Global optimization problems can also be quite difficult when high loss barriers exist between local minima.
- 
![](https://vitalflux.com/wp-content/uploads/2020/09/local-minima-vs-global-minima-1.png)

![](https://vitalflux.com/wp-content/uploads/2020/10/local_minima_vs_global_minima.gif)

**Fig 3. Animation representing local minima and global minima
**

Pay attention to some of the following in the above animation:

- The gradient at different points is found out.
- If the gradient value is positive at a point, it will be required to move to left or reduce the weight.
- If the gradient value is negative at a point, it will be required to increment the value of weight.
- The above two steps are done until the minima is reached.
- The minima could either be local minima or the global minima. There are different techniques which can be used to find local vs global minima.

**2. Vanishing and Exploding Gradient**
In a deep neural network, if the model is trained with gradient descent and backpropagation, there can occur two more issues other than local minima and saddle point.

- **Vanishing Gradients:**
Vanishing Gradient occurs when the gradient is smaller than expected. During backpropagation, this gradient becomes smaller that causing the decrease in the learning rate of earlier layers than the later layer of the network. Once this happens, the weight parameters update until they become insignificant.
- **Exploding Gradient:** 
Exploding gradient is just opposite to the vanishing gradient as it occurs when the Gradient is too large and creates a stable model. Further, in this scenario, model weight increases, and they will be represented as NaN. This problem can be solved using the dimensionality reduction technique, which helps to minimize complexity within the model.

##### There are a number of techniques that can be used to avoid gradient descent problems. Some of these techniques include:

- **Using a learning rate:** The learning rate is a hyperparameter that controls the size of the steps that the algorithm takes towards the minimum of the cost function. A small learning rate can help to prevent the algorithm from diverging, while a large learning rate can help to prevent the algorithm from converging too slowly.
- **Using momentum:** Momentum is a technique that helps to prevent the algorithm from getting stuck in local minima. Momentum works by adding a fraction of the previous gradient to the current gradient. This helps to smooth out the path that the algorithm takes towards the minimum of the cost function.
- **Using adaptive learning rates:** Adaptive learning rates are learning rates that change over time. Adaptive learning rates can help to prevent the algorithm from getting stuck in local minima and can also help to improve the convergence speed of the algorithm.
- Use a large batch size: Using a large batch size can help to stabilize the gradient descent algorithm and prevent it from diverging.
- **Regularize the model:** Regularization is a technique that adds a penalty to the cost function to prevent the model from overfitting the training data. Regularization can help to prevent the algorithm from getting stuck in local minima.
- **Use a validation set:** A validation set is a set of data that is not used to train the model. The model is evaluated on the validation set after each epoch to see if it is overfitting the training data. If the model is overfitting, the learning rate can be reduced or the regularization strength can be increased.





# Important Links and Resources

## Artificial Intelligence

- **Roadmap to becoming an Artificial Intelligence Expert in 2022:**  
  [AI Expert Roadmap](https://github.com/AMAI-GmbH/AI-Expert-Roadmap)

- **All Books and Resources for AI:**  
  [From 0 to Research Scientist resources guide](https://github.com/ahmedbahaaeldin/From-0-to-Research-Scientist-resources-guide)

## Machine Learning

- **Machine Learning with Python Roadmap:**  
  [Project Based Learning](https://github.com/practical-tutorials/project-based-learning#python)

- **Fast.ai Machine Learning Course:**  
  [Fast.ai Course](https://course18.fast.ai/ml.html)

- **Notes on Andrew Ng's Machine Learning Course:**  
  [Andrew NG Notes](https://github.com/ashishpatel26/Andrew-NG-Notes)

## Deep Learning

- **Deep Learning Roadmap:**  
  [Deep Learning Roadmap](https://github.com/instillai/deep-learning-roadmap)

- **Deep Learning YouTube Playlist:**  
  [Deep Learning YouTube Playlist](https://www.youtube.com/playlist?list=PLWqmopvp6uH21bSOxvnZZmnr2_RKJSItY)

- **Neural Networks and Deep Learning Summary:**  
  [DeepLearning.ai Summary](https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master/1-%20Neural%20Networks%20and%20Deep%20Learning)

- **Fast.ai Deep Learning Course:**  
  [Fast.ai Course](https://course.fast.ai/)

## Python

- **Python Developer Roadmap:**  
  [Python Roadmap](https://roadmap.sh/python)

- **100+ Python Challenging Programming Exercises:**  
  [Python Programming Exercises](https://github.com/zhiwehu/Python-programming-exercises)

- **Curated List of Project-Based Tutorials:**  
  [Project-Based Learning](https://github.com/practical-tutorials/project-based-learning#python)

- **Master Programming by Recreating Your Favorite Technologies from Scratch:**  
  [Build Your Own Programming Language](https://github.com/codecrafters-io/build-your-own-x#build-your-own-programming-language)

- **List of Awesome Beginners-Friendly Projects:**  
  [Awesome for Beginners](https://github.com/MunGell/awesome-for-beginners#python)


## Machine Learning and Deep Learning Resources

### 1. Articles to Gain Knowledge on ML and DL

#### Classifications Types and Differences
- **Binary Classification:**
  - Used for distinguishing between two distinct classes where data belongs exclusively to one class, e.g., classifying if a product review is positive or negative.
- **Multiclass Classification:**
  - Used for scenarios with three or more exclusive classes, e.g., classifying traffic signal lights as red, yellow, or green.
- **Multilabel Classification:**
  - Applicable when data may belong to none or multiple classes simultaneously, e.g., identifying various traffic signs in an image.

#### Understanding Sigmoid and Softmax Functions
[Difference between Sigmoid and Softmax](https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/)
- **Sigmoid Function:**
  - Suitable for multi-label classification where outputs are not mutually exclusive.
  - Allows high probabilities for multiple classes simultaneously.
  - Example: Classifying diseases in medical images.
- **Softmax Function:**
  - Ideal for multi-class classification with mutually exclusive outputs.
  - Ensures the sum of probabilities for all classes equals one.
  - Example: Classifying images of handwritten digits.
  
#### Relevant Notes and Points:
- **Multi-Label vs. Multi-Class:**
  - Multi-label classification allows for multiple classes to be present simultaneously, while multi-class classification restricts to a single class per instance.
- **Choice between Sigmoid and Softmax:**
  - Sigmoid is suitable for scenarios where multiple classes can be present simultaneously.
  - Softmax is preferable when only one class should be identified per instance.
- **Output Interpretation:**
  - Sigmoid outputs probabilities for each class independently, summing up to more than one.
  - Softmax outputs probabilities ensuring the sum equals one, facilitating exclusive class selection.
  
#### Visual Representation:
- ![Sigmoid Function Diagram](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/04/Screenshot-from-2021-04-01-17-25-02.png)
- ![Softmax Function Diagram](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/04/Screenshot-from-2021-04-01-17-26-10.png)

#### Additional Note:
- When dealing with output scenarios where more than one class can be present simultaneously, such as in medical diagnoses, sigmoid activation is appropriate to accommodate this multi-label classification requirement. Conversely, softmax activation is suitable for tasks where only one class should be predicted per instance, ensuring exclusive and mutually exclusive class assignments, such as in image recognition tasks.


##### Linear and Non Linear Functions

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




## Activation Functions in Neural Networks

(https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*p_hyqAtyI8pbt2kEl6siOQ.png)

![Characteristics-of-activation-functions-and-optimizers](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/99fc35d1-7358-402a-8fac-25bd877e7a14)


Activation functions play a crucial role in neural networks by introducing non-linearity and controlling the output range of neurons. Here's a detailed overview of some common activation functions used in neural networks:

### Sigmoid Hidden Layer Activation Function

The sigmoid activation function, also known as the logistic function, maps any real-valued number to a value between 0 and 1. It's commonly used in hidden layers, but it's important to note that it suffers from the vanishing gradient problem.

- The function is calculated as: `1.0 / (1.0 + e^-x)`
- Sigmoid activation is appropriate for binary classification tasks where the target labels are 0 or 1.

![sigmoid](https://machinelearningmastery.com/wp-content/uploads/2021/08/sigmoid.png)

### Tanh Hidden Layer Activation Function

The hyperbolic tangent (tanh) function is similar to the sigmoid function but maps values to the range -1 to 1. It's often preferred over the sigmoid function due to its zero-centered output, which helps mitigate the vanishing gradient problem.

- The function is calculated as: `(e^x – e^-x) / (e^x + e^-x)`

![tanh](https://miro.medium.com/v2/resize:fit:900/0*FIFkhXuir7JO0Utc.jpg)

### Activation for Output Layers

The output layer's activation function depends on the nature of the task:
1. **Linear Activation**: No activation function. It's suitable for regression tasks where the output values can be any real number.
2. **Logistic (Sigmoid)**: Suitable for binary classification tasks, similar to using it in hidden layers.
3. **Softmax**: Ideal for multi-class classification tasks. It outputs a probability distribution over multiple classes.

![output_activations](https://machinelearningmastery.com/wp-content/uploads/2020/12/How-to-Choose-an-Output-Layer-Activation-Function.png)

### Vanishing Gradient Problem

The vanishing gradient problem occurs when gradients become extremely small during backpropagation, leading to slow convergence or convergence to suboptimal solutions. It's often associated with sigmoid and tanh activation functions.

#### Methods to Overcome the Problem

1. **ReLU (Rectified Linear Unit)**: Replacing sigmoid and tanh with ReLU can alleviate the vanishing gradient problem. ReLU is computationally efficient and helps prevent gradient saturation.
2. **Weight Initialization**: Properly initializing weights, such as using He Normal or He Uniform initialization, can help prevent gradients from vanishing during training.

![vanishing_gradient](https://www.kdnuggets.com/wp-content/uploads/jacob-vanishing-gradient-7.jpg)

![relu_derivative](https://www.kdnuggets.com/wp-content/uploads/jacob-vanishing-gradient-8.png)

Certainly! Let's include information about the gradient exploding problem and how to address it in neural networks.

### Gradient Exploding Problem

The gradient exploding problem occurs when gradients become excessively large during backpropagation, leading to unstable training and divergent behavior. This phenomenon is particularly common in deep neural networks with recurrent connections, such as recurrent neural networks (RNNs), where gradients can accumulate and amplify as they propagate through the network.

#### Causes of Gradient Explosion

Gradient explosion can be caused by factors such as:
- High learning rates: Using learning rates that are too large can cause gradients to increase exponentially.
- Poor weight initialization: Initializing weights with large values can exacerbate gradient explosion.
- Long sequences: In RNNs, processing long sequences can lead to gradient accumulation and explosion over time.

#### Preventing Gradient Explosion

Several techniques can help prevent gradient explosion:
1. **Gradient Clipping**: Limiting the maximum gradient value during training can prevent it from growing too large. This is achieved by scaling down gradients if their norm exceeds a predefined threshold.
2. **Gradient Regularization**: Applying L2 regularization to the network's parameters can help mitigate the growth of gradients by penalizing large weight values.
3. **Proper Weight Initialization**: Initializing weights using techniques such as Xavier initialization or He initialization can help keep gradients within a reasonable range.
4. **Lowering Learning Rate**: Gradually reducing the learning rate during training, or using adaptive learning rate algorithms like Adam, can prevent sudden spikes in gradient magnitude.

#### Detecting Gradient Explosion

Detecting gradient explosion during training can be challenging, but there are several signs to watch for:
- Sudden spikes in loss function values or training/validation error.
- Oscillations or instability in training curves.
- NaN (not a number) or infinite values in gradients or weight matrices.

#### Impact of Gradient Explosion

Gradient explosion can lead to unstable training dynamics, making it difficult for the model to converge to a good solution. In extreme cases, it can cause the loss function to diverge to infinity, resulting in training failure.

#### Example Code Snippet for Gradient Clipping:

```python
import tensorflow as tf

# Define the optimizer with gradient clipping
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)  # Clip gradients to [-1.0, 1.0]
```




#### All Books and Resources for Deep Learmomg-- https://github.com/ahmedbahaaeldin/From-0-to-Research-Scientist-resources-guide

## 1. All Books and Resources for Deep Learning
- [GitHub Repository: From 0 to Research Scientist Resources Guide](https://github.com/ahmedbahaaeldin/From-0-to-Research-Scientist-resources-guide)

## 2. TensorFlow
- [edX Course: IBM Deep Learning with TensorFlow](https://learning.edx.org/course/course-v1:IBM+DL0120EN+2T2021/home)

## 3. Convolutional Neural Networks (CNN)

![](https://editor.analyticsvidhya.com/uploads/183560_qcMBDPuKpDvICcdd.png)

### 3.1 Stanford Course on CNN
- [Stanford University: CS231n - Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

### 3.2 Made with ML CNN Explanation
- [Made with ML: Foundations of Convolutional Neural Networks](https://madewithml.com/courses/foundations/convolutional-neural-networks/)

### 3.3 CNN Explainer
- [CNN Explainer by Poloclub](https://poloclub.github.io/cnn-explainer/)

### 3.4 Analytics Vidhya CNN Tutorial
- [Analytics Vidhya: Convolutional Neural Networks (CNN)](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/)

### 3.5 Simplilearn Deep Learning Tutorial
- [Simplilearn: Convolutional Neural Network Tutorial](https://www.simplilearn.com/tutorials/deep-learning-tutorial/convolutional-neural-network)




## NLP

#### All Books and Resources for Deep Learmomg-- https://github.com/ahmedbahaaeldin/From-0-to-Research-Scientist-resources-guide









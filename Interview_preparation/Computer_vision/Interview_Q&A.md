Max pooling is often used in CNNs for object recognition tasks, as it helps to identify the most distinctive features of an object, such as its edges and corners. In average pooling, the output value for each pooling region is the average of the input values within that region.


## Interview Questions

Prepare for projects you have done?

Few basics questions i have been asked and asking:

Why you used this particular approach for this problem, earlier methods you tried..

Type of activation function,Optimiser, loss function

What is gradient descent/minibatch/

Explain Back propogation

Explain how CNN work,What is CNN, RNN

How to avoid overfitting underfitting, 
Ans-dropout, regularisation, check data annotations

Cnn-- pooling layer, type of pooling layer, maxpool, global pool, avg. Pool
Padding

Given input image size, stride, padding, what will be output image size 

Different type of backbone, head architecture in CNN

Batch normalisation, standard scaling, minmax scaling

Standardization vs normalisation

Object detection -- NMS, IoU, anchor boxes


What u know about model quantization, pruning, optimization


1. Tell me about Yourself?

2. Tell me more about Projects and Tell me About your Project?

3. What are the Libraries did you use In your Project?

4. How do u use GPU on PyTorch ?

5. What are the ways we can use in GPU for faster processing for Deep Learning Models?

6. How can u speed the model process?

7. What are the things you will do when a model is overfitting?

8. What kind of Feature Extraction Methods did you use?

9. Can u tell me what are the Feature Extraction Methods we will use in Computer Vision?

10. What is Convolution and Tell me How does it work?

11. Can you tell me how Binary Classification Works in image Classification?

12. Tell me about Object Recognition methods you know?

13. Tell me more about YOLO?

14. Do you know any Creating/Streaming a video using OpenCV?

15. What are the metrics do you use when you are evaluating a Deep learning model and for Image classification model ?

16.how do you choose the best metric for a problem statement?

17.What is feature map in CNN?


Coding Questions:
1. Replace vowels in reverse order in a given word.

2. Get the sum of even numbers in dictionaries, even if it has nested dictionaries, using recursion.



### How can i speed up the processing in deep learning models 
Speeding up processing in deep learning models can be achieved through various techniques and optimizations. Here are some strategies you can consider:

1. **Hardware Acceleration**: Utilize hardware accelerators such as GPUs (Graphics Processing Units) or TPUs (Tensor Processing Units) for faster computation. These specialized hardware devices are designed to handle the heavy computational tasks involved in deep learning more efficiently than CPUs.

2. **Model Optimization**:
   - **Model Pruning**: Remove unnecessary connections or parameters from the model to reduce its size and computational complexity without significantly affecting performance.
   - **Quantization**: Reduce the precision of the model's weights and activations (e.g., from 32-bit floating point to 8-bit integers) to decrease memory usage and computational requirements.
   - **Model Compression**: Employ techniques like knowledge distillation or model distillation to train a smaller model that approximates the behavior of a larger, more complex model.
   - **Network Architecture**: Choose lighter network architectures or modify existing ones to reduce the number of parameters and operations required for inference.

3. **Batch Processing**: Increase the batch size during training and inference to take advantage of parallelism and reduce overhead, especially when using hardware accelerators.

4. **Parallelization**:
   - **Data Parallelism**: Distribute training across multiple devices or machines by splitting the dataset and processing batches in parallel.
   - **Model Parallelism**: Split the model across multiple devices or machines, allowing different parts of the model to be processed independently.

5. **Optimized Libraries and Frameworks**: Utilize optimized deep learning libraries and frameworks like TensorFlow with XLA (Accelerated Linear Algebra) or PyTorch with TorchScript, which are specifically designed to optimize computations and leverage hardware accelerators efficiently.

6. **Caching and Memoization**: Cache intermediate results or precompute expensive computations to avoid redundant calculations during inference.

7. **Asynchronous Execution**: Use asynchronous processing to overlap computation with data loading and preprocessing, reducing idle time and improving overall throughput.

8. **Algorithmic Improvements**: Explore and implement algorithmic optimizations specific to your model or problem domain, such as more efficient data augmentation techniques or custom loss functions.

9. **Profiling and Tuning**: Profile your code and identify bottlenecks using tools like TensorFlow Profiler or PyTorch Profiler. Optimize critical sections of code, such as data loading, forward passes, and backward passes, based on profiling results.

By applying these strategies judiciously, you can significantly improve the speed and efficiency of deep learning models, making them more practical for real-world applications.

### What is Log Likelihood and How do you use it as a Loss funtion?
 The log likelihood is a concept commonly used in statistics and machine learning. It measures the likelihood of a set of observations given a probability distribution and a set of parameters. In the context of a loss function, the negative log likelihood is often used as a component of the overall loss function in probabilistic models, such as those used in logistic regression or neural networks.

To understand how it makes a loss function, consider that in many machine learning algorithms, the goal is to maximize the likelihood of the observed data given the model parameters. However, maximizing the likelihood directly can be complex, so instead, the negative log likelihood is minimized. This is mathematically equivalent to maximizing the likelihood and has some computational and theoretical advantages.

By minimizing the negative log likelihood, the model is essentially being trained to produce high probabilities for the observed data. In other words, the model is penalized when it assigns low probabilities to the observed outcomes. This aligns with the goal of many machine learning tasks, which is to find model parameters that best explain the observed data.

In summary, the log likelihood is a measure of the likelihood of observed data given a model, and using the negative log likelihood as a component of the loss function in machine learning allows for effective training of probabilistic models.  


**Notes:**

- **Why use Max Pooling instead of Average Pooling?**
  - Features tend to encode the spatial presence of patterns or concepts across different tiles of the feature map.
  - Max pooling captures the maximal presence of different features, which is more informative than their average presence.

- **Use of Pooling after Convolution:**
  - Pooling is typically used after convolution to reduce the spatial dimensions of the feature maps, which helps in controlling overfitting and reducing computational complexity while retaining the most relevant information.

**Table for Problem Type, Last-layer Activation, and Loss Function:**

| Problem Type     | Last-layer Activation | Loss Function      |
|------------------|-----------------------|--------------------|
| Image Classification | Softmax               | Categorical Crossentropy |
| Object Detection    | Sigmoid (for binary classification) or Softmax (for multi-class classification) | Binary Crossentropy or Categorical Crossentropy |
| Image Segmentation  | Softmax               | Categorical Crossentropy |
| Sequence Classification (e.g., Sentiment Analysis) | Sigmoid (for binary classification) or Softmax (for multi-class classification) | Binary Crossentropy or Categorical Crossentropy |
| Sequence Generation (e.g., Language Modeling)      | Softmax               | Categorical Crossentropy |
| Regression (e.g., predicting house prices)         | Linear                | Mean Squared Error   |
| Time Series Forecasting                           | Linear or Tanh        | Mean Squared Error or Mean Absolute Error |

**Data Preprocessing Steps:**
1. Read the picture files.
2. Decode the JPEG content to RGB grids of pixels.
3. Convert these into floating-point tensors.
4. Rescale the pixel values (between 0 and 255) to the [0, 1] interval.

Keras utilities can handle these steps automatically, particularly through the `ImageDataGenerator` class in `keras.preprocessing.image`.

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
```

Note: In the provided code snippet, images are rescaled by 1/255 to normalize pixel values, and they are resized to 150x150 dimensions. The `class_mode` parameter is set to `'binary'` because the loss function used is `binary_crossentropy`, indicating binary classification.



Few openCV algorithms, about edge detection, circle detection, contour, threshold, masking particular region

Date: 02-05-2023
Company name: Walmart
role: Data Scientist
Topic : Computer vision, excel, time series

1.What are the different types of Pooling? Explain their characteristics.

Max pooling: Once we obtain the feature map of the input, we will apply a filter of determined shapes across the feature map to get the maximum value from that portion of the feature map. It is also known as subsampling because from the entire portion of the feature map covered by filter or kernel we are sampling one single maximum value.
Average pooling: Computes the average value of the feature map covered by kernel or filter, and takes the floor value of the result.
Sum pooling: Computes the sum of all elements in that window.


2. What is a Moving Average Process in Time series? 

In time-series analysis, moving-average process, is a common approach for modeling univariate time series. The moving-average model specifies that the output variable depends linearly on the current and various past values of a stochastic term.

3. What is the difference between SQL having vs where? 

The WHERE clause specifies the criteria which individual records must meet to be selected by a query. It can be used without the GROUP by clause. The HAVING clause cannot be used without the GROUP BY clause . The WHERE clause selects rows before grouping. The HAVING clause selects rows after grouping. The WHERE clause cannot contain aggregate functions. The HAVING clause can contain aggregate functions


4. What is Relative cell referencing in excel?

In Relative referencing, there is a change when copying a formula from one cell to another cell with respect to the destination. cells’ address Meanwhile, there is no change in Absolute cell referencing when a formula is copied, irrespective of the cell’s destination. This type of referencing is there by default. Relative cell referencing doesn’t require a dollar sign in the formula.

Today's Interview QnAs

Company - Wipro

Role: Data Scientist


1.What is Selection Bias and what are various types?

Ans: Selection bias takes place when data is chosen in a way that is not reflective of real-world data distribution. This happens because proper randomization is not achieved when collecting data.

Types of selection bias -

• Sampling bias: occurs when randomization is not properly achieved during data collection. To give an example, imagine that there are 10 people in a room and you ask if they prefer grapes or bananas. If you only surveyed the three females and concluded that the majority of people like grapes, you’d have demonstrated sampling bias.

•Convergence bias: occurs when data is not selected in a representative manner. e.g. when you collect data by only surveying customers who purchased your product and not another half, your dataset does not represent the group of people who did not purchase your product.

•Participation bias: occurs when the data is unrepresentative due to participations gaps in the data collection process.


2.Can you compare the validation test with the test set?

Ans: Validation Dataset: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.

Test Dataset: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.


3.What is the aim of conducting A/B Testing?

Ans: Creating a website or email marketing campaign is just the first step in marketing. Once you have a website, you’ll want to know if it helps or hinders sales.
A/B testing lets you know what words, phrases, images, videos, testimonials, and other elements work best. Even the simplest changes can impact conversion rates.
To understand how A/B testing works, let’s take a look at an example.

Imagine you have two different designs for a landing page—and you want to know which one will perform better.

After you create your designs, you give one landing to one group and you send the other version to the second group. Then you see how each landing page performs in metrics such as traffic, clicks, or conversions. To understand how A/B testing works, let’s take a look at an example.
Imagine you have two different designs for a landing page—and you want to know which one will perform better.
After you create your designs, you give one landing to one group and you send the other version to the second group. Then you see how each landing page performs in metrics such as traffic, clicks, or conversions.


4.What is Hierarchical Clustering and what are it's 2 types?

Ans: Hierarchical clustering  is a method of cluster analysis which seeks to build a hierarchy of clusters. 

Hierarchical Clustering is of two types: 

•Agglomerative: This is a "bottom-up" approach where each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

 • Divisive: This is a "top-down" approach where all observations start in one big cluster, and splits are performed recursively as one moves down the hierarchy.

..................


Company - Bosch

Role: Data Scientist


1. What is a logistic function? What is the range of values of a logistic function?

f(z) = 1/(1+e -z )
The values of a logistic function will range from 0 to 1. The values of Z will vary from -infinity to +infinity.


2. What is the difference between R square and adjusted R square?

R square and adjusted R square values are used for model validation in case of linear regression. R square indicates the variation of all the independent variables on the dependent variable. i.e. it considers all the independent variable to explain the variation. In the case of Adjusted R squared, it considers only significant variables(P values less than 0.05) to indicate the percentage of variation in the model.

Thus Adjusted R2 is always lesser then R2.


3. What is stratify in Train_test_split?

Stratification means that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset. So if my input data has 60% 0's and 40% 1's as my class label, then my train and test dataset will also have the similar proportions.


4. What is Backpropagation in Artificial Neuron Network?

Backpropagation is the method of fine-tuning the weights of a neural network based on the error rate obtained in the previous epoch (i.e., iteration). Proper tuning of the weights allows you to reduce error rates and make the model reliable by increasing its generalization.

..................



Company Name - EY

Role - Data Scientist
_________________

Q1.  What is Yolo?

Ans. YOLO - You Only Look Once is an algorithm proposed by by Redmond et. al in a research article published at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) as a conference paper, winning OpenCV People’s Choice Award.

Compared to the approach taken by object detection algorithms before YOLO, which repurpose classifiers to perform detection, YOLO proposes the use of an end-to-end neural network that makes predictions of bounding boxes and class probabilities all at once. 


Q2. Select perfect k for k means

Ans.  There is a popular method known as elbow method which is used to determine the optimal value of K to perform the K-Means Clustering Algorithm. The basic idea behind this method is that it plots the various values of cost with changing k. As the value of K increases, there will be fewer elements in the cluster.


Q3.  Model metrics when you have outliers?

Ans.  The bigger the MAE, the more critical the error is. It is robust to outliers. Therefore, by taking the absolute values, MAE can deal with the outliers.


Q4.  Features for food delivery data to give discount to selected customers?

Ans. The features for same can be: Total number of orders, Frequency of ordering per week, Amount paid per order, Distance travelled by delivery man etc.  

Today's Interview QnAs

Company Name - Course5i

Role - Data Scientist
___________________

Q1.   How to improve model performance?

Ans. Follow these techniques:
1. Use Validation methods
2. Add more data
3. Apply feature engineering techniques(Normalization, Imputation etc)
4. Compare Multiple algorithms
5. Hyperparameter Tuning


Q2.  Standardization vs log transformation?

Ans. Standardization is the process of putting different variables on the same scale. This process allows you to compare scores between different types of variables. Typically, to standardize variables, you calculate the mean and standard deviation for a variable. Log transformation is a data transformation method in which it replaces each variable x with a log(x). Log-transform decreases skew in some distributions, especially with large outliers. But, it may not be useful as well if the original distributed is not skewed. Also, log transform may not be applied to some cases (negative values), but standardization is always applicable (except σ=0).

Q3.  Object detection?

Ans. Object detection is a computer vision technique for locating instances of objects in images or videos. Object detection algorithms typically leverage machine learning or deep learning to produce meaningful results.


Q4.  Data cleaning steps?

Ans. Step 1: Remove duplicate or irrelevant observations. Remove unwanted observations from your dataset, including duplicate observations or irrelevant observations. 
*Step 2: Fix structural errors. 
*Step 3: Filter unwanted outliers. 
*Step 4: Handle missing data. 
*Step 5: Validate your data if it's appropriate according to problem statement


Q5.  Upsampling and downsampling methods

Ans. In a classification task, there is a high chance for the algorithm to be biased if the dataset is imbalanced. An imbalanced dataset is one in which the number of samples in one class is very higher or lesser than the number of samples in the other class.

To counter such imbalanced datasets, we use a technique called up-sampling and down-sampling.

In up-sampling, we randomly duplicate the observations from the minority class in order to reinforce its signal. The most common way is to resample with replacement.

In down-sampling, we randomly remove the observations from the majority class. Thus after up-sampling or down-sampling, the dataset becomes balanced with same number of observations in each class.


Q6.  hypothesis testing?

Ans.  Hypothesis testing is defined as the process of choosing hypotheses for a particular probability distribution, based on observed data. We use this to test whether a hypothesis can be accepted or not.



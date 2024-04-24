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



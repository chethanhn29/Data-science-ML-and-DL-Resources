#### 1. [Bias and Variance](#bias-and-variance)  
#### 2. [Bias and Variance Tradeoff](#bias-and-variance-tradeoff)
#### 3. [Underfitting](#underfitting)
#### 4. [Methods to Avoid Underfitting](#methods-to-avoid-underfitting)
#### 4. [Overfitting ](#overfitting)
#### 5. [Methods to avoid Overfitting](#methods-to-avoid-overfitting)
#### 5. [Regularization methods to control Overfitting and Underfitting](#regularization-methods-to-control-overfitting-and-underfitting)




## Bias and variance
- Bias is interpreted as the model error encountered for the training data. and 

- Variance is interpreted as the model error encountered for the test data.

![](https://miro.medium.com/v2/resize:fit:468/0*KMjuIi52_tgtdgnl.png)



![](https://www.embedded.com/wp-content/uploads/media-1307131-ibm-fairnessai.png)

![](https://static.javatpoint.com/tutorial/machine-learning/images/bias-and-variance-in-machine-learning.png)

## Bias and Variance Tradeoff

![](https://miro.medium.com/v2/resize:fit:1400/1*8nIJ2w5nRrdJQuLhtZ4L7g.png)

![](https://media.geeksforgeeks.org/wp-content/uploads/20200107023418/1_oO0KYF7Z84nePqfsJ9E0WQ.png)

## Underfitting
  - high Bias, low Variance
  - Simple Model , doesn't learn Much from the data so the Model predictions will always be far away from the ground truth(Actual Values) ie, High bias.
  - Error for both training data and unseen or test data is more.
  - ex:Training Error=15% and Testing error =16%
  - Accuracy is less for both training and test data.
  - 
![](https://miro.medium.com/v2/resize:fit:1396/1*lARssDbZVTvk4S-Dk1g-eA.png)
![](https://miro.medium.com/v2/resize:fit:1400/1*9DtauyXaQFAOi31Pp1XQUg.png)

## Methods to Avoid Underfitting
  - Less Regularization
  - Bigger network(Hiden layers,Hidden Units)
  - Train it Longer
  - Advanced optimization algotithms
  - Neural Network Architecture Search.

## Overfitting 
  - Traning Error is Less and Validation or Unseen data error is more compared to Training error. 
  - Ex: Training Error=1% and Validation error=11%
  - Accuracy[Training data > Test data ]
  - Low Bias and High Variance
  - Compex Model Learns all the Parameters of Training data.
  - Model Learns Too much from the Training data and it has more accuracy in Training data and Doesn't Do well in Validation or Test data or Unseen data.

## Methods to Avoid Overfitting
  - More Regularization
  - Increase Regularization impact,L and L2 Regularization and Droput Technique 
  - More Data
  - CNN Architecture Search.

![](https://miro.medium.com/v2/resize:fit:1400/1*9DtauyXaQFAOi31Pp1XQUg.png)

#### **Note**:
-  To avoid Bias-Variance Tradeoff without hurting each other we can choose Bigger network and More data.



## Regularization methods to control Overfitting and Underfitting
1. [L1 regularization Or Lasso Regression](#l1-regularization-or-lasso-regression)
2. [L2 Regularization or Ridge Regression](#l2-regularization-or-ridge-regression)
3. [Elastic Net](#elastic-net)

for more Regularization terms check Keras Docuementation
  - [Different Regularization Layers](https://keras.io/api/layers/regularization_layers/)
  - [Apply Different Regularizers for layers ](https://keras.io/api/layers/regularizers/)
  - [Regularization techniques for training deep neural networks](https://theaisummer.com/regularization/)
  - **When the number of features are quite large you can give L1 a shot but L2 should always be your blind eye pick.**
  - Even in the case when you have a strong reason to use L1 given the number of features, I would recommend going for Elastic Nets instead. 
  - Even in a situation where you might benefit from L1's sparsity in order to do feature selection, **using L2 on the remaining variables is likely to give better results than L1 by itself.**




#### L2 Regularization or Ridge Regression
  - To Watch Josh parmer Tutorial for Ridge Regression Click [Here](https://www.youtube.com/watch?v=Q81RR3yKn30&t=911s)
![](https://www.andreaperlato.com/img/ridge.png)
  - weight Sharing Technique.
  - The main idea behind Rigge Regression is to find a **New Line** that doesn't fit the training data too well because it causes Model to Overfit.
  - The Ridge regression induce little Bias to **Least Square Lines** and minimizes the Least  Squares for overfitted Moel line.
  - This makes to significantly decreasing in variance by adding little bias , this makes to predict better(Unseen or test data).
  - Overall Ridge regression provides Better Predictions tha Least Squares by adding penalty( Lamda). 
  -**Ridge Regression= (Sum of Squared Residuals)+ ( Lamda * Slope^2 )**
  - The Ridge regression is usefull when we have mose usefull variables.
  - **Weight sharing** Weight sharing refers to the ability of a regularization technique to share weights between different features. This can be useful when there are multiple features that are correlated with each other. Weight sharing can help to reduce the number of parameters that need to be estimated, and improve the generalization performance of the model. refers to the ability of a regularization technique to share weights between different features. This can be useful when there are multiple features that are correlated with each other. Weight sharing can help to reduce the number of parameters that need to be estimated, and improve the generalization performance of the model.
  - L2 regularization, also known as Ridge regression, is a weight sharing regularization technique. It works by adding a penalty to the model objective function that is proportional to the square of the model coefficients. This penalty encourages the model coefficients to be small, but it does not force them to be zero.
  - 
![ezgif com-gif-maker](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/aa5540ce-b71f-485f-95ad-084fff53c8e7)

  - In **features and Target** Plot, When the Slope is steeper , the change in small amount of Features(weights) induces large amount of targets.ie,when the Slope is steeper,the prediction for Target(size) is Very much sensitive to Small changes in features(Weights).
  - When the Slope is small the , the Large amount of Weights induce small amount of changes in target Values.ie,When the Slope is small ,prediction for Target(size) is Less sensitive to  changes in features(Weights).
  - when the lamda Increases , the Slope of Regression line becomes very small and we can predict Lamda value by Cross Validation technique.
  - **As lambda increases, the penalty term becomes larger, and the model coefficients are shrunk more.**

![](https://algotech.netlify.com/img/ridge-lasso/ridge_lasso_ffff5.png)
  - It adds an L2 penalty which is equal to the square of the magnitude of coefficients. For example, Ridge regression and SVM implement this method.
  - L2 regularization punishes big number more due to squaring. Of course, L2 is more 'elegant' in smoothness manner.
  - Even in a situation where you might benefit from L1's sparsity in order to do feature selection, **using L2 on the remaining variables is likely to give better results than L1 by itself.**
![ezgif com-gif-maker (1)](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/be2dae9d-0d60-43b1-bc86-2e98040f9d47)


#### L1 regularization Or Lasso Regression
  - To Watch Josh parmer Tutorial for Ridge Regression Click [Here](https://www.youtube.com/watch?v=NGf0voTMlcs)
  - Sparsity Inducing technique.
  - Lasso Regression is same as Ridge Regression but it adds the absolute value of Slope instead of Squaring the Slope.
  -  **Lasso Regression= (Sum of Squared Residuals)+ ( Lamda * |Slope| )**
  - It is Very Usefull when we have lot of useless variables because It can shrink all the way upto 0.
  - **Sparsity inducing** refers to the ability of a regularization technique to force the model coefficients to be zero or close to zero. This can be useful when there are many features in the dataset, and only a few of them are actually important for predicting the target variable. Sparsity inducing can help to reduce the complexity of the model, and improve its interpretability.\
  -  L1 regularization, also known as Lasso regression, is a sparsity inducing regularization technique. It works by adding a penalty to the model objective function that is proportional to the absolute value of the model coefficients. This penalty encourages the model coefficients to be zero or close to zero.

It adds an L1 penalty that is equal to the absolute value of the magnitude of coefficient, or simply restricting the size of coefficients. For example, Lasso regression implements this method. 
  - L1 regularization is used for sparsity. This can be beneficial especially if you are dealing with big data as L1 can generate more compressed models than L2 regularization. This is basically due to as regularization parameter increases there is a bigger chance your optima is at 0.
  - If you would like to identify important features. For most cases, L1 regularization does not give higher accuracy but may be slower in training.
  - So in shorts L1 regularization works best for feature selection in sparse feature spaces.

![ezgif com-gif-maker (2)](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/7f5435c3-4790-4c5c-97c9-affa1dcda107)



#### Elastic Net: 
When L1 and L2 regularization combine together, it becomes the elastic net method, it adds a hyperparameter.
  - **Elastic Net= (Sum of Squared Residuals)+ ( Lamda1 * |Slope| ) + ( Lamda2 * Slope^2 )**  
  -  when lamda1=0, and lamda2>0,Ridge Regression, viceversa, Lasso Rtegression.
  -  when both Lamda>0, Hybrid of 2 Regularisation ie, Elastic Net
  -  we use Elastic Net when we have ton of vairbles , and we dont know which are useful and which are useless variables in those.
  -  Elastic Net regularization is a combination of L1 and L2 regularization. It works by adding a penalty to the model objective function that is a linear combination of the L1 and L2 penalties. This allows the model to balance the benefits of sparsity inducing and weight sharing.
  -  Sparsity inducing and weight sharing are both important techniques for regularizing machine learning models. They can help to improve the performance of the models, and make them more interpretable.
  
![ezgif com-gif-maker (3)](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/a95a5659-5b08-4c87-af5a-9082b9c66bb3)


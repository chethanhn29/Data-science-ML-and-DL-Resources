
- [Machine learning learning Interview questions](https://aman.ai/primers/ai/interview/#deep-learning)
- [Machine Learning  Notes](https://github.com/praj2408/Machine-Learning-Notes/tree/main)
- [DS Interview Questions1](https://www.youtube.com/watch?v=dBvjBwga8pU&list=PLKnIA16_RmvZTD2oti9S1jDTX7xTC7PSO)
- [Data Science Interview Preparation](https://www.youtube.com/watch?v=ZcYVdmL8pzI&list=PLKnIA16_RmvZTD2oti9S1jDTX7xTC7PSO&index=2)
- [Machine Learning Algorithms Notes  ](https://github.com/piyushpathak03/Machine-learning-algorithm-PDF/tree/main)
## Hands on Machine learning with scikit learn,keras and tensorflow
  
  - You studied the data.
  - You selected a model.
  - You trained it on the training data (i.e., the learning algorithm searched
  - for the model parameter values that minimize a cost function).
  - Finally, **you applied the model to make predictions on new cases (this is
  called inference),** hoping that this model will generalize well.

## [ML Interview notes](https://vinija.ai/concepts/fundamentals/)
### Main Challenges of Machine Learning
####  Bad data
- insufficient of data
- **data matters more than algorithms for complex problems and all algorithms give same results if we have the sufficient data**
- Nonrepresentative Training Data , the training data should contain all generalized data.
- Poor-Quality Data like noise, Outliers,Missing values
- Irrelevant Features
    - **Feature engineering**, involves the following steps:
        - Feature selection (selecting the most useful features to train on among
existing features)
        - Feature extraction (combining existing features to produce a more
          useful one —as we saw earlier, dimensionality reduction algorithms
          can help)
        - Creating new features by gathering new data
####  Bad Model
- **Overfitting** the Training Data :overfitting: it means that the model performs
well on the training data, but it does not generalize well.

- **Underfitting** the Training Data
As you might guess, underfitting is the opposite of overfitting: it occurs when
your model is too simple to learn the underlying structure of the data

Here are the main options for fixing this problem:
    - Select a more powerful model, with more parameters.
    - Feed better features to the learning algorithm (feature engineering).
    - Reduce the constraints on the model (for example by reducing the
    regularization hyperparameter)
### Testing and Validating
The only way to know how well a model will generalize to new cases is to
actually try it out on new cases

### Hyperparameter Tuning and Model Selection
The problem is that you measured the generalization error multiple times on
the test set, and you adapted the model and hyperparameters to produce the
best model for that particular set. This means the model is unlikely to
perform as well on new data.
A common solution to this problem is called holdout validation (Figure 1-
25): you simply hold out part of the training set to evaluate several candidate
models and select the best one. The new held-out set is called the validation
set (or the development set, or dev set). 

**More specifically, you train multiple
models with various hyperparameters on the reduced training set (i.e., the full
training set minus the validation set), and you select the model that performs
best on the validation set. After this holdout validation process, you train the
best model on the full training set (including the validation set), and this
gives you the final model. Lastly, you evaluate this final model on the test set
to get an estimate of the generalization error**

### Cross Validation
cross-validation, using many small validation sets. Each model is
evaluated once per validation set after it is trained on the rest of the data.
 By
averaging out all the evaluations of a model, you get a much more accurate
measure of its performance. **There is a drawback, however: the training time
is multiplied by the number of validation sets.**


## Chapter 2. End-to-End MachineLearning Project
      1. Look at the big picture.
        - Frame the Problem(Business Objective)
        - Type of Ml(Supervised,Unsupervised,semisupervised,Self supervised, Reinforcement learning)
        - Regression, Classification or Clustering or type of task.
        - Batch learning(Training once) or Online learning(Incremental training)
        - Select Performance Measire.
      2. Get the data.
      3. Explore and visualize the data to gain insights.
      4. Prepare the data for machine learning algorithms.
      5. Select a model and train it.
      6. Fine-tune your model.
      7. Present your solution.
      8. Launch, monitor, and maintain your system.
      
### PIPELINES
A sequence of data processing components is called a data pipeline.

Pipelines are very common in machine learning systems, since there is a
lot of data to manipulate and many data transformations to apply.
Components typically run asynchronously. **Each component pulls in a
large amount of data, processes it, and spits out the result in another data
store. Then, some time later, the next component in the pipeline pulls in
this data and spits out its own output.** Each component is fairly selfcontained: the interface between components is simply the data store.
This makes the system simple to grasp (with the help of a data flow
graph), and different teams can focus on different components. Moreover,
if a component breaks down, the downstream components can often
continue to run normally (at least for a while) by just using the last output
from the broken component. This makes the architecture quite robust.

### RMSE
A typical performance measure for regression problems is the root mean square error (RMSE). **It
gives an idea of how much error the system typically makes in its predictions,
with a higher weight given to large errors**

**Both the RMSE and the MAE are ways to measure the distance between two
two vectors: the vector of predictions and the vector of target(Actual) values. Various
distance measures, or norms, are possible:**

## [Best Evaluation Metrics for Your Regression Model](https://www.analyticsvidhya.com/blog/2021/05/know-the-best-evaluation-metrics-for-your-regression-model/)
  
####  Mean Absolute Error(MAE) - Mahatten distance

MAE is a very simple metric which calculates the absolute difference between actual and predicted values.
Advantages of MAE

   -  The MAE you get is in the same unit as the output variable.
  - It is most Robust to outliers.

#### Mean Squared Error(MSE)
MSE is a most used and very simple metric with a little bit of change in mean absolute error. Mean squared error states that finding the squared difference between actual and predicted value.

    - If you have outliers in the dataset then it penalizes the outliers most and the calculated MSE is bigger. So, in short, It is not Robust to outliers which were an advantage in MAE.

#### Root Mean Squared Error(RMSE) - Euclidean distance
As RMSE is clear by the name itself, that it is a simple square root of mean squared error.
  - This is why the RMSE is more sensitive to outliers than the
  MAE.
  - But when outliers are exponentially rare (like in a bell-shaped curve),
  the RMSE performs very well and is generally preferred.
##### Disadvantages of RMSE
It is not that robust to outliers as compared to MAE.

### R Squared (R2)
**R2 score is a metric that tells the performance of your model, not the loss in an absolute sense that how many wells did your model perform.**

So, with help of R squared we have a baseline model to compare a model which none of the other metrics provides. The same we have in classification problems which we call a threshold which is fixed at 0.5. **So basically R2 squared calculates how must regression line is better than a mean line.**
Hence, R2 squared is also known as Coefficient of Determination or sometimes also known as **Goodness of fit.**

![](https://editor.analyticsvidhya.com/uploads/22091R2%20Squared%20Formula.png)

Now, how will you interpret the R2 score? suppose If the R2 score is zero then the above regression line by mean line is equal means 1 so 1-1 is zero. So, in this case, both lines are overlapping means model performance is worst, It is not capable to take advantage of the output column.

Now the second case is when the R2 score is 1, it means when the division term is zero and it will happen when the regression line does not make any mistake, it is perfect. In the real world, it is not possible.

So we can conclude that as our regression line moves towards perfection, R2 score move towards one. And the model performance improves.

The normal case is when the R2 score is between zero and one like 0.8 which means your model is capable to explain 80 per cent of the variance of data.
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(r2)

```
### What is R²?
R² is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. Essentially, it indicates how well the data points fit the regression line.

### Interpreting R²

1. **R² = 0**:
   - When the R² score is zero, it means that the regression line does not explain any of the variance in the dependent variable.
   - This is equivalent to the mean line (the horizontal line at the average value of the dependent variable) being just as good at predicting the dependent variable as the regression line.
   - **Implication**: The model's performance is poor, and it is not capable of capturing the relationship between the independent and dependent variables. The regression line overlaps with the mean line.

2. **R² = 1**:
   - An R² score of one indicates that the regression line perfectly explains all the variance in the dependent variable.
   - This occurs when the regression line passes through every data point without error.
   - **Implication**: The model's performance is perfect. However, in real-world scenarios, this is highly unlikely because there is almost always some level of noise or variability that the model cannot capture.

3. **0 < R² < 1**:
   - When the R² score is between zero and one, it indicates that the regression line explains some but not all of the variance in the dependent variable.
   - **Implication**: The closer the R² score is to one, the better the model is at predicting the dependent variable. Conversely, the closer it is to zero, the worse the model is.

### Summary
- **R² = 0**: The model does not explain any of the variance in the dependent variable. It is as good as predicting with the mean.
- **R² = 1**: The model perfectly explains all the variance in the dependent variable, which is an ideal but usually unrealistic scenario.
- **0 < R² < 1**: The model explains some variance in the dependent variable, and the degree to which it does so improves as R² approaches one.

### Conclusion
- As the R² score moves towards one, the regression line better fits the data points, indicating improved model performance.
- A higher R² score signifies that the model is more capable of capturing the variability in the dependent variable, making better predictions.

### Adjusted R Squared
The disadvantage of the R2 score is while adding new features in data the R2 score starts increasing or remains constant but it never decreases because It assumes that while adding more data variance of data increases.

But the problem is when we add an irrelevant feature in the dataset then at that time R2 sometimes starts increasing which is incorrect.

Hence, To control this situation Adjusted R Squared came into existence.

![](https://lh3.googleusercontent.com/-6T1LxrK1by8/YB6D5hjSCjI/AAAAAAAAAlk/gCmLpEJMJ3MpwO6r-sI7GQzuOQP2I1B3QCLcBGAsYHQ/w332-h179/image.png)

Now as K increases by adding some features so the denominator will decrease, n-1 will remain constant. R2 score will remain constant or will increase slightly so the complete answer will increase and when we subtract this from one then the resultant score will decrease. so this is the case when we add an irrelevant feature in the dataset.

And if we add a relevant feature then the R2 score will increase and 1-R2 will decrease heavily and the denominator will also decrease so the complete term decreases, and on subtracting from one the score increases.

n=40
k=2
adj_r2_score = 1 - ((1-r2)*(n-1)/(n-k-1))
print(adj_r2_score)
Hence, this metric becomes one of the most important metrics to use during the evaluation of the model.
[How to Calculate Adjusted R squared](https://stackoverflow.com/questions/49381661/how-do-i-calculate-the-adjusted-r-squared-score-using-scikit-learn)
```python
import numpy as np
from sklearn.metrics import r2_score

def r2(actual: np.ndarray, predicted: np.ndarray):
    """ R2 Score """
    return r2_score(actual, predicted)

def adjr2(actual: np.ndarray, predicted: np.ndarray, rowcount: np.int, featurecount: np.int):
    """ R2 Score """
    return 1-(1-r2(actual,predicted))*(rowcount-1)/(rowcount-featurecount)
```

| Metric                  | Purpose                                 | Advantages                                 | Disadvantages                              | When to Use                            | What It Depends On             | Remarks                                                                                                                                                                     |
|-------------------------|-----------------------------------------|--------------------------------------------|--------------------------------------------|----------------------------------------|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mean Absolute Error (MAE) | Measures the average absolute difference between predicted and actual values. | - Easy to interpret. - Resistant to outliers. | - Not sensitive to magnitude. - May not reflect the impact of outliers. | When you want to understand the average prediction error in the same units as the target variable. | - Scale of the data.            | MAE gives equal weight to all errors, which may not be suitable if some errors are more critical than others.                                                                                                                    |
| Mean Squared Error (MSE)  | Measures the average squared difference between predicted and actual values. | - Emphasizes larger errors. - Mathematically convenient. | - Sensitive to outliers. - Error is not in the original units. | When you want to penalize larger errors more strongly or when optimizing models using gradient-based methods. | - Scale of the data.            | MSE gives more weight to larger errors, which can be useful if you want to prioritize reducing big mistakes.                                                                                                                          |
| Root Mean Squared Error (RMSE) | Measures the square root of MSE, providing an error metric in the same units as the target variable. | - In the original units of the target. - Emphasizes larger errors. | - Sensitive to outliers. - Harder to interpret than MAE. | When you want to understand the average prediction error in the same units as the target variable. | - Scale of the data.            | RMSE is a commonly used metric in regression, providing a balance between MSE and MAE.                                                                                                                                                |
| R-squared (R²)             | Measures the proportion of variance in the target variable explained by the model. | - Provides a measure of goodness of fit. - Range between 0 and 1. | - Doesn't tell you about the quality of individual predictions. - Can be misleading with complex models. | When you want to understand how well the model explains the variance in the data. | - The choice of predictor variables. | R² is a relative metric, and a high R² doesn't necessarily mean the model makes accurate predictions. It may not be suitable for models with complex relationships.                                                      |
| Adjusted R-squared         | An adjusted version of R² that accounts for the number of predictors in the model. | - Penalizes adding irrelevant predictors. | - Still doesn't address model complexity fully. | When you want to assess model fit while considering the number of predictors. | - The choice of predictor variables. | Adjusted R-squared helps mitigate the issue of R² increasing with the addition of more predictors, but it has its limitations.                                                                                                        |
| Mean Absolute Percentage Error (MAPE) | Measures the average percentage difference between predicted and actual values. | - Provides a percentage error interpretation. - Applicable for relative error assessment. | - Undefined when actual values are zero. - Can lead to division by zero. | When you want to assess prediction errors as a percentage of the actual values, common in forecasting tasks. | - Actual values.                 | MAPE is suitable for scenarios where relative errors are more meaningful than absolute errors, but it has limitations when actual values are close to zero.                                                                  |

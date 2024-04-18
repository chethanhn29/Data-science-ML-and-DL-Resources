# Table of Contents

1. [A simpler way to understand what the bias is](#simpler-understanding-bias)
2. [Why do we need Bias in Machine learning](#need-bias-ml)
3. [Errors in Machine learning](#errors-ml)
4. [Bias](#bias)
5. [Ways to reduce High Bias](#reduce-high-bias)
6. [Variance](#variance)
7. [Ways to Reduce High Variance](#reduce-high-variance)
8. [Bias and Variance and its Tradeoff](#bias-variance-tradeoff)

---

## 1. A simpler way to understand what the bias is <a name="simpler-understanding-bias"></a>

It is somehow similar to the constant b of a linear function:

\[
y = ax + b
\]

It allows you to move the line up and down to fit the prediction with the data better. Without \( b \), the line always goes through the origin \((0, 0)\) and you may get a poorer fit. The bias is not an NN term. It's a generic algebra term to consider.

\[
Y = M*X + C \quad \text{(straight line equation)}
\]

Now if \( C \) (Bias) = 0 then, the line will always pass through the origin, i.e. \((0,0)\), and depends on only one parameter, i.e. \( M \), which is the slope so we have less things to play with. If we ignore the bias, many inputs may end up being represented by a lot of the same weights (i.e. the learnt weights mostly occur close to the origin \((0,0)\)). The model would then be limited to poorer quantities of good weights, instead of the many more good weights it could better learn with bias. (Where poorly learnt weights lead to poorer guesses or a decrease in the neural net’s guessing power).

## 2. Why do we need Bias in Machine learning <a name="need-bias-ml"></a>

- If you see the left panel in which we have drawn a Regression Line without bias i.e., The line passing through the origin (\( Y = mx \)). Hence we have no control on our regression line which is best fit to the data points.
- In the right panel, we added bias "C" hence our equation of regression line becomes (\( Y = mx + C \)) which passes through point \((0,C)\) on the y-axis.
- This gives more control to identify the line of best fit by varying different "m" and "C". Hence this underlines bias gives you more control over the line of best fit.
- Example: In the case of a house price model, the bias term would represent the price of a house with zero bedrooms. This is important because even a house with no bedrooms will have some value. For example, it could be used as a storage unit or a workshop. The bias term allows the model to account for this value.

Without the bias term, the model would only be able to learn the relationship between the number of bedrooms and the price of a house. It would not be able to learn the value of a house with zero bedrooms. This would result in the model predicting a negative price for houses with zero bedrooms.

- Bias can help to prevent overfitting. Overfitting occurs when a neural network learns the training data too well and is unable to generalize to new data. Bias can help to prevent overfitting by adding a degree of noise to the model.
- Bias can help to improve the accuracy of the model. By adding bias, the model is able to learn more complex relationships between the input data and the output data.
- Bias can help to speed up the training process. By adding bias, the model is able to converge on a solution more quickly.

![Regression Line with and without Bias](https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2017-06-29-at-5.34.44-PM.png)

![Bias Addition Example](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRF1osE7Ez3z_YE-9Dnyl0623qqrm7xVPQEPGDlXBDc9lPhrbhb7UYiCiPF3fa8IYodNA&usqp=CAU)

## 3. Errors in Machine learning <a name="errors-ml"></a>

In machine learning, an error is a measure of how accurately an algorithm can make predictions for the previously unknown dataset. On the basis of these errors, the machine learning model is selected that can perform best on the particular dataset. There are mainly two types of errors in machine learning, which are:

**Reducible errors**: These errors can be reduced to improve the model accuracy. Such errors can further be classified into bias and variance.
and **Noise** is the irreducible error.

## 4. Bias <a name="bias"></a>

In general, a machine learning model analyzes the data, finds patterns in it, and makes predictions. While training, the model learns these patterns in the dataset and applies them to the test data for prediction.

**While making predictions, a difference occurs between prediction values made by the model and actual values/expected values, and this difference is known as bias errors or Errors due to bias.**

It can be defined as an inability of machine learning algorithms such as Linear Regression to capture the true relationship between the data points. Each algorithm begins with some amount of bias because bias occurs from assumptions in the model, which makes the target function simple to learn. A model has either:

- **Low Bias**: A low bias model will make fewer assumptions about the form of the target function. -- Good Model
- **High Bias**: A model with high bias makes more assumptions, and the model becomes unable to capture the important features of our dataset. A high bias model also cannot perform well on new data.

### Algorithms which have low and high bias

- Some examples of machine learning algorithms with low bias are Decision Trees, k-Nearest Neighbours, and Support Vector Machines. 
- At the same time, an algorithm with high bias is Linear Regression, Linear Discriminant Analysis, and Logistic Regression.

### Ways to reduce High Bias:

High bias mainly occurs due to a much simple model. Below are some ways to reduce high bias:

- Increase the input features as the model is underfitted.
- Decrease the regularization term.
- Use more complex models, such as including some polynomial features.

## 5. Ways to reduce High Bias <a name="reduce-high-bias"></a>

- Increase the input features as the model is underfitted.
- Decrease the regularization term.
- Use more complex models, such as including some polynomial features.

## 6. Variance <a name="variance"></a>

The variance would specify the amount of variation in the prediction if different training data were used.

- **Low variance** means there is a small variation in the prediction of the target function with changes in the training data set. 
- **High variance** shows a large variation in the prediction of the target function with changes in the training dataset

.
- A model that shows high variance learns a lot and performs well with the training dataset, and does not generalize well with the unseen dataset. As a result, such a model gives good results with the training dataset but shows high error rates on the test dataset.
- Since, with high variance, the model learns too much from the dataset, it leads to overfitting of the model. A model with high variance has the below problems:

  - A high variance model leads to overfitting.
  - Increase model complexities.

- Some examples of machine learning algorithms with low variance are Linear Regression, Logistic Regression, and Linear discriminant analysis.
- At the same time, algorithms with high variance are decision tree, Support Vector Machine, and K-nearest neighbors.

### Ways to Reduce High Variance:

- Reduce the input features or the number of parameters as a model is overfitted.
- Do not use a much complex model.
- Increase the training data.
- Increase the Regularization term.

## 7. Bias and Variance and its Tradeoff <a name="bias-variance-tradeoff"></a>

For an accurate prediction of the model, algorithms need low variance and low bias. But this is not possible because bias and variance are related to each other:

- If we decrease the variance, it will increase the bias.
- If we decrease the bias, it will increase the variance.

**1. Low-Bias, Low-Variance:**

The combination of low bias and low variance shows an ideal machine learning model. However, it is not possible practically.

**2. Low-Bias, High-Variance:**

With low bias and high variance, model predictions are inconsistent and accurate on average. This case occurs when the model learns with a large number of parameters and hence leads to overfitting.

**3. High-Bias, Low-Variance:**

With high bias and low variance, predictions are consistent but inaccurate on average. This case occurs when a model does not learn well with the training dataset or uses few numbers of parameters. It leads to underfitting problems in the model.

**4. High-Bias, High-Variance:**

With high bias and high variance, predictions are inconsistent and also inaccurate on average.

![Bias and Variance Tradeoff](https://i.stack.imgur.com/6Y87l.png)

---

With everything organized and a table of contents added, it should be easier for readers to navigate through the document.

## Read this blog by Andrew Karapathy [Blog](https://karpathy.github.io/2019/04/25/recipe/)


# Neural Network Development: Common Mistakes

Neural networks are powerful tools for various machine learning tasks, but they are prone to specific errors that can hinder their performance. Being aware of these common mistakes is essential for effectively designing, training, and deploying neural networks.

## List of Common Mistakes:

1. **Not attempting to overfit a single batch**: Overfitting a single batch helps ensure that the model can memorize and fit the training data, serving as a sanity check before training on the entire dataset.

### 1. Overfitting a Single Batch:
```python
Copy code
# Train the model on a single batch to check fitting capability
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx == 0:  # Only train on the first batch
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        break
```

2. **Forgetting to toggle train/eval mode**: Switching between training and evaluation modes in the neural network is crucial to obtain accurate predictions.
  
### 2. Toggle Train/Eval Mode:
```python
# Switch to evaluation mode during model evaluation
model.eval()
with torch.no_grad():
    # Evaluation code here
```

3. **Failing to zero the gradient before backward pass**: Forgetting to zero gradients (`optimizer.zero_grad()`) before backpropagation can lead to incorrect gradient updates.
### 3. Zeroing Gradient before Backward Pass:
```python
# Zero gradients before backpropagation
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```
4. **Passing softmaxed outputs to loss instead of raw logits**: Certain loss functions expect raw logits rather than probabilities obtained from softmax, leading to unexpected or incorrect optimization steps during training.

### 4. Handling Softmax Output in Loss Calculation:
```python
# Use log_softmax in combination with the negative log likelihood loss
output = model(data)
log_softmax_output = F.log_softmax(output, dim=1)
loss = F.nll_loss(log_softmax_output, target)
```

5. **Using inappropriate activation functions**: Inappropriate activation functions can hinder learning, especially using sigmoid in deeper networks leading to vanishing gradients.

6. **Ignoring data preprocessing**: Not properly preprocessing data can negatively impact the model's learning process.

7. **Neglecting class imbalances**: Failing to address imbalanced class distributions in classification tasks can result in biased models.

8. **Inadequate model evaluation**: Relying solely on training metrics and not properly validating or testing the model on unseen data can lead to overfitting or an inaccurate assessment of the model's performance.

9. **Overfitting due to complex model architectures**: Using overly complex models for simpler tasks can cause overfitting, reducing the model's ability to generalize well on unseen data.

10. **Incorrect hyperparameter tuning**: Poor choices in hyperparameters can hinder the model's learning process and convergence.

11. **Lack of regularization techniques**: Not applying regularization techniques can lead to overfitting and reduced generalization.

12. **Ignoring the importance of the bias term**: Neglecting to include or initialize biases in neural network layers can hinder the model's fitting capacity.

13. **Failure to monitor and debug**: Not continuously monitoring training runs or debugging for issues can result in inefficient model learning.

14. **Ignoring vanishing or exploding gradients**: Failing to address issues with gradients during backpropagation can hinder convergence.

15. **Insufficient data augmentation**: Not employing proper data augmentation techniques can limit the model's generalization capabilities in computer vision tasks.

16. **Inadequate learning rate scheduling**: Not using effective learning rate scheduling strategies can affect convergence.

17. **Not considering the model capacity**: Using models with insufficient or excessive capacity can lead to suboptimal performance.

18. **Ignoring early stopping techniques**: Neglecting early stopping mechanisms can result in overfitting or unnecessary computation.

19. **Not using transfer learning effectively**: Overlooking the benefits of transfer learning can limit the model's performance.



5. Using Inappropriate Activation Functions:
```python
# Choosing appropriate activation functions based on the network architecture
import torch.nn as nn

# Example: Using ReLU activation function
activation = nn.ReLU()
# Include in model architecture: e.g., within a neural network layer
layer = nn.Linear(input_size, output_size)
# ...
output = activation(layer(input_data))
```
6. Data Preprocessing:
```python
# Data preprocessing steps (e.g., normalization, scaling, handling missing values)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit on training data
scaler.fit(train_data)
# Transform training and test data
train_data_normalized = scaler.transform(train_data)
test_data_normalized = scaler.transform(test_data)
```
7. Handling Class Imbalances:
```python
# Addressing class imbalance using weighted loss functions or resampling techniques
from torch.utils.data import WeightedRandomSampler

# Create a weighted sampler to address class imbalance
weights = calculate_weights_for_classes(dataset)
sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
```
8. Model Evaluation:
```python
# Properly evaluating the model on validation or test data
model.eval()
with torch.no_grad():
    # Forward pass and evaluation code here
```
9. Overfitting Due to Complex Models:
```python
# Regularization techniques like dropout to prevent overfitting
import torch.nn as nn

# Add dropout layer in the model architecture
dropout = nn.Dropout(p=0.5)  # Example dropout rate
# Include in the model architecture, e.g., before or after a layer
layer_with_dropout = nn.Sequential(nn.Linear(input_size, hidden_size), dropout, nn.ReLU())
```
10. Hyperparameter Tuning:
```python
# Using grid search or random search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Example: Grid search for hyperparameters
param_grid = {'parameter1': [value1, value2], 'parameter2': [value3, value4]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(train_data, train_labels)
best_params = grid_search.best_params_
```
20. **Disregarding hardware limitations**: Not considering hardware constraints can lead to issues during model deployment or training.

---





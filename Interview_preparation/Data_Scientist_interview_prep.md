### [Descriptive and inferential Statistics](https://careerfoundry.com/en/blog/data-analytics/inferential-vs-descriptive-statistics/)

### [Books to study Statistics](https://www.kaggle.com/discussions/general/205585)

#### Types of Distibutions
![](https://cdn-images-1.medium.com/max/747/1*cr9_-ts4vqVBOttf-EVuQQ.png)


#### [Siginificance testing](https://www.westga.edu/academics/research/vrc/assets/docs/tests_of_significance_notes.pdf)
- [best Notes](https://home.csulb.edu/~msaintg/ppa696/696stsig.htm)
- [Khan Academy Practice with theory](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample)

###  Measures in a variable
**Measures of Central Tendency:**

1. **Mean (Arithmetic Average):**
   - **When to Use:** Use the mean when the dataset is symmetrically distributed and does not contain extreme outliers.
   - **Advantages:** Provides a precise measure of central tendency, suitable for parametric statistical analyses.
   - **Disadvantages:** Sensitive to outliers, may not accurately represent the central value if the data is skewed or contains extreme values.
   - **When Not to Use:** Avoid using the mean when the dataset is heavily skewed or contains extreme outliers, as it may distort the central tendency measure.

2. **Median:**
   - **When to Use:** Use the median when the dataset is skewed or contains outliers, as it is less affected by extreme values.
   - **Advantages:** Robust to outliers, provides a reliable measure of central tendency even in the presence of skewed data.
   - **Disadvantages:** May not provide a precise estimate of central tendency for symmetrically distributed data.
   - **When Not to Use:** While the median is suitable for skewed or non-normally distributed data, it may not accurately represent central tendency in datasets with symmetrical distributions and no outliers.

3. **Mode:**
   - **When to Use:** Use the mode for categorical or discrete data to identify the most common value(s) in the dataset.
   - **Advantages:** Useful for identifying predominant values in categorical data, simple to compute.
   - **Disadvantages:** May not exist or be unique in continuous datasets, does not provide information about the spread or variability of data.
   - **When Not to Use:** Avoid using the mode as the sole measure of central tendency for continuous datasets, especially when variability is important.

**Measures of Dispersion:**

1. **Range:**
   - **When to Use:** Use the range for quick assessment of the spread of data, especially in exploratory data analysis.
   - **Advantages:** Simple to calculate, provides a straightforward measure of spread.
   - **Disadvantages:** Sensitive to outliers, does not consider the distribution of values within the dataset.
   - **When Not to Use:** Avoid using the range as the sole measure of spread for datasets with extreme outliers, as it may not accurately represent the variability of the data.

2. **Interquartile Range (IQR):**
   - **When to Use:** Use the IQR to measure the spread of the central portion of the data, especially when outliers are present.
   - **Advantages:** Robust to outliers, provides a measure of spread that focuses on the middle 50% of the data.
   - **Disadvantages:** Ignores information about the tails of the distribution, may not fully capture the variability of the entire dataset.
   - **When Not to Use:** Avoid using the IQR as the sole measure of spread when information about the entire distribution is needed.

3. **Variance and Standard Deviation:**
   - **When to Use:** Use variance and standard deviation to quantify the overall spread of data points around the mean.
   - **Advantages:** Provide precise measures of dispersion, widely used in statistical analyses.
   - **Disadvantages:** Sensitive to outliers, variance is not in the original units of the data.
   - **When Not to Use:** Avoid using variance and standard deviation when the dataset contains extreme outliers or when interpretability in the original units of the data is required.

4. **Mean Absolute Deviation (MAD):**
   - **When to Use:** Use MAD when you want a measure of dispersion that is less sensitive to outliers compared to variance.
   - **Advantages:** Robust to outliers, provides a measure of dispersion in the original units of the data.
   - **Disadvantages:** Does not account for the squared differences, which may not fully capture the variability of the data.
   - **When Not to Use:** Avoid using MAD as the sole measure of dispersion when precise quantification of variability is necessary, as it may underestimate the true spread of the data.

Each measure of central tendency and dispersion has its own strengths and weaknesses, and the choice of which to use depends on the specific characteristics of the dataset and the goals of the analysis.

### Measures of Spread
The spread in data is the measure of how far the numbers in a data set are away from the mean or the median. or It Descibes how similar or varied data points for particular variable.

**Measures of Spread in Data**

1. **Range:**
   - **Definition:** The range is the difference between the maximum and minimum values in a dataset.
   - **Calculation:** \( \text{Range} = \text{Max} - \text{Min} \)
   - **Interpretation:** It provides a simple measure of the spread but is sensitive to outliers.

2. **Interquartile Range (IQR):**
   - **Definition:** IQR is the range covered by the middle 50% of the data. or In descriptive statistics, the interquartile range (IQR) is a measure of the spread of data. It is the difference between the 75th and 25th percentiles of the data. or The interquartile range (IQR) is the range of values that resides in the middle of the scores
   - **Calculation:** \( \text{IQR} = Q3 - Q1 \), where \( Q3 \) is the third quartile (75th percentile) and \( Q1 \) is the first quartile (25th percentile).
   - **Interpretation:** IQR is robust to outliers and provides a measure of the spread of the central portion of the data.
![](https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2019/01/boxplot_pdf.png?resize=437%2C328&ssl=1)
3. **Mean Absolute Deviation (MAD):**
   - **Definition:** MAD is the average absolute difference between each data point and the mean.
   - **Calculation:** \( \text{MAD} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \text{mean}| \)
   - **Interpretation:** It provides a measure of dispersion that is less sensitive to outliers compared to variance.

4. **Variance:**
   - **Definition:** Variance measures the average of the squared differences between each data point and the mean.
   - **Calculation:** \( \text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \text{mean})^2 \)
   - **Interpretation:** It quantifies the overall spread of the data but is sensitive to outliers due to squaring the differences.

5. **Standard Deviation:**
   - **Definition:** Standard deviation is the square root of the variance.
   - **Calculation:** \( \text{Standard deviation} = \sqrt{\text{Variance}} \)
   - **Interpretation:** It provides a measure of the average distance of data points from the mean, in the same units as the original data.

6. **Coefficient of Variation (CV):**
   - **Definition:** CV measures the relative variability of the data compared to the mean.
   - **Calculation:** \( \text{CV} = \frac{\text{Standard deviation}}{\text{Mean}} \times 100\% \)
   - **Interpretation:** It allows for comparison of variability between datasets with different means.

7. **Mean Squared Error (MSE) or Mean Squared Deviation (MSD):**
   - **Definition:** MSE is the average of the squared differences between observed and predicted values in regression analysis.
   - **Calculation:** \( \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \)
   - **Interpretation:** It quantifies the average discrepancy between observed and predicted values in regression models.

#### Differences between variance and standard deviation:

|   | Variance | Standard Deviation |
|---|----------|--------------------|
| **Definition** | The average of the squared differences between each data point and the mean. | The square root of the variance. |
| **Calculation** | \( \text{Variance} = \frac{\sum_{i=1}^{n} (x_i - \text{mean})^2}{n} \) | \( \text{Standard deviation} = \sqrt{\text{Variance}} \) |
| **Units** | Square of the original units of the data. | Same units as the original data. |
| **Sensitivity to Outliers** | Sensitive to outliers due to squaring the differences. | Sensitive to outliers but less than variance due to square root operation. |
| **Interpretation** | Measures the average spread of data points from the mean. | Provides a measure of the average distance of data points from the mean, in the same units as the original data. |
| **Advantages** | Provides a precise measure of dispersion. | More interpretable as it is in the same units as the original data. |
| **Disadvantages** | Not in the original units of the data, sensitive to outliers. | Same as variance but less sensitive to outliers. |

### **[Covariance and Correlation](https://www.analyticsvidhya.com/blog/2023/07/covariance-vs-correlation/):**

1. **Covariance:**
   - **Definition:** Covariance measures the degree to which two variables change together. A positive covariance indicates that the variables tend to move in the same direction, while a negative covariance indicates they move in opposite directions.  or Cova**riance implies whether the two variables are directly or inversely proportional.**
   - **Calculation:** Covariance between variables \( X \) and \( Y \) is calculated as:
     \[ \text{Cov}(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})}{n} \]
   - **Interpretation:** A large positive covariance indicates a strong positive relationship, while a large negative covariance indicates a strong negative relationship. A covariance close to zero suggests little to no linear relationship between the variables.
   - **Advantages:** Provides insight into the direction of the relationship between variables.
   - **Disadvantages:** Covariance is not standardized and depends on the scale of the variables, making it difficult to interpret.

2. **Correlation:**
   - **Definition:** Correlation is a standardized measure of the linear relationship between two variables. It ranges from -1 to 1, where 1 indicates a perfect positive linear relationship, -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.
   - **Calculation:** Pearson correlation coefficient (\( \rho \)) between variables \( X \) and \( Y \) is calculated as:
     \[ \rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y} \]
   - **Interpretation:** Correlation quantifies the strength and direction of the linear relationship between variables. A correlation coefficient close to 1 or -1 indicates a strong linear relationship, while a coefficient close to 0 indicates little to no linear relationship.
   - **Advantages:** Standardized measure, making it easier to interpret and compare across different datasets. Also, robust to differences in scale.
   - **Disadvantages:** Only captures linear relationships, may not capture non-linear associations between variables.

**Differences between Covariance and Correlation:**

|   | Covariance | Correlation |
|---|------------|-------------|
| **Definition** | Measures the degree to which two variables change together. | Standardized measure of the linear relationship between two variables. |
| **Calculation** | Covariance between variables \( X \) and \( Y \) is calculated using the formula for covariance. | Correlation coefficient (\( \rho \)) is calculated as the covariance divided by the product of the standard deviations of the variables. |
| **Range** | Unbounded, can take any real value. | Ranges from -1 to 1, inclusive. |
| **Units** | Not standardized, depends on the units of the variables. | Standardized, unitless measure. |
| **Interpretation** | Indicates the direction of the relationship between variables but not the strength. | Indicates both the strength and direction of the linear relationship between variables. |
| **Advantages** | Provides insight into the direction of the relationship between variables. | Standardized measure, easier to interpret and compare across datasets. |
| **Disadvantages** | Not standardized, difficult to interpret. | Only captures linear relationships, may not capture non-linear associations. |

**When to Use Covariance and Correlation:**

- **Covariance:** Use covariance when you want to understand the direction of the relationship between two variables. It is suitable for exploring the relationship between variables but does not provide a standardized measure.
  
- **Correlation:** Use correlation when you want to quantify the strength and direction of the linear relationship between two variables. It is useful for comparing relationships across different datasets and is robust to differences in scale.

**Why Use Covariance or Correlation:**

- **Covariance:** Covariance is useful for understanding the direction of the relationship between variables, which can help in exploratory data analysis and hypothesis generation.

- **Correlation:** Correlation provides a standardized measure of the strength and direction of the linear relationship between variables, making it easier to interpret and compare across different datasets.

Understanding the differences between covariance and correlation helps in selecting the appropriate measure for analyzing relationships between variables in different contexts.

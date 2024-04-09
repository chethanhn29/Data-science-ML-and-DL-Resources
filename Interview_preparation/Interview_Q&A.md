### [Hypothesis testing](https://towardsdatascience.com/hypothesis-tests-explained-8a070636bd28)

Hypothesis testing is a statistical method used to make inferences about a population based on sample data. The process involves formulating two competing hypotheses: the null hypothesis (H0) and the alternative hypothesis (H1). The null hypothesis typically represents the status quo or a default assumption, while the alternative hypothesis represents what the researcher is trying to prove.

The general process of hypothesis testing involves the following steps:

1. Formulate the null hypothesis (H0) and the alternative hypothesis (H1).
2. Select an appropriate statistical test based on the type of data and the research question.
3. Collect data and calculate the test statistic.
4. Determine the probability of obtaining the observed results (or more extreme results) if the null hypothesis were true.
5. Make a decision to either reject the null hypothesis in favor of the alternative hypothesis or fail to reject the null hypothesis based on the calculated probability.

### What is Hypothesis Testing in Statistics? Types and Examples

What Is Hypothesis Testing in Statistics?

**Hypothesis Testing is a type of statistical analysis in which you put your assumptions about a population parameter to the test. It is used to estimate the relationship between 2 statistical variables.**

Let's discuss few examples of statistical hypothesis from real-life - 

- A teacher assumes that 60% of his college's students come from lower-middle-class families.
- A doctor believes that 3D (Diet, Dose, and Discipline) is 90% effective for diabetic patients.

**Hypothesis Testing Formula**

Z = ( x̅ – μ0 ) / (σ /√n)

- Here, x̅ is the sample mean,
- μ0 is the population mean,
- σ is the standard deviation,
- n is the sample size.

### How Hypothesis Testing Works?
An analyst performs hypothesis testing on a statistical sample to present evidence of the plausibility of the null hypothesis. Measurements and analyses are conducted on a random sample of the population to test a theory. Analysts use a random population sample to test two hypotheses: the null and alternative hypotheses.

The null hypothesis is typically an equality hypothesis between population parameters; for example, a null hypothesis may claim that the population means return equals zero. The alternate hypothesis is essentially the inverse of the null hypothesis (e.g., the population means the return is not equal to zero). As a result, they are mutually exclusive, and only one can be correct. One of the two possibilities, however, will always be correct.


#### Null Hypothesis and Alternate Hypothesis
The Null Hypothesis is the assumption that the event will not occur. A null hypothesis has no bearing on the study's outcome unless it is rejected.

H0 is the symbol for it, and it is pronounced H-naught.

The Alternate Hypothesis is the logical opposite of the null hypothesis. The acceptance of the alternative hypothesis follows the rejection of the null hypothesis. H1 is the symbol for it.


### Types of Hypothesis Testing

![](https://miro.medium.com/v2/resize:fit:960/0*oJWBEpXPpIcZZX-K.jpg)

#### Z Test
To determine whether a discovery or relationship is statistically significant, hypothesis testing uses a z-test. It usually checks to see if two means are the same (the null hypothesis). Only when the population standard deviation is known and the sample size is 30 data points or more, can a z-test be applied.

The Z-test is a statistical method used to determine whether the mean of a sample is significantly different from a known population mean. It's particularly useful when you have a large sample size (typically more than 30) and know the population's standard deviation.

Here's a simple explanation of how it works:

1. **Formulate hypotheses**: Like in any hypothesis test, you start by stating your null hypothesis (H0) and alternative hypothesis (H1). For example, if you're testing whether a new teaching method improves test scores, your null hypothesis might be that the new method has no effect (mean test scores are the same), and the alternative hypothesis might be that the new method does have an effect (mean test scores are different).

2. **Collect data**: Gather your sample data. You need to know the sample mean, sample size, and the population standard deviation.

3. **Calculate the Z-score**: The Z-score tells you how many standard deviations your sample mean is away from the population mean. It's calculated using the formula: \( Z = \frac{{\bar{x} - \mu}}{{\frac{{\sigma}}{{\sqrt{n}}}}} \), where:
   - \( \bar{x} \) is the sample mean,
   - \( \mu \) is the population mean,
   - \( \sigma \) is the population standard deviation,
   - \( n \) is the sample size.

4. **Determine the critical value or p-value**: Based on your hypotheses and the significance level (usually denoted as α, commonly set to 0.05), you determine whether to use a one-tailed or two-tailed test and find the corresponding critical Z-value from a standard normal distribution table or use statistical software to find the p-value.

5. **Make a decision**: Compare your calculated Z-score with the critical Z-value or use the p-value. If the calculated Z-score is greater than the critical Z-value (for a one-tailed test) or falls outside the critical region (for a two-tailed test), you reject the null hypothesis and accept the alternative hypothesis. If the p-value is less than α, you reject the null hypothesis.

In simpler terms, the Z-test helps you decide if your sample's mean is different enough from what you'd expect by chance, given what you know about the population. If it's different enough, you conclude that there's something real going on (you reject the null hypothesis). If it's not different enough, you don't have enough evidence to say there's a real difference (you fail to reject the null hypothesis).

A Z-score, also known as a standard score, is a statistical measure that tells you how many standard deviations a particular value is from the mean of a dataset. It's used to compare individual data points to the average of the dataset and understand how typical or unusual a specific value is.

Here's how you calculate a Z-score for a single data point:

\[ Z = \frac{{X - \mu}}{{\sigma}} \]

Where:
- \( X \) is the value you're interested in (individual data point),
- \( \mu \) is the mean of the dataset,
- \( \sigma \) is the standard deviation of the dataset.

For example, if you have a dataset of test scores with a mean of 70 and a standard deviation of 10, and you want to find the Z-score for a test score of 80:

\[ Z = \frac{{80 - 70}}{{10}} = 1 \]

This means that the test score of 80 is 1 standard deviation above the mean.

Key points about Z-scores:

1. A Z-score of 0 means the data point is exactly at the mean.
2. Positive Z-scores indicate values above the mean, while negative Z-scores indicate values below the mean.
3. The farther a Z-score is from 0, the more unusual the data point is relative to the rest of the dataset.
4. Z-scores can be used to compare data from different distributions, as long as they're normally distributed or you have a large enough sample size for the Central Limit Theorem to apply.
5. Z-scores are often used in hypothesis testing, such as in Z-tests and in understanding standard normal distributions.


| **Aspect**                 | **Description**                                                                                                                                                                                                                          |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Advantages**             | 1. Simple to understand and apply<br>2. Well-suited for large sample sizes<br>3. Known population standard deviation<br>4. Useful for testing hypotheses about population means<br>5. Provides precise results                           |
| **When to Use**            | 1. Large sample sizes<br>2. Known population standard deviation<br>3. Comparing sample means to a population mean<br>4. Quality control processes                                                                                                                                                      |
| **Disadvantages**          | 1. Requirement of known population standard deviation<br>2. Limited applicability to small sample sizes<br>3. Sensitive to outliers                                                                                                                                                                  |
| **Other Applications**     | 1. Comparison of means<br>2. Quality control and process improvement<br>3. Financial analysis<br>4. A/B testing                                                                                                                                                                                       |

This table provides a concise overview of the key points related to Z-tests, including their advantages, disadvantages, when to use them, and other relevant applications.

#### T Test
A statistical test called a t-test is employed to compare the means of two groups. To determine whether two groups differ or if a procedure or treatment affects the population of interest, it is frequently used in hypothesis testing.

#### T Test

The t-test is a statistical method used to determine if there's a significant difference between the means of two groups. It's commonly used when dealing with small sample sizes or when the population standard deviation is unknown.

The t-test is particularly useful when:

1. **Dealing with Small Sample Sizes**: Unlike the z-test, which requires large sample sizes, the t-test can be applied even when dealing with small samples, making it more versatile in practical research settings.

2. **Population Standard Deviation is Unknown**: When the population standard deviation is unknown, the t-test offers a reliable alternative, as it estimates the population standard deviation based on the sample data.

3. **Comparing Means of Two Groups**: The t-test is often used to compare the means of two groups, such as comparing the effectiveness of two different treatments or interventions.

Here's a breakdown of how the t-test works:

1. **Formulate hypotheses**: Start by stating your null hypothesis (H0) and alternative hypothesis (H1). For example, if you're comparing the effectiveness of two teaching methods, your null hypothesis might be that there's no difference in student performance between the two methods.

2. **Collect data**: Gather data from your two groups, including sample means, sample sizes, and sample standard deviations.

3. **Calculate the t-score**: The t-score measures the difference between the two sample means relative to the variation within the groups. It's calculated using the formula:

   \[ t = \frac{{\bar{x}_1 - \bar{x}_2}}{{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}} \]

   Where:
   - \( \bar{x}_1 \) and \( \bar{x}_2 \) are the sample means of the two groups,
   - \( s_p \) is the pooled standard deviation, calculated as:
     \[ s_p = \sqrt{\frac{{(n_1 - 1)s^2_1 + (n_2 - 1)s^2_2}}{{n_1 + n_2 - 2}}} \]
   - \( n_1 \) and \( n_2 \) are the sample sizes of the two groups,
   - \( s^2_1 \) and \( s^2_2 \) are the sample variances of the two groups.

4. **Determine the critical value or p-value**: Similar to the z-test, determine the critical t-value or calculate the p-value based on the significance level (usually 0.05).

5. **Make a decision**: Compare the calculated t-score with the critical t-value or use the p-value to determine whether to reject the null hypothesis. If the calculated t-score is greater than the critical t-value or the p-value is less than the significance level, you reject the null hypothesis and conclude that there's a significant difference between the two groups.

Here's a summary of the key points related to t-tests:

| **Aspect**                 | **Description**                                                                                                                                                                                             |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Advantages**             | 1. Suitable for small sample sizes<br>2. Does not require knowledge of population standard deviation<br>3. Versatile in various research settings                                                         |
| **When to Use**            | 1. Small sample sizes<br>2. Unknown population standard deviation<br>3. Comparing means of two groups                                                                                                                                                                    |
| **Disadvantages**          | 1. Assumes normality and homogeneity of variances<br>2. Less powerful than z-test for large sample sizes<br>3. Sensitive to outliers                                                                                                                                     |
| **Other Applications**     | 1. Comparing treatment outcomes in medical research<br>2. Assessing the effectiveness of educational interventions<br>3. Quality control in manufacturing processes<br>4. Analyzing market research data |

The t-test offers a flexible and widely applicable method for comparing means of two groups, especially when dealing with small sample sizes or unknown population standard deviations. However, it's essential to consider its assumptions and limitations when applying it in practice.


#### Chi-Square 
You utilize a Chi-square test for hypothesis testing concerning whether your data is as predicted. To determine if the expected and observed results are well-fitted, the Chi-square test analyzes the differences between categorical variables from a random sample. The test's fundamental premise is that the observed values in your data should be compared to the predicted values that would be present if the null hypothesis were true.

### Chi-Square Test

The chi-square test is a statistical method used to determine whether there is a significant association between two categorical variables. It's commonly used to analyze data in the form of counts or frequencies and assess whether observed differences between groups are due to chance or if they reflect a true relationship.

The chi-square test is particularly useful when:

1. **Analyzing Categorical Data**: It's specifically designed to analyze categorical data, making it suitable for comparing proportions or frequencies across different groups or categories.

2. **Assessing Independence**: The chi-square test assesses whether there's a significant association between two categorical variables, helping researchers understand if changes in one variable are related to changes in another variable.

Here's how the chi-square test works:

1. **Formulate hypotheses**: Start by stating your null hypothesis (H0) and alternative hypothesis (H1). The null hypothesis typically assumes that there is no association between the two categorical variables, while the alternative hypothesis suggests that there is a significant association.

2. **Collect data**: Gather data in the form of counts or frequencies for the categories of the two variables you're interested in analyzing.

3. **Calculate the chi-square statistic**: The chi-square statistic measures the difference between the observed frequencies and the frequencies we would expect if the variables were independent. It's calculated using the formula:

   \[ \chi^2 = \sum{\frac{{(O - E)^2}}{{E}}} \]

   Where:
   - \( O \) is the observed frequency,
   - \( E \) is the expected frequency under the assumption of independence.

4. **Determine the critical value or p-value**: Similar to other hypothesis tests, determine the critical chi-square value from the chi-square distribution table or calculate the p-value based on the significance level (usually 0.05).

5. **Make a decision**: Compare the calculated chi-square statistic with the critical chi-square value or use the p-value to determine whether to reject the null hypothesis. If the calculated chi-square statistic is greater than the critical value or the p-value is less than the significance level, you reject the null hypothesis and conclude that there's a significant association between the variables.

Here's a summary of the key points related to chi-square tests:

| **Aspect**                 | **Description**                                                                                                                                                                    |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Advantages**             | 1. Specifically designed for analyzing categorical data<br>2. Assess association between variables without assuming linearity or normality                                           |
| **When to Use**            | 1. Analyzing categorical data<br>2. Assessing independence between two categorical variables                                                                                      |
| **Disadvantages**          | 1. Assumes independent observations and expected frequencies should not be too low<br>2. Not suitable for continuous data or small sample sizes                                      |
| **Other Applications**     | 1. Assessing the effectiveness of marketing strategies<br>2. Analyzing survey responses<br>3. Investigating genetic inheritance patterns<br>4. Evaluating the impact of educational interventions |

The chi-square test offers a valuable tool for analyzing categorical data and assessing associations between variables. However, it's essential to ensure that the assumptions of the test are met and to interpret the results cautiously, especially when dealing with small expected frequencies.

### Analysis of Variance (ANOVA) Test

The Analysis of Variance (ANOVA) test is a statistical method used to compare the means of three or more groups to determine if there are statistically significant differences between them. It's commonly used when you have multiple groups and want to assess whether any of them differ from the others.

The ANOVA test is particularly useful when:

1. **Comparing Multiple Groups**: It's specifically designed to compare the means of three or more groups simultaneously, making it efficient for analyzing complex experimental designs with multiple treatment groups.

2. **Assessing Variability**: ANOVA assesses both the within-group variability (variance within each group) and the between-group variability (variance between the group means), allowing researchers to determine if differences between groups are due to chance or if they reflect true differences.

Here's how the ANOVA test works:

1. **Formulate hypotheses**: Start by stating your null hypothesis (H0) and alternative hypothesis (H1). The null hypothesis assumes that there are no differences between the means of the groups, while the alternative hypothesis suggests that there is at least one significant difference.

2. **Collect data**: Gather data for each of the groups you want to compare. Ensure that the data meet the assumptions of ANOVA, including independence of observations, normality of data distribution within each group, and homogeneity of variances between groups.

3. **Calculate the F-statistic**: The F-statistic measures the ratio of between-group variability to within-group variability. It's calculated using the formula:

   \[ F = \frac{{MS_{between}}}{{MS_{within}}} \]

   Where:
   - \( MS_{between} \) is the mean square between groups,
   - \( MS_{within} \) is the mean square within groups.

4. **Determine the critical value or p-value**: Similar to other hypothesis tests, determine the critical F-value from the F-distribution table or calculate the p-value based on the significance level (usually 0.05).

5. **Make a decision**: Compare the calculated F-statistic with the critical F-value or use the p-value to determine whether to reject the null hypothesis. If the calculated F-statistic is greater than the critical value or the p-value is less than the significance level, you reject the null hypothesis and conclude that there are statistically significant differences between at least two of the groups.

Here's a summary of the key points related to ANOVA tests:

| **Aspect**                 | **Description**                                                                                                                                                           |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Advantages**             | 1. Designed for comparing means of three or more groups<br>2. Efficient for analyzing complex experimental designs<br>3. Provides information about both within-group and between-group variability                                 |
| **When to Use**            | 1. Comparing means of multiple groups<br>2. Assessing differences in treatment effects across different conditions or interventions                                             |
| **Disadvantages**          | 1. Assumes independence of observations and normality of data distribution<br>2. Sensitive to violations of assumptions, especially when sample sizes are unequal |
| **Other Applications**     | 1. Medical research to compare treatment outcomes across different patient groups<br>2. Market research to assess consumer preferences among multiple product variants |
|                             | 3. Educational research to evaluate the effectiveness of various teaching methods                                                                                          |

ANOVA offers a powerful and versatile tool for comparing means across multiple groups and assessing differences in treatment effects. However, it's essential to ensure that the assumptions of the test are met and to interpret the results cautiously, especially when dealing with complex experimental designs.

## Hypothesis testing methods
| **Hypothesis Test**               | **Formula**                                     | **Advantages**                                                                                     | **Disadvantages**                                                                                                  | **When to Use**                                                                                                          | **Cautions**                                                                                                               | **Other Information**                                                                                                                             |
|----------------------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Z-Test                           | \( Z = \frac{{\bar{x} - \mu}}{{\frac{{\sigma}}{{\sqrt{n}}}}} \) | 1. Simple to apply and understand<br>2. Suitable for large sample sizes<br>3. Provides precise results | 1. Requires known population standard deviation<br>2. Less applicable for small sample sizes                        | Large sample sizes with known population standard deviation                                                               | Ensure data meet assumptions of normality and homogeneity of variances                                                           | Useful for comparing sample mean to population mean                                                                                               |
| T-Test                           | \( t = \frac{{\bar{x}_1 - \bar{x}_2}}{{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}} \) | 1. Suitable for small sample sizes<br>2. Does not require known population standard deviation     | 1. Assumes normality and homogeneity of variances<br>2. Less powerful for large sample sizes                        | Small sample sizes or unknown population standard deviation                                                               | Verify assumptions of normality, homogeneity of variances, and independence of observations                                      | Applicable for comparing means of two groups                                                                                                      |
| Chi-Square Test                  | \( \chi^2 = \sum{\frac{{(O - E)^2}}{{E}}} \)    | 1. Specifically designed for categorical data analysis<br>2. Assesses association between variables | 1. Assumes independent observations and sufficient expected frequencies<br>2. Not suitable for continuous data     | Analyzing categorical data to assess association between variables                                                           | Ensure expected frequencies are not too low                                                                                   | Applicable for analyzing contingency tables and testing independence between categorical variables                                                |
| Analysis of Variance (ANOVA)     | \( F = \frac{{MS_{between}}}{{MS_{within}}} \)  | 1. Compares means of multiple groups simultaneously<br>2. Efficient for complex experimental designs  | 1. Assumes normality, homogeneity of variances, and independence of observations<br>2. Sensitive to assumptions         | Comparing means of three or more groups                                                                                    | Verify assumptions of normality, homogeneity of variances, and independence of observations                                      | Useful for assessing differences in treatment effects across multiple conditions or groups                                                           |

This table provides a comprehensive overview of various hypothesis testing methods, including their formulas, advantages, disadvantages, when to use them, cautions, and additional information. Depending on the specific research question and characteristics of the data, researchers can select the most appropriate hypothesis testing method for their analysis.
### P values

P-values, or probability values, play a crucial role in hypothesis testing. A p-value represents the probability of obtaining the observed results, or results more extreme, under the assumption that the null hypothesis is true. In other words, it indicates the strength of the evidence against the null hypothesis.

**A p-value is a number that helps us understand how likely it is that our results happened by random chance**. If the p-value is small, it suggests that our results are probably not just due to luck. In statistical terms, a small p-value means there's strong evidence against the idea we're testing (usually called the null hypothesis). If the p-value is large, it means our results could easily have happened by chance, so we don't have strong evidence against the null hypothesis.


#### Key points about p-values:

1. A low p-value (typically less than a predefined threshold, commonly 0.05) suggests that the observed results are unlikely to have occurred if the null hypothesis were true. In such cases, the null hypothesis is often rejected in favor of the alternative hypothesis.

2. A high p-value suggests that the observed results are reasonably likely to occur even if the null hypothesis were true. In such cases, there is insufficient evidence to reject the null hypothesis.

3. The choice of significance level (alpha level), which is the threshold for determining statistical significance, is arbitrary and depends on the context of the study and the consequences of making Type I and Type II errors.

4. It's important to note that a p-value alone does not provide evidence in favor of the alternative hypothesis or prove a specific hypothesis. It simply indicates the strength of evidence against the null hypothesis.

In summary, hypothesis testing helps researchers draw conclusions about population parameters based on sample data, and p-values provide a quantitative measure of the strength of evidence against the null hypothesis.

### To undestand easily

1. **Hypothesis Testing**: You start with a "null hypothesis," which is like saying, "Let's assume the medicine doesn't work." Then, you have an "alternative hypothesis," which says, "No, the medicine does work." You collect data by giving the medicine to some people and seeing if it helps them.

2. **P-values**: Think of p-values as a measure of surprise. If the medicine really doesn't work (like the null hypothesis says), you'd expect to see certain results just by chance. But if you see results that are really unlikely to happen by chance, it's like getting a big surprise. The p-value tells you how surprising your results are if the medicine doesn't work.

- If the p-value is really small (like less than 0.05), it means your results are super surprising if the medicine doesn't work. So, you start thinking, "Hmm, maybe the medicine actually does work!"
  
- If the p-value is not so small, it means your results could happen fairly often even if the medicine doesn't work. So, you're less surprised, and you might think, "Well, maybe the medicine doesn't really do much."

That's it in a nutshell! Hypothesis testing helps you decide between two ideas, and p-values tell you how surprising your results are if one of those ideas is true.



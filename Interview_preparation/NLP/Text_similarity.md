### [comparison-of-different-word-embeddings-on-text-similarity](https://intellica-ai.medium.com/comparison-of-different-word-embeddings-on-text-similarity-a-use-case-in-nlp-e83e08469c1c)
   - [Cosine Similarity](https://www.kaggle.com/code/cdabakoglu/word-vectors-cosine-similarity)
   - Word- Novers Distance
   - Euclidian distance

There are several ways to measure similarity between two vectors in a vector space. The choice of similarity measure depends on the context of the problem and the characteristics of the vectors. Here are some common similarity measures:

1. **Cosine Similarity**:
   - Cosine similarity measures the cosine of the angle between two vectors in the vector space. It is a measure of similarity between two non-zero vectors.
   - Formula:
     \[ \text{cos}(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} \]
   - Cosine similarity ranges from -1 (perfect dissimilarity) to 1 (perfect similarity), with 0 indicating orthogonality (no similarity).

2. **Euclidean Distance**:
   - Euclidean distance is a measure of the straight-line distance between two points in Euclidean space. It is the most commonly used distance metric.
   - Formula:
     \[ d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2} \]
   - Smaller Euclidean distance implies higher similarity.

3. **Manhattan Distance**:
   - Manhattan distance (also known as city block distance or L1 norm) is the sum of the absolute differences between corresponding coordinates of points in a space.
   - Formula:
     \[ d(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} |a_i - b_i| \]

4. **Minkowski Distance**:
   - Minkowski distance is a generalized form of Euclidean and Manhattan distances, where the distance between two points is calculated using a parameter \( p \).
   - Formula:
     \[ d(\mathbf{a}, \mathbf{b}) = \left( \sum_{i=1}^{n} |a_i - b_i|^p \right)^{\frac{1}{p}} \]
   - When \( p = 1 \), it is equivalent to Manhattan distance; when \( p = 2 \), it is equivalent to Euclidean distance.

5. **Jaccard Similarity**:
   - Jaccard similarity measures the similarity between two sets by comparing their intersection and union.
   - Formula:
     \[ \text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|} \]
   - It is commonly used in text analysis and document retrieval.

6. **Pearson Correlation Coefficient**:
   - Pearson correlation coefficient measures the linear correlation between two vectors. It assesses how well the relationship between two variables can be described by a straight line.
   - Formula:
     \[ \text{Pearson Correlation} = \frac{\text{cov}(\mathbf{a}, \mathbf{b})}{\sigma_{\mathbf{a}} \sigma_{\mathbf{b}}} \]
   - It ranges from -1 to 1, where 1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation.

These are some of the common measures of similarity between vectors. The choice of measure depends on the specific application, the nature of the data, and the desired properties of the similarity measure.
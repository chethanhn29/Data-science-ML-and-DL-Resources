
###  Self attention Mechanism in  Transformers [Link7](#link7) or Checkout this [Blog](https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021#8607) or watch this [Tutorial](https://www.youtube.com/watch?v=g2BRIuln4uc&t=2s)


## For Better Understanding go through this material 
- [ Jalammar Blog](http://jalammar.github.io/illustrated-transformer/),
- After the above blog Watch this [Tutorial](https://www.youtube.com/watch?v=g2BRIuln4uc&t=2s) to know all about transformers and Self Attention and Multi Head Attention
- [Krish naik Explanation](https://www.youtube.com/watch?v=SMZQrJ_L1vo&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=30),
- [Youtube Tutorial ](https://www.youtube.com/watch?v=TQQlZhbC5ps),
- [Improved Versions of Transformers for Different Applications by Jalammar blog ](https://jalammar.github.io/illustrated-bert/)

![](http://jalammar.github.io/images/t/transformer_decoding_2.gif)
![](http://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)
![](http://jalammar.github.io/images/t/self-attention-output.png)

**Attention mechanism describes a weighted average of (sequence) elements with the weights dynamically computed based on an input query and elements’ keys.**

**Query:** The query is a feature vector that describes what we are looking for in the sequence, i.e. what would we maybe want to pay attention to.

**Keys:** For each input element, we have a key which is again a feature vector. This feature vector roughly describes what the element is “offering”, or when it might be important. The keys should be designed such that we can identify the elements we want to pay attention to based on the query.

**Values:** For each input element, we also have a value vector. This feature vector is the one we want to average over.

**Score function:** To rate which elements we want to pay attention to, we need to specify a score function 
. The score function takes the query and a key as input, and output the score/attention weight of the query-key pair. It is usually implemented by simple similarity metrics like a dot product, or a small MLP.

### The model is primarily composed of two blocks:

#### **Encoder (left):**
The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
#### **Decoder (right):**
The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

![](https://heidloff.net/assets/img/2023/02/transformers.png)

#### Transformers Models Types 
Each of these parts can be used independently, depending on the task:

**[Encoder-only models](https://huggingface.co/learn/nlp-course/chapter1/5?fw=pt):**
- Good for tasks that require understanding of the input, such as sentence classification and named entity recognition, extractive question answering.
- ALBERT
,BERT
,DistilBERT
,ELECTRA
,RoBERTa

**[Decoder-only models](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt):**
- Good for generative tasks such as text generation.
- CTRL,GPT,GPT-2,Transformer XL

**[Encoder-decoder models or sequence-to-sequence models](https://huggingface.co/learn/nlp-course/chapter1/7?fw=pt):**
- Good for generative tasks that require an input, such as translation or summarization,generative question answering.
-  Ex: BART,mBART,Marian,T5
##### Dependencies between words in the target language refer to the relationships between words in a sentence. These relationships can be grammatical, semantic, or pragmatic.

For example, in the sentence "The quick brown fox jumps over the lazy dog", the word "quick" is grammatically related to the word "fox", because they are both adjectives that modify the noun "fox". The word "quick" is also semantically related to the word "jumps", because they both refer to the same concept of speed. And the word "lazy" is pragmatically related to the word "dog", because it is used to describe the dog's behavior.

The Transformer architecture is able to learn these dependencies between words in the target language by using the self-attention mechanism. The self-attention mechanism allows the model to learn how each word in a sentence is related to all the other words in the sentence. This allows the model to predict the next word in the sentence more accurately, because it takes into account the context of the sentence.

## Transformers

Transformers are a type of deep learning model that has achieved remarkable success in various natural language processing (NLP) tasks. However, they can also be applied to other domains, such as computer vision, including image analysis and generation.

At the core of transformers is the self-attention mechanism, which allows the model to capture relationships between different elements within a sequence. In the case of language processing, these elements are usually words in a sentence. In the case of image analysis, the elements can be pixels or smaller image patches.

**self-attention mechanism**:Self-attention is a key mechanism used in transformers, which allows the model to focus on different parts of the input sequence when processing each element.

**The self-attention mechanism relies on three key components:** queries, keys, and values. To understand how these components work, let's use the example of language processing. Suppose we have a sentence: "The cat sat on the mat." Each word in this sentence will have a query, a key, and a value associated with it.

Queries are used to calculate the similarity between the current word and other words in the sentence. Keys represent the other words and are used to compute the relevance of each word to the current word. Values contain information about each word and are used to create the final output.

To compute the similarity between a query and a key, we use a mathematical operation called dot product. The dot product measures how much the query and key align with each other. By performing dot products between a query and all the keys, we obtain a similarity score for each word in the sentence. These similarity scores determine how much attention each word should pay to the others.

After calculating the similarity scores, they are normalized using a softmax function, which ensures that the attention weights sum up to 1. This allows the model to focus more on the relevant words while suppressing the less important ones.

Finally, the attention weights are used to weigh the corresponding values. The values are then combined using a weighted sum, resulting in the final output for the current word.

**By applying self-attention to each word in the sentence, the transformer model can capture contextual information and dependencies between words. This enables the model to generate more accurate predictions and understand the meaning of the text.**

In the case of image analysis, transformers can be adapted by dividing the image into smaller patches and treating them as the elements for self-attention. This allows the model to capture relationships between different patches and understand the overall structure of the image.

In summary, transformers use the self-attention mechanism to capture relationships between elements in a sequence, such as words in a sentence or patches in an image. By computing similarity scores between queries and keys, weighing values, and combining them, transformers can capture context and dependencies, leading to powerful representations and predictions in various tasks.


## Step By Step Process of Transformers
Transformers were introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017). Let's go through the main steps and processes of transformers as described in the paper:

### Input Representation: 
The input to the transformer is a sequence of tokens. Each token is transformed into a high-dimensional vector using an embedding layer. The position of each token in the sequence is also encoded using positional encodings, which provide information about the order of the tokens.

### Self-Attention:
The core of the transformer is the self-attention mechanism. It allows the model to capture relationships between different positions in the input sequence. The self-attention mechanism is applied in parallel to all positions and consists of three linear transformations: query, key, and value.

### Query, Key, and Value:
For each position in the input sequence, the self-attention mechanism generates three vectors: query, key, and value. These vectors are obtained by multiplying the input embeddings with learned weight matrices.

### Similarity Scores:
The similarity between two positions is computed by taking the dot product between the query of one position and the key of another position. This yields a raw score that represents the relevance or similarity between the two positions.

### Attention Weights: 
The raw similarity scores are scaled by dividing them by the square root of the dimension of the query vector. Then, a softmax function is applied to normalize the scores, ensuring that they sum up to 1. These normalized scores, known as attention weights, determine the importance or attention given to each position.

### Weighted Sum:
The attention weights are used to weigh the corresponding value vectors. The weighted sum of the values gives the output of the self-attention mechanism for each position.

### Multi-Head Attention: 
To capture different types of information and improve the expressiveness of the model, the self-attention mechanism is applied multiple times in parallel. Each parallel self-attention operation is referred to as a "head." Each head has its own set of learned weight matrices for query, key, and value transformations. The outputs of all the heads are concatenated and linearly transformed to produce the final multi-head attention output.

### Position-wise Feed-Forward Networks:
After the multi-head attention, the transformer uses a position-wise feed-forward network for each position independently. It consists of two linear transformations with a non-linear activation function in between. This helps the model capture complex relationships and higher-level features.

### Residual Connections and Layer Normalization:
To address the issue of vanishing gradients and facilitate the flow of information, residual connections are added around each sub-layer (self-attention and position-wise feed-forward network) in the transformer. Additionally, layer normalization is applied to the outputs of each sub-layer to normalize the values and stabilize the training process.

### Encoder-Decoder Architecture: 
Transformers are commonly used in sequence-to-sequence tasks, such as machine translation. For these tasks, the transformer architecture consists of an encoder and a decoder. The encoder processes the input sequence, and its final hidden states are used as inputs to the decoder. The decoder then generates the output sequence, attending to the encoder's hidden states and using self-attention and the other transformer components.

### Training and Inference: 
Transformers are typically trained using the self-attention mechanism to predict the next token in the sequence (autoregressive training). During inference, the model generates the output sequence one token at a time, using the previously generated tokens as input.

These are the main steps and processes involved in transformers, as described in the original research paper. They enable transformers to effectively capture dependencies and relationships in sequences, leading to state-of-the-art performance in various natural language processing and other tasks.

![](http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

## Self Attenction Mechanism in Transformers
Adding some context to the words in a sentence is known as Self-Attention.

In the example you have provided, the sentence "Bark is very cute and he is a dog" is represented as a sequence of word embeddings. The self-attention mechanism then computes a weight for each word in the sentence, based on how closely the word is related to the other words in the sentence. These weights are then used to combine the word embeddings, resulting in a new representation of the sentence that takes into account the relationships between the words.


The self-attention mechanism gives high weight to the words "cute" and "dog" by computing a score for each word in the sentence, based on how closely the word is related to the other words in the sentence. This score is then used to determine the weight of the word in the new representation of the sentence.

The score for each word is computed using a dot product between the word embedding for that word and the word embeddings for all the other words in the sentence. The dot product is a measure of how similar two vectors are, so a high dot product indicates that two words are closely related.

In the sentence "Bark is very cute and he is a dog", the word embedding for the word "Bark" will have a high dot product with the word embeddings for the words "cute" and "dog", because these words are all related to the same animal. This means that the score for the word "Bark" will be high, which will cause it to have a high weight in the new representation of the sentence.

In transformers, self-attention is a mechanism that allows the model to weigh the importance of different words or tokens in a sequence when processing it. It enables the model to capture relationships and dependencies between different positions in the input sequence. Self-attention is often referred to as the "Scaled Dot-Product Attention."

Here's a step-by-step explanation of how self-attention works:

- **Inputs:** Suppose we have an input sequence of tokens, typically represented as word embeddings or numerical vectors. Each token has an associated query, key, and value vectors. These vectors are derived from the input embeddings using linear transformations.

- **Query, Key, and Value:** Each token's query vector is used to calculate the similarity between that token and other tokens in the sequence. Similarly, the key vectors represent the tokens that are being compared, and the value vectors contain the information that needs to be attended to.

- **Query vector:** The query vector represents the part of the input sequence that the transformer is currently trying to understand. It is a vector that is typically the same size as the key vectors. or  Queries is a set of vectors you want to calculate attention for.
- **Key vector:** The key vector represents each word in the input sequence. It is a vector that is typically the same size as the query vector.or  Keys is a set of vectors you want to calculate attention against.
- **Value vector:** The value vector represents the meaning of each word. It is a vector that is typically much larger than the query and key vectors.
- The attention mechanism calculates a weight for each word in the input sequence, based on how well the query vector matches the key vectors. The words with the highest weights are then used to update the query vector. or As a result of dot product multiplication you'll get set of weights a (also vectors) showing how attended each query against Keys. Then you multiply it by Values to get resulting set of vectors.

- This process is repeated for each word in the input sequence, and the final query vector is used to represent the entire input sequence. The attention mechanism allows the transformer to learn which words are most important for understanding the meaning of a sentence, and it is a key reason why transformers are so successful at natural language processing tasks.

- **Similarity Scores**: To compute the similarity between two tokens, the dot product between the query vector of one token and the key vector of another token is taken. These dot products are then scaled by dividing them by the square root of the dimension of the key vectors to stabilize the gradients during training.

- **Attention Weights:** The similarity scores are passed through a softmax function, which converts them into attention weights. The softmax ensures that the weights sum up to 1 and represent the importance of each token with respect to the current token.

- **Weighted Sum:** The attention weights are used to weigh the value vectors of the tokens. Each token's value vector is multiplied by its corresponding attention weight, and these weighted vectors are summed up to obtain the context vector for the current token.

- **Multi-Head Attention:** Transformers often employ multiple sets of query, key, and value vectors, called attention heads. Each attention head learns different relationships and attends to different parts of the input sequence independently. The context vectors obtained from each attention head are concatenated and linearly transformed to produce the final output.

- **Residual Connection and Layer Normalization:** To preserve important information from the original input, the output of the self-attention mechanism is added to the input (residual connection). The result is then passed through a layer normalization operation, which normalizes the values across the sequence dimension.

The self-attention mechanism allows the model to capture both local and global dependencies in the input sequence, making transformers highly effective in modeling context and capturing long-range relationships. It has been instrumental in achieving state-of-the-art results on various NLP tasks such as machine translation, text summarization, and language understanding.






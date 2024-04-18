The transformer architecture is made up of two main parts: the encoder and the decoder. The encoder takes the input sequence and produces a sequence of hidden states. The decoder takes the hidden states from the encoder and produces the output sequence.Transformers are highly efficient due to their parallelization capability, allowing them to process vast amounts of data with extensive context windows.
- [Encoder](#encoder)
- [Decoder](#decoder)
![](https://heidloff.net/assets/img/2023/02/transformers.png)


# Encoder
The encoder is made up of a stack of encoder blocks. Each encoder block consists of the following sublayers:

## [Input Processing for Transformers](https://www.youtube.com/watch?v=dichIcUZfOw)
This video delves into understanding the input fed to the transformer encoder. It begins with the fundamental step of input processing, converting natural language (English) into a machine-interpretable format using vocabulary dictionaries and numerical indexing.

- **Vocabulary Dictionary and Index Assignment:** Words from the training data are indexed numerically, forming the basis of the input. Each word is represented by a corresponding index (denoted as 'x').

## 1) Embedding layer:
- Following index assignment, the inputs move through the embedding layer. Here, each word index is mapped to a vector (denoted as 'e') from a predefined vocabulary. These vectors initially contain random numbers but are optimized during training to facilitate the model in its learning.

- **Word Embeddings:** These vectors serve as representations of words in a multi-dimensional space, capturing various linguistic features learned by the model during training.
-  This sublayer maps each word in the input sequence to a vector representation. The vector representation is typically a 1024-dimensional vector. or The input to the transformer is a sequence of tokens. Each token is transformed into a high-dimensional vector using an embedding layer. The position of each token in the sequence is also encoded using positional encodings, which provide information about the order of the tokens.

## 2)Positional encoding:
- This sublayer adds positional information to the word vectors. The positional information helps the model to learn the order of the words in the input sequence.

#### How does Positional encoding work?
Retaining word order information poses a significant challenge for transformers compared to sequential models like LSTMs. Position embeddings are introduced to address this challenge.

- **Importance of Position Information:** The arrangement of words in a sentence significantly impacts its meaning and sentiment.
- **Generating Position Embeddings:** Initial attempts to add simple positional values to word embeddings might distort their information. To solve this, the authors of the original transformer paper devised a method using sinusoidal wave frequencies to capture position information.

#### Conclusion of Position embedding
To summarize, the input undergoes processing into word indices, followed by association with word embeddings. Position embeddings are then added to create position-aware word embeddings, ensuring the preservation of crucial word order information for the transformers. The upcoming video will delve into the pivotal multi-head attention layer, a core component utilizing self-attention in transformer neural models.


### 3)Self-Attention:
The core of the transformer is the self-attention mechanism. It allows the model to capture relationships between different positions in the input sequence. The self-attention mechanism is applied in parallel to all positions and consists of three linear transformations: query, key, and value.

  - ### Query, Key, and Value:
For each position in the input sequence, the self-attention mechanism generates three vectors: query, key, and value. These vectors are obtained by multiplying the input embeddings with learned weight matrices.

  - ### Similarity Scores:
The similarity between two positions is computed by taking the dot product between the query of one position and the key of another position. This yields a raw score that represents the relevance or similarity between the two positions.

  - ### Attention Weights: 
The raw similarity scores are scaled by dividing them by the square root of the dimension of the query vector. Then, a softmax function is applied to normalize the scores, ensuring that they sum up to 1. These normalized scores, known as attention weights, determine the importance or attention given to each position.

  - ### Weighted Sum:
The attention weights are used to weigh the corresponding value vectors. The weighted sum of the values gives the output of the self-attention mechanism for each position.
## 4) Multi-head attention:
- This sublayer calculates the attention weights for each word in the input sequence. The attention weights determine how much each word should attend to the other words in the sequence using Query,Key and values vectors.
 #### or
- To capture different types of information and improve the expressiveness of the model, the self-attention mechanism is applied multiple times in parallel. Each parallel self-attention operation is referred to as a "head." Each head has its own set of learned weight matrices for query, key, and value transformations. The outputs of all the heads are concatenated and linearly transformed to produce the final multi-head attention output.
## 5)Feed-forward:
-  This sublayer applies a feed-forward neural network to the output of the multi-head attention sublayer. The feed-forward neural network helps to improve the representation of the input sequence.
## 6)Residual connection:
- This sublayer adds the output of the multi-head attention sublayer to the output of the feed-forward neural network. The residual connection helps to prevent the model from overfitting.
The output of the encoder is a sequence of hidden states. The hidden states represent the meaning of the input sequence.
- To address the issue of vanishing gradients and facilitate the flow of information, residual connections are added around each sub-layer (self-attention and position-wise feed-forward network) in the transformer. Additionally, layer normalization is applied to the outputs of each sub-layer to normalize the values and stabilize the training process.


# Decoder

The decoder is made up of a stack of decoder blocks. Each decoder block consists of the following sublayers:
## 1) Embedding layer:
- This sublayer maps each word in the output sequence to a vector representation. The vector representation is typically the same size as the vector representation used in the encoder.
## 2) Masked multi-head attention:
- This sublayer calculates the attention weights for each word in the output sequence. The attention weights determine how much each word should attend to the other words in the sequence, as well as the hidden states from the encoder. The masked multi-head attention sublayer prevents the decoder from attending to the future words in the output sequence.
## 3) Position-wise Feed-Forward Networks:
After the multi-head attention, the transformer uses a position-wise feed-forward network for each position independently. It consists of two linear transformations with a non-linear activation function in between. This helps the model capture complex relationships and higher-level features.
## 4) Residual connection:
-  This sublayer adds the output of the masked multi-head attention sublayer to the output of the feed-forward neural network. The residual connection helps to prevent the model from overfitting.
The output of the decoder is a sequence of words. The words represent the translation of the input sequence.

### Encoder-Decoder Architecture: 
Transformers are commonly used in sequence-to-sequence tasks, such as machine translation. For these tasks, the transformer architecture consists of an encoder and a decoder. The encoder processes the input sequence, and its final hidden states are used as inputs to the decoder. The decoder then generates the output sequence, attending to the encoder's hidden states and using self-attention and the other transformer components.

### Training and Inference: 
Transformers are typically trained using the self-attention mechanism to predict the next token in the sequence (autoregressive training). During inference, the model generates the output sequence one token at a time, using the previously generated tokens as input.

These are the main steps and processes involved in transformers, as described in the original research paper. They enable transformers to effectively capture dependencies and relationships in sequences, leading to state-of-the-art performance in various natural language processing and other tasks.

![](http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)


### Input and output of each layer and sublayer
The input and output of each layer and sublayer in the transformer model are as follows:

| Layer | Sublayer | Input | Output |
|---|---|---|---|
| Encoder | Embedding | Word tokens | Word vectors |
| Encoder | Positional encoding | Word vectors | Word vectors |
| Encoder | Multi-head attention | Word vectors | Attention weights |
| Encoder | Feed-forward | Attention weights | Hidden states |
| Decoder | Embedding | Word tokens | Word vectors |
| Decoder | Masked multi-head attention | Word vectors | Attention weights |
| Decoder | Feed-forward | Attention weights | Hidden states |



- **Model Overview**: The Transformer is a deep learning model architecture used primarily in natural language processing (NLP) tasks. It introduced a new way of handling sequences that relies on self-attention mechanisms. Below are the key components and steps in the Transformer architecture:

  - **Input Sequence**: The input sequence consists of word embeddings, and each word is represented as a vector.
  - **Positional Encoding**: Positional encoding is added to the word embeddings to account for word order.
  - **Multi-Head Self-Attention (Encoder)**: The encoder's multi-head self-attention mechanism captures dependencies among words in the input sequence. It uses multiple attention heads to capture different relationships.
  - **Layer Normalization (Encoder)**: Layer normalization stabilizes training by normalizing the output of self-attention.
  - **Residual Connection (Encoder)**: Residual connections help preserve the original information while adding context.
  - **Feedforward Neural Network (Encoder)**: A feedforward network enhances the model's representation of the data.
  - **Layer Normalization (Encoder)**: Layer normalization stabilizes training after the feedforward layer.
  - **Residual Connection (Encoder)**: Residual connections preserve the original embeddings while adding the transformed information.
  - **Multi-Head Self-Attention (Decoder)**: The decoder's self-attention mechanism captures dependencies in the output sequence and encoder's output.
  - **Layer Normalization (Decoder)**: Layer normalization ensures stability in the decoder's self-attention output.
  - **Residual Connection (Decoder)**: Residual connections preserve information in the decoder's self-attention output.
  - **Encoder-Decoder Attention (Decoder)**: The encoder-decoder attention aligns the decoder's output with the encoder's context.
  - **Layer Normalization (Decoder)**: Layer normalization stabilizes training in the decoder.
  - **Residual Connection (Decoder)**: Residual connections maintain original embeddings in the decoder's encoder-decoder attention output.
  - **Positional Decoding (Decoder)**: Positional decoding adds positional information to the decoder's embeddings.
  - **Output Layer**: The output layer produces probability distributions for generating output words in the desired format.

- **Additional Considerations**:
  - Multi-Head Attention Variants: Transformers may use variants like scaled dot-product attention.
  - Masking: Masking may be applied to prevent attention to future positions in the decoder.
  - Hyperparameters: The number of layers, attention heads, and other hyperparameters are fine-tuned.
  - Training: Training involves minimizing a loss function and optimizing model weights.
  - Inference: Models use decoding strategies like greedy decoding or beam search during inference.
  - Applications: Transformers are used in various NLP tasks and beyond.

- The Transformer architecture has significantly improved the performance of various NLP applications, and its parallelization-friendly design has made it a reference model for cloud-based machine learning services.

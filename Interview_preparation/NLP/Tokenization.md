# Text Preprocessing level -1
#### [Tokenizers practical](https://www.kaggle.com/code/funtowiczmo/hugging-face-tutorials-training-tokenizer)
## Tokenization
Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. If the text is split into words, then its called as 'Word Tokenization' and if it's split into sentences then its called as 'Sentence Tokenization'.
  - Tokenization by Words 
  - Tokenization by Sentence

```python
# Tokenization of paragraphs/sentences
import nltk
nltk.download()

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence."""
               
# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)

# Tokenizing words
words = nltk.word_tokenize(paragraph)
```

### Tokenizers
#### [Tokenizers Explanation by HUggingface](https://huggingface.co/docs/transformers/main/tokenizer_summary)

There are several types of tokenizers that you can build for processing text, each with its own advantages and disadvantages. Here are some common types of tokenizers along with their characteristics:

1. **Whitespace Tokenizer**:
   - **Advantages**: Simple to implement, separates tokens based on whitespace characters (spaces, tabs, newlines).
   - **Disadvantages**: Doesn't handle punctuation marks or special characters well, may not be suitable for languages with complex orthographies like Kannada.
   - **Suitability for Kannada**: Limited, may not capture meaningful linguistic units accurately.

2. **Word Tokenizer**:
   - **Advantages**: Splits text into individual words, handles punctuation marks, and special characters better than whitespace tokenizer.
   - **Disadvantages**: May not handle compound words or idiomatic expressions well, especially in languages with agglutinative morphology like Kannada.
   - **Suitability for Kannada**: Better than whitespace tokenizer but still may not capture all linguistic units accurately.

3. **Sentence Tokenizer**:
   - **Advantages**: Splits text into sentences, useful for tasks like text summarization, machine translation, and sentiment analysis.
   - **Disadvantages**: Highly language-dependent, may struggle with languages that have complex sentence structures or lack clear sentence boundaries.
   - **Suitability for Kannada**: Useful for text processing tasks involving sentence-level analysis.

4. **Subword Tokenizer** (e.g., Byte-Pair Encoding (BPE), WordPiece, SentencePiece):
   - **Advantages**: Segments text into subword units, helpful for handling rare words, out-of-vocabulary (OOV) tokens, and morphologically rich languages.
   - **Disadvantages**: Requires pre-training or learning from data, may produce large vocabularies, leading to increased computational complexity.
   - **Suitability for Kannada**: Effective for capturing subword units in Kannada, especially beneficial for handling agglutinative morphology and OOV tokens.

5. **Character Tokenizer**:
   - **Advantages**: Tokenizes text into individual characters, useful for tasks like character-level language modeling or handling languages with complex orthographies.
   - **Disadvantages**: Increases token sequence length, may struggle with languages that have large character sets or complex grapheme clusters.
   - **Suitability for Kannada**: Can be useful for certain tasks like character-level language modeling, but may not capture higher-level linguistic units accurately.

For Kannada language specifically, a subword tokenizer like Byte-Pair Encoding (BPE) or SentencePiece would likely be the most suitable choice. Kannada is a morphologically rich language with complex word forms, and a subword tokenizer can effectively capture the subword units, helping to handle OOV tokens and reducing vocabulary size. Additionally, subword tokenization is language-agnostic and can adapt well to different languages, making it a versatile choice for multilingual applications.


###### Stemming and lemmatization are both natural language processing (NLP) techniques that are used to reduce inflected words to their base forms, known as stems or lemmas. However, there are some key differences between the two techniques.

 ## Stemming
 
Stemming is a more simplistic process that simply removes affixes from words, regardless of their meaning. This can sometimes lead to the creation of incorrect or nonsensical stems. For example, the word "running" could be stemmed to "runn", which is not a valid word.

stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form—generally a written word form.
  - Poor representation-- base word 
  - It can leads to Misspelling or Incorrect Words ex: Caring ===>> Car
  - Root Word or Base Word
  - Less Time 
  - Applications
         - Gmail classifier
         - Positive , Negative Classifyer 
         - Sentiment Classifiyer 
  
 
 Example for History,Historical==>>> Histri
 
 ```python
 import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds.""
                              
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
```
 
 ## Lemmmatization
 
 Lemmatization is a more complex process that takes into account the meaning of words in order to identify their correct base forms. This is done by using a dictionary or thesauraus to look up the word and its various forms. For example, the word "running" could be lemmatized to "run", which is its correct base form.
 
 Lemmatization is a text pre-processing technique used in natural language processing (NLP) models to break a word down to its root meaning to identify similarities. For example, a lemmatization algorithm would reduce the word better to its root word, or lemme, good
  - Proper representation
  - Difinitive meaning of words ex: Caring ===>> Care
  - lot of time
  - Applications( The response should be Meaningful )
        - Chatbots
        - Natural Language Generation
        - Language translation
        - 
  

Ex: History and Historical==>> History

```python
lemmatizer = WordNetLemmatizer()
# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
```
 
 ## Stopwords

Stop words are a set of commonly used words in any language. For example, in English, “the”, “is” and “and”, would easily qualify as stop words. In NLP and text mining applications, stop words are used to eliminate unimportant words, allowing applications to focus on the important words instead.

1. They provide no meaningful information, especially if we are building a text classification model. Therefore, we have to remove stopwords from our dataset.
2. As the frequency of stop words are too high, removing them from the corpus results in much smaller data in terms of size. Reduced size results in faster computations on text data and the text classification model need to deal with a lesser number of features resulting in a robust model.

![](https://wisdomml.in/wp-content/uploads/2022/08/stop_words-1024x556.jpg)
![](https://www.computerhope.com/jargon/s/stop-words.png)

#### Where we should not remove stop words ?
There are some applications in NLP such as Part of Speech (PoS) Tagging, Named Entity Recognition (NER), Parsing, etc., where we should not remove stop words besides we should preserve them as they actually provide grammatical information in those applications.


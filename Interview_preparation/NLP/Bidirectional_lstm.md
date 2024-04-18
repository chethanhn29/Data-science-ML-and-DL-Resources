# BI-LSTM(Bi-directional long short term memory)
Bidirectional long-short term memory(bi-lstm) is the process of making any neural network o have the sequence information in both directions backwards (future to past) or forward(past to future).

- In bidirectional, our input flows in two directions, making a bi-lstm different from the regular LSTM. With the regular LSTM, we can make input flow in one direction, either backwards or forward. However, in bi-directional, we can make the input flow in both directions to preserve the future and the past information. For a better explanation, let’s have an example.    

- In the sentence “boys go to …..” we can not fill the blank space. Still, when we have a future sentence “boys come out of school”, we can easily predict the past blank space the similar thing we want to perform by our model and bidirectional LSTM allows the neural network to perform this.
- In other words, rather than encoding the sequence in the forward direction only, we encode it in the backward direction as well and concatenate the results from both forward and backward LSTM at each time step. The encoded representation of each word now understands the words before and after the specific word.

Below is the basic architecture of Bi-LSTM.
![]([https://files.codingninjas.in/article_images/bidirectional-lstm-1-1644656900.webp](https://www.iloveimg.com/download/mvztnftj9zdxvjn9vs31c58ynb96bn5f43ftwqlm0Ahtkd156lb8kbh70jp364wcscc4qt3d0x90dj88sltp9z5vnhkfxj0590tAAs3fmr790cAd1x6dy8pck3v1td68rhwn7hts74Ak5dvg4qndjfmw428wA7btw6pm8hml5gf2gy51fd11/6))
![](https://149695847.v2.pressablecdn.com/wp-content/uploads/2021/07/image-5.jpeg)

## Working of Bi-LSTM
Let us understand the working of Bi-LSTM using an example. Consider the sentence “I will swim today”. The below image represents the encoded representation of the sentence in the Bi-LSTM network.
- So when forward LSTM occurs, “I” will be passed into the LSTM network at time t = 0, “will” at t = 1, “swim” at t = 2, and “today” at t = 3.
-  In backward LSTM “today” will be passed into the network at time t = 0, “swim” at t = 1, “will” at t = 2, and “I” at t = 3.
-   In this way, results from both forward and backward LSTM at each time step are calculated.
![bidirectional-lstm-2-1644656900](https://github.com/chethanhn29/Personal-Collection-of-Resources-to-learn/assets/110838853/a097d5e7-1792-4e70-8eef-46ca2f593ca9)

## Application

BiLSTM will have a different output for every component (word) of the sequence (sentence). As a result, the BiLSTM model is beneficial in some NLP tasks, such as sentence classification, translation, and entity recognition. In addition, it finds its applications in speech recognition, protein structure prediction, handwritten recognition, and similar fields.

## Bi-LSTM in keras
- To implement [Bi-LSTM in keras](https://keras.io/api/layers/recurrent_layers/bidirectional/), we need to import the Bidirectional class and [LSTM class](https://keras.io/api/layers/recurrent_layers/lstm/) provided by keras.
- Now, to implement the Bi-LSTM, we just need to wrap the LSTM layer inside the Bidirectional class.
```python
tf.keras.layers.Bidirectional(LSTM(units))
```

```python
 import numpy as np
 from keras.preprocessing import sequence
 from keras.models import Sequential
 from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
 from keras.datasets import imdb 
#Here we are going to use the IMDB data set for text classification using keras and bi-LSTM network 

 n_unique_words = 10000 # cut texts after this number of words
 maxlen = 200
 batch_size = 128 
# In the above, we have defined some objects we will use in the next steps. In the next step, we will load the data set from the Keras library.

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=n_unique_words)
# To fit the data into any neural network, we need to convert the data into sequence matrices. For this, we are using the pad_sequence module from keras.preprocessing.

 x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
 x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
 y_train = np.array(y_train)
 y_test = np.array(y_test)

## In the next, we are going to make a model with bi-LSTM layer.

 model = Sequential()
 model.add(Embedding(n_unique_words, 128, input_length=maxlen))
 model.add(Bidirectional(LSTM(64)))
 model.add(Dropout(0.5))
 model.add(Dense(1, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Here in the above codes we have in a regular neural network we have added a bi-LSTM layer using keras. Keras of tensor flow provides a new class [bidirectional] nowadays to make bi-LSTM.

## In the next step we will fit the model with data that we loaded from the Keras.

 history=model.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=12,
           validation_data=[x_test, y_test])
 print(history.history['loss'])
 print(history.history['accuracy'])
```

## FAQ

1. What is the difference between GRU and LSTM?

A: GRU has two gates, i.e., reset and update gate, whereas LSTM has three gates, i.e, input, output, and forget gate. GRU is preferred in small datasets, whereas LSTM is preferred while handling larger datasets.

2. Why is Bi-LSTM better than LSTM?

A: At every time step, LSTM calculates the results of forwarding LSTM, but in the case of Bi-Direction results from both forward and backward LSTM at each time step are calculated.

3. What are the limitations of Bi-LSTM?

A: Bi-LSTM takes more time to train than normal LSTM networks. Also, they acquire more memory to train. They are easy to overfit and dropout implementation is hard in Bi-LSTM.

4. What is a bidirectional layer?

A: Bidirectional recurrent neural networks (BRNN) connect two hidden layers of opposite directions to the same output. With this form of generative deep learning, the output layer can simultaneously get information from past (backward), and future (forward) states.

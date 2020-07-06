# NLP - Word Vector

View the book with "<i class="fa fa-book fa-fw"></i> Book Mode".

### One-hot-encoding
One-hot-encoding is a representation of categorical variables as binary vectors. This first requires that the categorical values be mapped to integer values. Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.

* Curse of dimensionality: The number of dimensions increases linearly as we add new words, causing sparsity and hard to find similiarity between words.
* It doesn’t allow an algorithm to generalize the cross words such as inner products equal 0.
* The embedding matrix is very sparse, mainly made up of zeros.
* No shared information between words and no commonalities between similar words. Un-ordered, therefore the context of words are lost.
* The vector representation is in binary form, therefore no frequency information is taken into account.

### Bag-of-Words
Bag-of-Words is a method to extract features from text documents. These features can be used for training machine learning algorithms. It creates a vocabulary dictionary of all the unique words occurring in all the training documents. Then, calculating the frequency of each word in the corresponding documents, called frequency matrix or term frequency (TF).

* Semantic meaning: the basic approach does not consider the meaning of the word in the document. It completely ignores the context, but the same words could be used with different meanings based on the context or nearby words.
* Vector size: For a large document, the vector size can be huge resulting in a lot of computation and time. You may need to ignore words based on relevance to your use case.

### TF-IDF
Term Frequency-Inverse Document Frequency (TF-IDF) is a statistical measure used to evaluate the importance of a word to a document in a collection or corpus. The TF-IDF scoring value increases proportionally to the number of times a word appears in the document, but it is offset by the number of documents in the corpus that contain the word.

* Term Frequency (TF): a scoring of the frequency of the word in the current document. The number of times word appears in a document/total number of words in the document.
* Inverse Document Frequency (IDF): a scoring of how rare the word is across documents. Log(Total number of documents/Number of douments with word).
* It's similiar with Bag-of-Words that not capture position in text, semantics, co-occurrences in different documents.
$$TF-IDF = TF * IDF$$


### Skip Grams
Skip Gram predicts the surrounding context words within specific window given current word. The input layer contains the current word and the output contains the context words. 

### Word Embedding
Word Embedding is used for mapping words to vectors of real numbers, which represents words or phrases in vector space with several dimensions. Word2vec gives similarity in vector representation. The basic idea of word embedding is words that occur in similar context tend to be closer to each other in vector space. Embedding is dense vector with similarity. Similarity comes from neighbor words.

To train Word2vec, firstly extract words with a neighbor word (might apply skip gram). Second, transform words to numerical vectors by one-hot-encoding. Third, throw the current word one-hot-encoding as inputs and neighbor words as target into a 2-layer neutral network. we can use cross entropy as cost function and gradient descent as optimizer to learn the weights of neutral network. Finally, the hidden layer weight is the Word2vec.

![](https://i.imgur.com/6qbj2Lv.png)
![](https://i.imgur.com/KaEnSNv.png)

We can treat the dimensions of word2vec as hyper-parameter. Typically, its range is between 100–300, least 50D to achieve lowest accuracy. If training time is not a big deal for your application, stick with 200 dimensions as it gives nice features. Extreme accuracy can be obtained with 300D. After 300D word features won’t improve dramatically, and training will be extremely slow.

Pros:
* Calculating the semantic similarity of two words by word vector.
* Perhaps the first scalable model that generated word embeddings for large corpus (millions of unique words). Feed the model raw text and it outputs word vectors.

Cons:
* Multiple sense embeddings not captured. For example, a word like “cell” that could mean “prison, “biological cell”, “phone” etc are all represented in one vector.
* Can’t handle out-of-vocabulary words, have to re-train to add new words.

### GloVe
GloVe is an unsupervised learning algorithm for obtaining vector representations for words that puts emphasis on the importance of word-word co-occurences to extract meaning rather than other techniques such as skip-gram or bag of words. The idea behind is that a certain word generally co-occurs more often with one word than another. The word "ice" is more likely to occur alongside the word water for instance. The goal of Glove is to enforce the word vectors to capture sub-linear relationships in the vector space. Thus, it proves to perform better than Word2vec in the word analogy tasks.

### FastText
The key difference between FastText and Word2Vec is the use of n-grams. Word2Vec learns vectors only for complete words found in the training corpus. FastText, on the other hand, learns vectors for the n-grams that are found within each word, as well as each complete word.

At each training step in FastText, the mean of the target word vector and its component n-gram vectors are used for training. The adjustment that is calculated from the error is then used uniformly to update each of the vectors that were combined to form the target. This adds a lot of additional computation to the training step. At each point, a word needs to sum and average its n-gram component parts. The trade-off is a set of word-vectors that contain embedded sub-word information. These vectors have been shown to be more accurate than Word2Vec vectors by a number of different measures.

* Generate better word embeddings for rare words: Word2Vec is simply catching word neighbor, so it's more often resulting in better word vectors for common word phrases.
* Handle Out-of-Vocabulary words: Construct the vector for a word from its character n grams even if word doesn’t appear in training corpus, however both Word2vec and Glove can’t.
* Hyperparamters of n-grams: Since the training is at character n-gram level, it takes longer to generate fasttext embeddings compared to word2vec — the choice of hyper parameters controlling the minimum and maximum n-gram sizes has a direct bearing on this time.
* Shortage of character embeddings: Boost the performance compared to using word embeddings like word2vec or Glove, but fasttext embeddings generation takes longer to train, but likely to be faster than LSTMs.


### Handle Out-of-Vocabulary words
1. Use pre-trained embedding with a large vocabulary
2. FastText
3. LSTM

###### tags: `Book`

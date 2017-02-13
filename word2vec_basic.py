# http://github.com/timestocome

# adapted from:
# https://www.tensorflow.org/tutorials/word2vec/
# Apache 2.0 Lic.


# paper
# http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf



# model
import math
import random
import numpy as np
import tensorflow as tf


# clean up text
import collections
import string as st 
from itertools import chain
import re


n_sentence_size = 10

# plotting
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Read the cleaned data file into separate words
def read_data(filename):

  with open(filename, encoding='utf-8') as f:
    input = f.read()
    data = input.split() 

  return data

filename = 'cleaned_buzzfeed.txt'
words = read_data(filename)
n_words_total = len(words)

n_unique_words = len(set(words))
vocabulary_size = int(n_unique_words * .7)

print("Unique words: ", n_unique_words)
print("Total Words", n_words_total)


# read in headlines 
# remove zero length ones (data is on every other line)
# min == 1, max == 35, avg == 9
def read_line(filename):

  headlines = []

  with open(filename) as f:
   
    min_sentence = 9999999999999
    for line in f:
     
      sentence = line.split()
      n_words = len(sentence)

      if n_words > 0:
          headlines.append(sentence)
      
  return headlines

headlines = read_line(filename)
print("headlines", len(headlines))




# Build the dictionary and replace rare words with UNK token.
def build_dataset(words):

  # create a dictionary with the most common words indexed in order of number times in text
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  
  for word, _ in count:
    dictionary[word] = len(dictionary)
  
  data = list()
  unk_count = 0
  
  for word in words:
    if word in dictionary:
      index = dictionary[word]

    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    
    data.append(index)
  
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  
  return data, count, dictionary, reverse_dictionary



data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # finished with original text input


print('Most common words (+UNK)', count[:25])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])




def create_batches(skip_window):
  
    input_batch = []
    target_labels = []
    #batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    #labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)


    for h in headlines:
        for i in range(skip_window, len(h) - (skip_window+1)):

            word = h[i]
            

            # look back
            input_batch.append(h[i])
            target_labels.append(h[i-skip_window])


            # look ahead
            input_batch.append(h[i])
            target_labels.append(h[i+skip_window])


    # convert to ints using dictionary and ndarrays
    batch = np.ndarray(shape=(len(input_batch)), dtype=np.int32)
    labels = np.ndarray(shape=(len(input_batch), 1), dtype=np.int32)
    for i in range(len(input_batch)):
        if dictionary.get(input_batch[i]):
            batch[i] = dictionary.get(input_batch[i])
        else: batch[i] = 0

        if dictionary.get(target_labels[i]):
            labels[i] = dictionary.get(target_labels[i])
        else: labels[i] = 0
    
    return batch, labels


batch, labels = create_batches(skip_window=1)





for i in range(n_sentence_size):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])




#########################################################################
# Build and train a skip-gram model.
########################################################################
batch_size = 16
embedding_size = 256  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right. 
num_skips = 2         # How many times to reuse an input to generate a label.



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the most frequent words
valid_size = 32      # Random set of words to evaluate similarity on.
valid_window = 128   # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64     # Number of negative examples to sample.


graph = tf.Graph()

with graph.as_default():

  # Placeholders for input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):

    # Look up embeddings for inputs.
    # start with random weights
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
    # look up vector for a given word
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    # weights and biases
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
              tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))
                     
                     

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()





################################################################
# Training
#################################################################
num_steps = 180001
batch_index = 0
n_samples = len(batch)
print("n_samples", n_samples)

def generate_batch(index):
  return batch[index:index+batch_size], labels[index:index+batch_size]

with tf.Session(graph=graph) as session:

  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in range(num_steps):

    # create and feed training batch to graph
    batch_inputs, batch_labels = generate_batch(batch_index)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    
    # update batch - add shuffle here?
    batch_index += batch_size
    if batch_index > (n_samples-batch_size): batch_index = 0
    

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    # give the user some feedback
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000

      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0


    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:

      sim = similarity.eval()
      for i in range(valid_size):
        
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to %s:" % valid_word
        
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        
        print(log_str)
  
  final_embeddings = normalized_embeddings.eval()




################################################################################
# Visualize the embeddings.
####################################################################################
def plot_with_labels(low_dim_embs, labels, filename='buzz_feed_vectors.png'):

  plt.figure(figsize=(20, 20))  # in inches
  
  for i, label in enumerate(labels):
  
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)
  plt.show()
  

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 400

low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]

plot_with_labels(low_dim_embs, labels)
  




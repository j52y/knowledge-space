'''Common functions
'''

import itertools
import numpy as np
import nltk
from collections import defaultdict


def softmax(x):
   maximum = np.max(x)
   norm = x - maximum
   numer = np.exp(norm)
   denom = np.sum(numer)
   return numer / denom


def softmax_batch(x):
   maximum = np.max(x, axis=1, keepdims=True)
   norm = x - maximum
   numer = np.exp(norm)
   denom = np.sum(numer, axis=1, keepdims=True)
   return numer / denom


def softmax_derivative(y):
   l = y.shape[0]
   der = np.outer(y, y)
   der -= y * np.eye(l)
   return -der


def softmax_derivative_batch(y):
   l = y.shape[1]
   der = y.T.dot(y)
   col_sum = np.sum(y, axis=1, keepdims=True)
   der -= col_sum * np.eye(l)
   return -der


def cross_entropy_one_hot_derivative(y, n):
   dEdy = np.copy(y)
   dEdy[n] -= 1
   return dEdy


def cross_entropy_one_hot_derivative_batch(y, n):
   dEdy = np.copy(y)
   indice = np.arange(n.shape[0])
   dEdy[indice, n] -= 1
   return dEdy


def softmax_cross_entropy_batch(y, n):
   l = y.shape[1]
   dEdy = cross_entropy_one_hot_derivative_batch(y, n)
   mul = dEdy * y
   der = y.T.dot(mul)
   col_sum = np.sum(mul, axis=1, keepdims=True)
   der -= col_sum * np.eye(l)
   return -der


def matrix_multi_derivative(w, x, dydx=True, x_is_multiplier=True):
   if w.ndim != 2 or x.ndim != 1:
       raise Exception('dimension error')

   m = w.shape[0]
   n = w.shape[1]

   if x_is_multiplier:
       '''y = np.dot(w,x)  m by n dot n by 1
       if multiplicand == True, return dy/dw
       else return dy/dx
       '''
       if dydx:
           # dy/dx
           return w.T
       else:
           # dy/dw
           tmpx = np.tile(x, n).reshape(m * n, 1)
           shape = np.eye(m).repeat(n, axis=0)
           return shape * tmpx
   else:
       '''y = np.dot(x,w) 1 by m dot m by n
       '''
       if dydx:
           return w
       else:
           shape = np.tile(np.eye(n), m).T
           tmpx = x.repeat(n).reshape(m * n, 1)
           return shape * tmpx


def one_hot_multiple(voca_size, targets):
   return np.eye(voca_size)[targets]


def one_hot(voca_size, target):
   one_hot = np.zeros(voca_size)
   one_hot[target] = 1
   return one_hot.reshape(1, voca_size)


def cos_sim(embedding):
   mul = embedding.dot(embedding.T)
   mag = np.diag(mul)
   inv_mag = 1 / mag
   inv_mag[np.isinf(inv_mag)] = 0
   inv_mag = np.sqrt(inv_mag)
   sim = mul * inv_mag
   return sim.T * inv_mag


# corpus_file = 'data-concept-small.txt'
corpus_file = 'data-concept.txt'

def read_corpus(corpus_file=corpus_file):
  cooc = defaultdict(lambda: defaultdict(lambda: 0.0))
  words = defaultdict(lambda: 0)
  total = {}
  with open(corpus_file) as f:
    for l in f:
      c1, c2, n = l.split('\t')
      n = int(n)
      words[c1] += n
      words[c2] += n   
      cooc[c1][c2] = n
  return cooc, words


# glove_file = 'glove_file.txt'
glove_file = 'test.txt'

def read_glove(glove_file=glove_file):
  cooc = defaultdict(lambda: defaultdict(lambda: 0.0))
  m = 0
  with open(glove_file) as f:
    for l in f:
      c1, c2, v = l.split(' ')
      cooc[int(c1)][int(c2)] = float(v)
      l = max(int(c1), int(c2))
      m = max(l, m)
  return cooc, m+1


def words(sentences):
   tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
   words = list(itertools.chain(*tokenized_sentences))
   print("%d words" % len(words))
   return words


def word_index(words):
   i2w = [w for w in sorted(words, key=lambda x:(words.get(x),x), reverse=True)]
   w2i = dict([(word, i+1) for i, word in enumerate(i2w)])  # word to index
   return w2i, i2w


cooc_file = 'probase_cooc.txt'
probase_voca = 'probase_vocab.txt'

def save_cooc():
  cooc, words = read_corpus()
  w2i, i2w = word_index(words)
  with open(cooc_file, 'w') as f:
    for c in cooc:
      i = w2i[c]
      for w in cooc[c]:
        j = w2i[w]
        f.write("%d %d %f\n" % (i, j, cooc[c][w])) 

  with open(probase_voca, 'w') as f:
    for w in i2w: 
      f.write("%s\t%d\n" % (w.replace(' ', '%'), words[w]))

def train_data(words, w2i, i2w):
   '''window size = 1
   '''
   length = len(words)
   data = []
   for i in range(1, length - 1):
       left = [w2i[words[i]], w2i[words[i + 1]]]
       right = [w2i[words[i]], w2i[words[i - 1]]]
       data.append(left)
       data.append(right)

   data = np.array(data)

   train_x = data[:, 0]
   train_y = data[:, 1]
   return train_x, train_y

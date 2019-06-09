import os
import tensorflow as tf
print('TensorFlow:{}'.format(tf.__version__))
import numpy as np
print('NumPy:{}'.format(np.__version__))
import matplotlib.pyplot as plt
import matplotlib
print('Matplotlib:{}'.format(matplotlib.__version__))

from tensorflow.contrib.tensorboard.plugins import projector

batch_size = 64
embedding_dimension = 5
negative_samples = 8
LOG_DIR = 'logs/word2vec_intro'

digit_to_word_map = dict({1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
                          6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'})
sentences  = []


for i in range(10000):
    rand_odd_ints = np.random.choice(range(1, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

#print(sentences[0:10])

word2index_map = {}
index = 0
for sent in sentences:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
# Inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

# Generate skip-gram pairs
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent)-1):
        word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                              word2index_map[tokenized_sent[i+1]]],
                             word2index_map[tokenized_sent[i]]
                             ]
        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][0]])
        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][1]])

def get_skipgram_batch(batch_size):
    instance_indeces = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indeces)
    batch = instance_indeces[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [skip_gram_pairs[i][1] for i in batch]
    return x, y

print(skip_gram_pairs[0:10])

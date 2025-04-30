import jax.numpy as jnp
import jax
import numpy as np
import optax
import tqdm
import pickle
import matplotlib.pyplot as plt
import collections

# def generate_train_vectors(train_data, vocab, window_size=4, batch_size=128):
#     """Generates training vectors from a list of words and vocabulary.

#     Generates (target_batch, context_batch) pairs, where target_batch is a
#     (batch_size,) array of target word IDs and context_batch is a
#     (batch_size, 2*window_size) array of context word IDs.

#     Stops when it runs out of data. The leftover data (whatever doesn't fit in
#     the last batch) will be discared.
#     """
#     target_batch = np.zeros(batch_size, dtype=np.int32)
#     context_batch = np.zeros((batch_size, 2 * window_size), dtype=np.int32)
#     batch_idx = 0

#     for i in range(len(train_data)):
#         if i + 2 * window_size >= len(train_data):
#             break

#         # 'i' is the index of the leftmost word in the context.
#         target_word = train_data[i + window_size]
#         left_context = train_data[i : i + window_size]
#         right_context = train_data[i + window_size + 1 : i + 2 * window_size + 1]

#         target_batch[batch_idx] = vocab.get(target_word, 0)
#         context_batch[batch_idx, :] = np.array(
#             [vocab.get(word, 0) for word in left_context + right_context]
#         )

#         batch_idx += 1
#         if batch_idx == batch_size:
#             yield np.array(target_batch), np.array(context_batch)
#             batch_idx = 0


# def generate_train_vectors_with_negative_samples(train_data, vocab, window_size=4, batch_size=128, n_negatives=5):
#     """Generates training vectors from a list of words and vocabulary.

#     Generates (target_batch, context_batch) pairs, where target_batch is a
#     (batch_size,) array of target word IDs and context_batch is a
#     (batch_size, 2*window_size) array of context word IDs.

#     Stops when it runs out of data. The leftover data (whatever doesn't fit in
#     the last batch) will be discared.
#     """
#     target_batch = np.zeros(batch_size, dtype=np.int32)
#     context_batch = np.zeros((batch_size, 2 * window_size), dtype=np.int32)
#     batch_idx = 0
#     neg_batch = np.zeros((batch_size, n_negatives), dtype=np.int32)
    
#     all_words = np.array(list(set(vocab.values()) )) # to keep track of the negative samples

#     # word frequencies for negative sampling
#     word_freqs = collections.Counter(train_data)
#     words_total = len(train_data)
#     word_freqs = {w: f/words_total for w,f in dict(word_freqs).items()}
#     # instead of words, translate to token-ids
#     token_freqs = {ii: word_freqs[word] if word in word_freqs else 0 for word, ii in vocab.items()}
#     all_words, all_freqs = zip(*token_freqs.items())

#     def sample_neg_batch(bsize, n_negatives):
#         # need to allow replacement, otherwise we never get the same thing twice
#         p = np.array(all_freqs) ** 0.75
#         p = p / p.sum()
#         return np.random.choice(all_words, p=p, replace=True, size=(bsize,n_negatives))
    
#     for i in range(len(train_data)):
#         if i + 2 * window_size >= len(train_data):
#             break

#         # 'i' is the index of the leftmost word in the context.
#         target_word = train_data[i + window_size]
#         left_context = train_data[i : i + window_size]
#         right_context = train_data[i + window_size + 1 : i + 2 * window_size + 1]
#         context= left_context + right_context
#         target_batch[batch_idx] = vocab.get(target_word, 0)
#         context_batch[batch_idx, :] = np.array(
#             [vocab.get(word, 0) for word in context]
#         )

#         # neg samples
#         # used_words = (set(context_batch[batch_idx, :]) | set([target_word]))  # the word appearing in context
#         # potential = list(all_words - used_words)
#         # neg_batch[batch_idx, :] = np.random.choice(potential, replace=False, size=n_negatives)
#         # neg_batch[batch_idx, :] = np.random.choice(all_words, replace=False, size=n_negatives)
#         # neg_batch[batch_idx, :] = np.random.choice(all_words, p=all_freqs, replace=False, size=n_negatives)
        
#         # neg_candidates = [w for w in np.random.choice(all_words,p=all_freqs, size=3*n_negatives) if w != target_batch[batch_idx] and not w in context_batch[batch_idx, :]]
#         # neg_candidates = [w for w in np.random.choice(all_words, size=3*n_negatives) if w != target_batch[batch_idx] and not w in context_batch[batch_idx, :]]
#         # neg_batch[batch_idx, :] = neg_candidates[:n_negatives]

#         # if target_batch[batch_idx] in neg_batch[batch_idx, :]:
#         #     print('target in neg')

#         # if len(set(context_batch[batch_idx])& set(neg_batch[batch_idx]) ) > 0:
#         #     print('context in neg')
        
#         batch_idx += 1
#         if batch_idx == batch_size:
#             neg_batch = sample_neg_batch(batch_size, n_negatives)
#             yield np.array(target_batch), np.array(context_batch), np.array(neg_batch)
#             batch_idx = 0



class Loader:
    def __init__(self, train_data, vocab, shuffle=True):
        """
        :param shuffle: randomly shufle the samples in a minibatch?
        """
        self.train_data = train_data,
        self.vocab = vocab
        self.train_data_ids = [vocab.get(w, 0) for w in train_data]
        self.shuffle = shuffle

    def get_iterator(self, window_size, batch_size):
        target_batch = np.zeros(batch_size, dtype=np.int32)
        context_batch = np.zeros((batch_size, 2 * window_size), dtype=np.int32)
        batch_idx = 0 
        
        for i in range(len(self.train_data_ids)):
            if i + 2 * window_size >= len(self.train_data_ids):
                break
    
            # 'i' is the index of the leftmost word in the context.
            target_word = self.train_data_ids[i + window_size]
            left_context = self.train_data_ids[i : i + window_size]
            right_context = self.train_data_ids[i + window_size + 1 : i + 2 * window_size + 1]
    
            target_batch[batch_idx] = target_word
            context_batch[batch_idx, :] = np.array(left_context + right_context)
    
            batch_idx += 1
            if batch_idx == batch_size:

                target_batch = np.array(target_batch) 
                context_batch = np.array(context_batch)
                if self.shuffle:
                    # Shuffle the batch.
                    indices = np.random.permutation(len(target_batch))
                    target_batch = target_batch[indices]
                    context_batch = context_batch[indices]
                
                yield target_batch, context_batch
                batch_idx = 0



class Loader_Negative:
    def __init__(self, train_data, vocab, shuffle=True):
        """
        :param shuffle: randomly shufle the samples in a minibatch?
        """
        self.train_data = train_data,
        self.vocab = vocab
        self.train_data_ids = [vocab.get(w, 0) for w in train_data]
        self.shuffle = shuffle
        
        # some prep for neg sampling
        all_words = np.array(list(set(vocab.values()) )) # to keep track of the negative samples

        # word frequencies for negative sampling
        word_freqs = collections.Counter(train_data)
        words_total = len(train_data)
        word_freqs = {w: f/words_total for w,f in dict(word_freqs).items()}
        # instead of words, translate to token-ids
        token_freqs = {ii: word_freqs[word] if word in word_freqs else 0 for word, ii in vocab.items()}
        all_words, all_freqs = zip(*token_freqs.items())
        self.all_words = all_words
        self.all_freqs = all_freqs

    def sample_neg_batch(self, batch_size, n_negatives):
        """
        samples an ENTIRE batch of negative, i.e. the result will be (batch_size, n_negatives)
        """
        # need to allow replacement, otherwise we never get the same thing twice
        p = np.array(self.all_freqs) ** 0.75
        p = p / p.sum()
        return np.random.choice(self.all_words, p=p, replace=True, size=(batch_size,n_negatives))


    def get_iterator(self, window_size, batch_size, n_negatives):
        target_batch = np.zeros(batch_size, dtype=np.int32)
        context_batch = np.zeros((batch_size, 2 * window_size), dtype=np.int32)
        batch_idx = 0 
        
        for i in range(len(self.train_data_ids)):
            if i + 2 * window_size >= len(self.train_data_ids):
                break
    
            # 'i' is the index of the leftmost word in the context.
            target_word = self.train_data_ids[i + window_size]
            left_context = self.train_data_ids[i : i + window_size]
            right_context = self.train_data_ids[i + window_size + 1 : i + 2 * window_size + 1]
    
            target_batch[batch_idx] = target_word
            context_batch[batch_idx, :] = np.array(left_context + right_context)

            # neg samples
            # used_words = (set(context_batch[batch_idx, :]) | set([target_word]))  # the word appearing in context
            # potential = list(all_words - used_words)
            # neg_batch[batch_idx, :] = np.random.choice(potential, replace=False, size=n_negatives)
            # neg_batch[batch_idx, :] = np.random.choice(all_words, replace=False, size=n_negatives)
            # neg_batch[batch_idx, :] = np.random.choice(all_words, p=all_freqs, replace=False, size=n_negatives)
            
            # neg_candidates = [w for w in np.random.choice(all_words,p=all_freqs, size=3*n_negatives) if w != target_batch[batch_idx] and not w in context_batch[batch_idx, :]]
            # neg_candidates = [w for w in np.random.choice(all_words, size=3*n_negatives) if w != target_batch[batch_idx] and not w in context_batch[batch_idx, :]]
            # neg_batch[batch_idx, :] = neg_candidates[:n_negatives]
    
            # if target_batch[batch_idx] in neg_batch[batch_idx, :]:
            #     print('target in neg')
    
            # if len(set(context_batch[batch_idx])& set(neg_batch[batch_idx]) ) > 0:
            #     print('context in neg')

            batch_idx += 1
            if batch_idx == batch_size:
                neg_batch = self.sample_neg_batch(batch_size, n_negatives)
                target_batch = np.array(target_batch)
                context_batch = np.array(context_batch)
                
                if self.shuffle:
                    # Shuffle the batch.
                    indices = np.random.permutation(len(target_batch))
                    target_batch = target_batch[indices]
                    context_batch = context_batch[indices]
                    neg_batch = neg_batch[indices]

                yield target_batch, context_batch, neg_batch
                batch_idx = 0

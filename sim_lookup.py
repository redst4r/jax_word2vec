from sklearn.neighbors import NearestNeighbors

class SimilarityLookup():
    def __init__(self, embeddings, vocab, knn, distance_metric='euclidean'):
        self.embeddings = embeddings
        self.knn = knn
        self.vocab = vocab
        self.inv_vocab= {the_id: the_word for the_word, the_id in vocab.items()}
        self.nn = NearestNeighbors(n_neighbors=knn, metric=distance_metric)
    def fit(self):
        self.nn.fit(self.embeddings)

    def query(self, query_word):
        q = self.get_embedding(query_word)
        terms_and_dist = self.get_nn_from_embedding(q)
        return terms_and_dist

    def get_embedding(self, query_word):
        """translate the word to a vector"""
        return self.embeddings[self.vocab[query_word]].reshape(1,-1)


    def get_nn_from_embedding(self, query_embedding):
        """look up the words closted to the vector"""
        # return self.embeddings[self.vocab[query_word]].reshape(1,-1)
        dist, ind = self.nn.kneighbors(query_embedding)
        terms = [self.inv_vocab[_] for _ in ind.flatten()]
        terms_and_dist = {self.inv_vocab[i]: d for i,d  in zip(ind.flatten(), dist.flatten())}
        return terms_and_dist
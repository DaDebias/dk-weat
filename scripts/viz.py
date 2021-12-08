
from utility_functions import *
import sys, os
sys.path.append(os.path.join('..', '..'))
from gensim.models.keyedvectors import KeyedVectors
import plotly
import numpy as np
import plotly.graph_objs as go
#from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# load model 
k_model_w2v = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 

# test it is loaded
k_model_w2v.most_similar("hende")

# define restrict words func
def restrict_wv(wv, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(wv.vocab)):
        word = wv.index2entity[i]
        vec = wv.vectors[i]
        vocab = wv.vocab[word]
        vec_norm = wv.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            new_vectors_norm.append(vec_norm)

    wv.vocab = new_vocab
    wv.vectors = np.array(new_vectors)
    wv.index2entity = np.array(new_index2entity)
    wv.index2word = np.array(new_index2entity)
    wv.vectors_norm = np.array(new_vectors_norm)


# define plot func
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(n_components=2, perplexity =10)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(12, 12), dpi=600) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig('tsne_plot.png')
    plt.show()



# define words to plot 
#words_to_plot = {"mand", "kvinde", "pige", "dreng", "han", "hun", "feminin", "maskulin"}
#w1 = ["mand", "kvinde", "pige", "dreng", "han", "hun", "feminin", "maskulin"]
#w2 = ["haha", "kvinde", "pige", "dreng", "han", "hun", "feminin", "maskulin"]

#w3 = set(w1+w2)


# restrict embedding
#restrict_wv(wv = k_model_w2v, restricted_word_set = w3)

# plot
#tsne_plot(k_model_w2v)



# kenneths plot func (takes in a list)
#ax = plot_word_embeddings(words=words_to_plot, embedding=k_model_w2v)
#ax.plot()



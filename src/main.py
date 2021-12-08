#from gensim.models.keyedvectors import KeyedVectors
from utility_functions import *
import sys, os
sys.path.append(os.path.join('..', '..'))
from cool_programmer_tshirts.debiaswe.we import WordEmbedding
import pkg_resources
pkg_resources.require("gensim==3.8.0") 

model_w2v = WordEmbedding('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin') 
model_fasttext = WordEmbedding('/work/dagw_wordembeddings/fasttext_model/fasttext.txt');
#model_w2v = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 
#model_fasttext = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/fasttext_model/fasttext.txt', binary=False)
#model_fasttext1 = WordEmbedding('/work/dagw_wordembeddings/fasttext_model/fasttext_vectors.txt')#, binary=False) 


#Set permutations
p =10000

#Set target and attribute words - Career vs. Family
A = ['mandlig', 'mand','dreng','bror','han','ham','hans','søn'] #Attribute words for mand
B = ['kvindelig', 'kvinde', 'pige', 'søster', 'hun', 'hende', 'hendes', 'datter'] #Attribute words for kvinde
X = ['leder', 'ledelse', 'professionel', 'virksomhed', 'løn', 'kontor', 'forretning', 'karriere'] #Target words for Career
Y = ['hjem','forældre', 'børn', 'familie','bedsteforældre', 'ægteskab', 'bryllup', 'pårørende'] #Target words for Family

print(results_weat(X, Y, A, B, model_fasttext.index, p))

print(results_weat(X, Y, A, B, model_w2v.index, p))

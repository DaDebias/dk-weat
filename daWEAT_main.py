from gensim.models.keyedvectors import KeyedVectors
from daWEAT_func import *

model_test = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 

#Set permutations
p =10000

#Set target and attribute words - Science vs. Arts
A = ['mandlig', 'mand','dreng','bror','han','ham','hans','søn'] #Attribute words for mand
B = ['kvindelig', 'kvinde', 'pige', 'søster', 'hun', 'hende', 'hendes', 'datter'] #Attribute words for kvinde
X = ['videnskab', 'teknologi', 'fysik', 'kemi', 'rumfart', 'eksperiment', 'astronomi', 'biologi'] #Target words for Science
Y = ['poesi', 'kunst', 'dans', 'litteratur', 'roman', 'symfoni', 'drama', 'skulptur'] #Target words for Arts

print(results_weat(X, Y, A, B, model_test, p))


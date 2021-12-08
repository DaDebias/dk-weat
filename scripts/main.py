
from utility_functions import *
from viz import restrict_wv
from gensim.models.keyedvectors import KeyedVectors


def weat_func(wordembedding, permutations, male, female, word_list1, word_list2):
    #Set permutations
    p = permutations

    #Set target and attribute words - Career vs. Family
    sub1 = word_list1
    sub2 = word_list2
    
    #run the WEAT
    results = results_weat(sub1, sub2, male, female, wordembedding, p)

    return print(results)

#import sys, os
#sys.path.append(os.path.join('..', '..'))
#from cool_programmer_tshirts.debiaswe.we import WordEmbedding
#k_model_fasttext = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/fasttext_model/fasttext.txt', binary=False)

k_model_w2v = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 


male = ['mandlig', 'mand','dreng','bror','han','ham','hans','søn'] #Attribute words for mand
female = ['kvindelig', 'kvinde', 'pige', 'søster', 'hun', 'hende', 'hendes', 'datter'] #Attribute words for kvinde

science = ['videnskab', 'teknologi', 'fysik', 'kemi', 'computer', 'eksperiment', 'data', 'biologi', 'mand'] #Target words for Science
arts = ['poesi', 'kunst', 'dans', 'litteratur', 'roman', 'symfoni', 'drama', 'skulptur', 'kvinde'] #Target words for Arts
math = ['matematik', 'algebra', 'geometri', 'regning', 'ligning', 'beregning', 'tal', 'addition'] #Target words for Math
career = ['leder', 'bestyrelse', 'professionel', 'virksomhed', 'løn', 'arbejde', 'forretning', 'karriere'] #Target words for Career
family = ['hjem','forældre', 'børn', 'familie','bedsteforældre', 'ægteskab', 'bryllup', 'pårørende'] #Target words for Family

weat_func(model, 10000, male, female, career, family)
weat_func(model, 10000, male, female, science, arts)
weat_func(model, 10000, male, female, math, arts)

words_to_plot = set(male)

# restrict embedding
restrict_wv(wv = k_model_w2v, restricted_word_set = words_to_plot)

# plot
tsne_plot(k_model_w2v)



'''
#hjemmelavet
Z = ['stærk', 'beslutsom', 'muskler', 'forsørger', 'helt', 'modig', 'kriger', 'stor'] #Target words for Career
W = ['svag','kærlig', 'diversitet', 'smuk','lille', 'underdanig', 'kreativ', 'hjemmegående'] #Target words for Family

'''
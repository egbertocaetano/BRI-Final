
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import xml.etree.ElementTree as ET
from nltk import word_tokenize
import re
import codecs
import reader
import numpy as np
import pickle

def score_sim(matriz_queries, matriz_corpus):
    
    return pairwise_distances(matrix_queries,matrix_corpus, 'cosine', -1)
    

def ranking(matriz_queries, matriz_corpus):
    
    pairwise_results = score_sim(matriz_queries, matriz_corpus)
    
    indices = []
    distancias = []
    
    for j in range(len(pairwise_results)):

        indices.append([i[0] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1])])
        distancias.append([i[1] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1])])
    
    return indices, distancias


def f1_score(precision, recall):
    
    if(precision + recall == 0):
        return 0

    return 2*((precision*recall)/(precision+recall))


def metricas(targets, id_sources, indices, distancias):
    
    precisions = np.zeros(indices.shape[1])
    recalls = np.zeros(indices.shape[1])
    f1 = np.zeros(indices.shape[1])
    hfm = np.zeros(len(targets))
    separation = np.zeros(len(targets))

    for i in range(len(targets)):
        
        relevantes = 0
        retornados = 0
        
        rankingi = indices[i]
        targeti = targets[i]
        
        total_relevantes = len(rankingi)
        
        for j in range(len(rankingi)):
            
            retornados = j+1
        
            if rankingi[j] in targeti:
                relevantes += 1
                separation[i] = (distancias[i][j] * 100. / distancias[i][0]) - hfm[i]

            elif hfm[i] == 0:
                hfm[i] =  distancias[i][j] * 100. / distancias[i][0]
                
            precisions[j] = precisions[j] + relevantes/retornados
            recalls[j] = recalls[j] + relevantes/total_relevantes
            f1[j] = f1[j] + f1_score(precisions[j], recalls[j])
        
    precisions /= len(targets)
    recalls /= len(targets)  
    f1 /= len(targets)

   
    return precisions, recalls, f1, hfm, separation


def tokenize_text(text):
    return word_tokenize(text.lower())


def remove_stopwords(tokens):
    
    no_stopwords = []
    stoplist = set('for a of the and to in'.split())
    for token in tokens:
        if len(token) > 1 and token.isalpha() and not token in stoplist:
            no_stopwords.append(token)
    
    return no_stopwords


def mount_vocab_source(l):
    vocab = []
    corpus = []
    id_sources = {}
    i = 0

    for path in l[1:]:
        if(path[-4:] == ".txt"):
            
            file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
            text = file.read()

            vocab += remove_stopwords(tokenize_text(text))                
            vocab = list(set(vocab))
            id_sources[path.split("/")[-1]] = i
            corpus.append( text.lower() )
            i += 1
    
    
    return vocab, corpus, id_sources


def mount_vocab_querie(l, id_sources):
    vocab = []
    queries = []
    targets = []
    
    for path in l[1:]:
        if(path[-4:] == ".txt"):
           
            file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
            text = file.read()
        
            vocab += remove_stopwords(tokenize_text(text))                
            vocab = list(set(vocab))
            queries.append( text.lower() )

            tree = ET.parse(path[:-4]+".xml")
            target_aux = list()
            for feature in tree.iter("feature"):
                if(feature.get("name") == "plagiarism") and feature.get("source_reference") in id_sources and id_sources[feature.get("source_reference")] not in target_aux:
                    target_aux.append(id_sources[feature.get("source_reference")])
            targets.append(target_aux)
    
    
    return vocab, queries, targets

    
#r = reader.Reader("/home/jones/external-detection-corpus/source-document/")
r = reader.Reader("/home/jones/pan-plagiarism-corpus-2011_alterado/external-detection-corpus/source-document/")
l = r.get_paths()

print("Montando vocabulario source")
vocab_source, corpus, id_sources = mount_vocab_source(l)

#r = reader.Reader("/home/jones/external-detection-corpus/suspicious-document/")
r = reader.Reader("/home/jones/pan-plagiarism-corpus-2011_alterado/external-detection-corpus/suspicious-document/")
l = r.get_paths()

print("Montando vocabulario querie")
vocab_queries, queries, targets = mount_vocab_querie(l, id_sources)

vocab = list(set(vocab_queries + vocab_source))
cv = CountVectorizer(vocabulary = vocab)

del vocab_queries, vocab_source, vocab, r, l

print("Contruindo os vetores de frequencias das palavras das queries")
matrix_queries = cv.fit_transform(queries).toarray()

print("Contruindo os vetores de frequencias das palavras do corpus")
matrix_corpus = cv.fit_transform(corpus).toarray()

print("Montando espaço ball_tree n = 40")
neigh = NearestNeighbors(n_neighbors=40, algorithm='ball_tree')
neigh.fit(matrix_corpus)

distancias, indices = neigh.kneighbors(matrix_queries, 40)

print("Resultados")
precisions, recalls, f1, hfm, separation = metricas(targets, id_sources, indices, distancias)

pickle.dump(precisions, open("/home/jones/resultados/precisions_bt.txt", "wb"))
pickle.dump(recalls, open("/home/jones/resultados/recalls_bt.txt", "wb"))
pickle.dump(f1, open("/home/jones/resultados/f1_bt.txt", "wb"))
pickle.dump(hfm, open("/home/jones/resultados/hfm_bt.txt", "wb"))
pickle.dump(separation, open("/home/jones/resultados/separation_bt.txt", "wb"))

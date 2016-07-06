
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
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
    
    return np.array(indices), np.array(distancias)


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
        
            if targeti[rankingi[j]] == 1:
                relevantes += 1
                separation[i] = (distancias[i][j] * 100. / distancias[i][0]) - hfm[i]

            elif hfm[i] == 0:
                hfm[i] = distancias[i][j] * 100. / distancias[i][0]
                
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

            target = zeros(len(id_sources))
            tree = ET.parse(path[:-4]+".xml")
            target_aux = list()
            for feature in tree.iter("feature"):
                if(feature.get("name") == "plagiarism") and feature.get("source_reference") in id_sources and id_sources[feature.get("source_reference")] not in target_aux:
                    target[id_sources[feature.get("source_reference")]] = 1

            targets.append(target)

    return vocab, queries, targets

def extract_pan():
    if(os.path.exists("/home/jones/sources.txt"))
        corpus = pickle.load(open("/home/jones/sources.txt", "rb"))
        queries = pickle.load(open("/home/jones/queries.txt", "rb"))
        targets = pickle.load(open("/home/jones/targets.txt", "rb"))

        return queries, corpus, targets

    r = reader.Reader("/home/jones/external-detection-corpus/source-document/")
    l = r.get_paths()

    print("Montando vocabulario source")
    vocab_source, corpus, id_sources = mount_vocab_source(l)

    r = reader.Reader("/home/jones/external-detection-corpus/suspicious-document/")
    l = r.get_paths()

    print("Montando vocabulario querie")
    vocab_queries, queries, targets = mount_vocab_querie(l, id_sources)

    vocab = list(set(vocab_queries + vocab_source))

    pickle.dump(vocab, open("/home/jones/vocab.txt", "wb"))
    pickle.dump(corpus, open("/home/jones/sources.txt", "wb"))
    pickle.dump(queries, open("/home/jones/queries.txt", "wb"))
    pickle.dump(id_sources, open("/home/jones/id_sources.txt", "wb"))
    pickle.dump(targets, open("/home/jones/targets.txt", "wb"))

    return queries, corpus, targets


if __name__ == '__main__':
    
    queries, corpus, targets = extract_pan()
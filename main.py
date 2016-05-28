from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from nltk import word_tokenize
import re
import codecs
import reader
import numpy as np

def score_sim(matriz_queries, matriz_corpus):
    
    matriz_similaridade = pairwise_distances(matrix_queries,matrix_corpus, 'cosine', -1)
    
    return matriz_similaridade

def ranking(matriz_queries, matriz_corpus):
    
    pairwise_results = score_sim(matriz_queries, matriz_corpus)
    
    indices = []
    distancias = []
    
    for j in range(len(pairwise_results)):

        indices.append([i[0] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1])])
        distancias.append([i[1] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1])])
    
    return indices, distancias

def metricas(target, ranking_consulta):
    
    precisions = np.zeros((target.shape[1]))
    recalls = np.zeros((target.shape[1]))
    
    ks = np.array([i for i in range(target.shape[1])])
    
    for i in range(queries_index.shape[0]):# target tem o mesmo tamanho da consulta
            
        relevantes = 0
        retornados = 0
        
        consultai = ranking_consulta[i]
        targeti = target[i]
        
        total_relevantes = np.sum(targeti)
        
        for j in range(len(consultai)):
            
            retornados = j+1
            
            pos = consultai[j]
        
            if (targeti[0,pos] == 1):
                relevantes += 1
                
            precisions[j] = precisions[j] + relevantes/retornados
            recalls[j] = recalls[j] + relevantes/total_relevantes
            
    precisions = precisions / matriz_queries.shape[0]
    
    recalls = recalls / matriz_queries.shape[0]        
   
    return precisions, recalls, ks 

def tokenize_text(text):
	return word_tokenize(text.lower())


def remove_stopwords(tokens):
	
	no_stopwords = []
	stoplist = set('for a of the and to in'.split())
	for token in tokens:
		if len(token) > 1 and token.isalpha() and not token in stoplist:
			no_stopwords.append(token)
	
	return no_stopwords

def mount_vocab(l):
	queries = []
	corpus = []
	id_path = []
	id_source = []
	vocab = []
	for path in l[1:]:
		file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
		text = file.read()
		if path.find("source") == -1:
			id_path.append(path)
			vocab += remove_stopwords(tokenize_text(text))
			queries.append( text.lower() )
		else:
			id_source.append(path)
			vocab += remove_stopwords(tokenize_text(text))
			corpus.append( text.lower() )
		
	vocab = list(set(vocab))
	
	return vocab, queries, corpus, id_path, id_source

	

r = reader.Reader("data/spa_corpus/corpus-20090418")
l = r.get_paths()
vocab, queries, corpus, id_path, id_source = mount_vocab(l)

cv = CountVectorizer(vocabulary = vocab)

matrix_queries = cv.fit_transform(queries).toarray()
matrix_corpus = cv.fit_transform(corpus).toarray()

indices, distances = ranking(matrix_queries, matrix_corpus)

pos = 23
print(id_path[pos], " - ", id_source[indices[pos][0]],id_source[indices[pos][1]], id_source[indices[pos][2]], id_source[indices[pos][3]], id_source[indices[pos][4]], " / ", distances[pos])
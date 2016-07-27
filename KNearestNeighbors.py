from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

def ranking(pairwise_results):
    
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


def metricas(targets, indices, distancias):
    indices = np.array(indices)
    distancias = np.array(distancias)

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
        
        total_relevantes = np.sum(targeti)
        
        for j in range(len(rankingi)):
            
            retornados = j+1
        	
            if targeti[rankingi[j]] == 1:
                relevantes += 1
                separation[i] = (distancias[i][j] * 100. / distancias[i][0]) - hfm[i]

            elif hfm[i] == 0:
                hfm[i] =  distancias[i][j] * 100. / distancias[i][0]
            
            precision = relevantes/retornados
            recall = relevantes/total_relevantes

            precisions[j] = precisions[j] + precision
            recalls[j] = recalls[j] + recall

            f1[j] = f1[j] + f1_score(precision, recall)

    precisions /= len(targets)
    recalls /= len(targets)  
    f1 /= len(targets)

   
    return precisions, recalls, f1, hfm, separation


all_texts = pickle.load(open("/home/jones/resultados/at.txt", "rb"))
ft = pickle.load(open("/home/jones/resultados/ft.txt", "rb"))
targets = pickle.load(open("/home/jones/resultados/targets.txt", "rb"))
tam_queries = pickle.load(open("/home/jones/resultados/tam_queries.txt", "rb"))
tam_corpus = pickle.load(open("/home/jones/resultados/tam_corpus.txt", "rb"))

sentences_docs = []
sentences_queries = []
count = 0
mapa = {}
root_sentences_docs = []
root_sentences_queries = []

for i in range(1, 2*tam_corpus, 2):
	l = []
	for sentence in ft[i]:
		root_sentences_docs.append(sentence[2][0])#conjunto de todos os vetores raizes de todas as sentencas
		l.append(np.sum(sentence[2], axis=0))#conjunto de todos os vetores de todos os nós de todas as sentencas
		mapa[len(root_sentences_docs) - 1 ] = count#mapeando cada vetor raiz com seu documento respectivo
	sentences_docs.append(np.sum(l, axis=0))	
	count += 1

for i in range(0, len(ft), 2*tam_corpus):
	l = []
	j = []
	for sentence in ft[i]:
		j.append(np.sum(sentence[2], axis=0))#conjunto de todos os vetores de todos os nós de todas as sentencas
		l.append(sentence[2][0])
	sentences_queries.append(np.sum(j, axis=0))
	root_sentences_queries.append(l)#conjunto de todos os vetores raizes de todas as sentencas


docs_results = [np.inf]*tam_queries*tam_corpus
docs_results = np.array(docs_results).reshape(tam_queries, tam_corpus)

for i, sentences in enumerate(root_sentences_queries):
	nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(root_sentences_docs)
	distances, indices = nbrs.kneighbors(sentences)
	suspects = []
	sources = []
	for posicao in indices:
		suspects.append(mapa[ posicao[0] ])
	
	for doc in set(suspects):
		docs_results[i][doc] = pairwise_distances(sentences_queries[i].reshape(1,-1), sentences_docs[doc].reshape(1,-1), 'euclidean')[0][0]

indices, distances = ranking(docs_results)
precision, recall, f1, hfm, separation = metricas(targets, indices, distances)
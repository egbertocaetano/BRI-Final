from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

def ranking(pairwise_results):
    
    indices = []
    
    for j in range(len(pairwise_results)):

        indices.append([i[0] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1], reverse=True)])
        
    return indices

def metricas(targets, indices):
    
    indices = np.array(indices)

    recalls = np.zeros(indices.shape[1])
    selectivitys = np.zeros(indices.shape[1])
  
    for i in range(len(targets)):
        relevantes = 0

        rankingi = indices[i]
        targeti = targets[i]
        
        total_relevantes = np.sum(targeti)
        
        for j in range(len(rankingi)):
        	
            if targeti[rankingi[j]] == 1:
                relevantes += 1

            recalls[j] += relevantes / total_relevantes
            selectivitys[j] += (j+1) / indices.shape[1]
            
    
    selectivitys /= len(targets)
    print(recalls)
    recalls /= len(targets)
    print(recalls)
    return recalls, selectivitys

'''
all_texts = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/at.txt", "rb"))
ft = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/ft.txt", "rb"))
targets = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/targets.txt", "rb"))
tam_queries = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/tam_queries.txt", "rb"))
tam_corpus = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/tam_corpus.txt", "rb"))
'''

all_texts = pickle.load(open("/home/jones/Documentos/resultados/hu0_n200/at.txt", "rb"))
ft = pickle.load(open("/home/jones/Documentos/resultados/hu0_n200/ft.txt", "rb"))
targets = pickle.load(open("/home/jones/Documentos/resultados/hu0_n200/targets.txt", "rb"))
tam_queries = pickle.load(open("/home/jones/Documentos/resultados/hu0_n200/tam_queries.txt", "rb"))
tam_corpus = pickle.load(open("/home/jones/Documentos/resultados/hu0_n200/tam_corpus.txt", "rb"))
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
        mapa[len(root_sentences_docs) - 1 ] = count#mapeando cada vetor raiz com seu documento respectivo
    count += 1

for i in range(0, len(ft), 2*tam_corpus):
    l = []
    for sentence in ft[i]:
        l.append(sentence[2][0])
    root_sentences_queries.append(l)#conjunto de todos os vetores raizes de todas as sentencas


docs_results = [0.0]*tam_queries*tam_corpus
docs_results = np.array(docs_results).reshape(tam_queries, tam_corpus)


nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(root_sentences_docs)

for i, sentences in enumerate(root_sentences_queries):

    distances, indices = nbrs.kneighbors(sentences)

    for j, queries_i in enumerate(indices):
        for l, k in enumerate(queries_i):
            docs_results[i][mapa[k]] += 1

indices = ranking(docs_results)

recall, selectivity = metricas(targets, indices)

plt.plot(selectivity, recall, marker='.')
plt.show()
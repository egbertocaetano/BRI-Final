from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

def ranking(pairwise_results):
    
    indices = []
    distance = []
    for j in range(len(pairwise_results)):

        indices.append([i[0] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1], reverse=True)])
        distance.append([i[1] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1], reverse=True)])
        
    return indices, distance

def metricas(targets, indices):
    
    indices = np.array(indices)


    recalls = np.zeros(targets.shape)
    selectivitys = np.zeros(indices.shape[1])
    targets_dense = targets.todense()
  	

    for i in range(len(targets_dense)):
        relevantes = 0

        rankingi = indices[2]

        #targeti = targets[i]
        targeti = np.transpose(targets_dense[2])
        total_relevantes = np.sum(targeti)

        for j in rankingi:
            if (targeti[j] == 1):
               # print("relevante", targeti[j])
                
                relevantes += 1

            recalls[i, j] += relevantes / total_relevantes
            selectivitys[j] += (j+1) / indices.shape[1]
        
    selectivitys /= len(targets_dense)
    
    recall = np.mean(recalls[:, [0, 50, 100, 213, 427, 641, 855]], axis=0)
    
    selectivitys = [0, 50, 100, 213, 427, 641, 855]
    return recall, selectivitys

'''
all_texts = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/at.txt", "rb"))
ft = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/ft.txt", "rb"))
targets = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/targets.txt", "rb"))
tam_queries = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/tam_queries.txt", "rb"))
tam_corpus = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/tam_corpus.txt", "rb"))
'''

all_texts = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/at_200.txt", "rb"))
ft = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/ft_200.txt", "rb"))
targets = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/targets_200.txt", "rb"))
tam_queries = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/tam_queries_200.txt", "rb"))
tam_corpus = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/tam_corpus_200.txt", "rb"))

sentences_docs = []
sentences_queries = []
count = 0
mapa = {}
root_sentences_docs = []
root_sentences_queries = []

print("Corpus:", tam_corpus)
print("All:", len(all_texts))
print("Queries:", tam_queries)
print("ft:", ft.shape)
print("target:", targets.shape)

queries_indices = np.nonzero(targets.sum(axis=1))[0]
targets = targets[queries_indices, :]
print(targets.shape)
for i in range(1, 2*tam_corpus, 2):
    l = []
    for sentence in ft[i]:
        root_sentences_docs.append(sentence[2][0])#conjunto de todos os vetores raizes de todas as sentencas
        mapa[len(root_sentences_docs) - 1 ] = count#mapeando cada vetor raiz com seu documento respectivo

    count += 1

for i in range(0, 2*tam_queries, 2):

    l = []
    for sentence in ft[i]:
        l.append(sentence[2][0])
    root_sentences_queries.append(l)#conjunto de todos os vetores raizes de todas as sentencas


docs_results = [0.0]*tam_queries*tam_corpus
docs_results = np.array(docs_results).reshape(tam_queries, tam_corpus)

nbrs = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='euclidean').fit(root_sentences_docs)
k = 5
for i in queries_indices:
    
    sentences = root_sentences_queries[i]

    distances, indices = nbrs.kneighbors(sentences)

    a = set()
    for j, queries_i in enumerate(indices):
        for l, m in enumerate(queries_i):
            a.add(mapa[m])
            if len(a) > k:
                break
            docs_results[i][mapa[m]] += 1
            
        if len(a) > k:
            break

#targets_dense = targets.todense()
#pprint(docs_results[10])
#pprint(np.nonzero(docs_results[657]))

indices, distances = ranking(docs_results)
recall, selectivity = metricas(targets, indices)

plt.plot(selectivity, recall, marker='.')
plt.show()
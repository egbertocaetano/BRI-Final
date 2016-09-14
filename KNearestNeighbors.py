from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

def ranking(pairwise_results):
    
    indices = []
    distancias = []
    
    for j in range(len(pairwise_results)):

        indices.append([i[0] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1], reverse=True)])
        distancias.append([i[1] for i in sorted(enumerate(pairwise_results[j]), key=lambda x:x[1], reverse=True)])
    	
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
    selectivitys = np.zeros(indices.shape[1])
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
            
            retornados +=1
        	
            if targeti[rankingi[j]] == 1:
                relevantes += 1
                #separation[i] = (distancias[i][j] * 100. / distancias[i][0]) - hfm[i]

            #elif hfm[i] == 0:
                #hfm[i] =  distancias[i][j] * 100. / distancias[i][0]
            
            precision = relevantes/retornados
            recall = relevantes/total_relevantes
            selectivity = retornados/indices.shape[1]

            precisions[j] = precisions[j] + precision
            recalls[j] = recalls[j] + recall
            selectivitys[j] = selectivitys[j] + selectivity
            f1[j] = f1[j] + f1_score(precision, recall)


    selectivitys /= len(targets)
    precisions /= len(targets)
    recalls /= len(targets)
    f1 /= len(targets)
    return precisions, recalls, f1, hfm, separation, selectivitys

'''
all_texts = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/at.txt", "rb"))
ft = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/ft.txt", "rb"))
targets = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/targets.txt", "rb"))
tam_queries = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/tam_queries.txt", "rb"))
tam_corpus = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/tam_corpus.txt", "rb"))
'''

all_texts = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/at.txt", "rb"))
ft = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n200/ft.txt", "rb"))
targets = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/targets.txt", "rb"))
tam_queries = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/tam_queries.txt", "rb"))
tam_corpus = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/tam_corpus.txt", "rb"))
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


docs_results = [0.0]*tam_queries*tam_corpus
docs_results = np.array(docs_results).reshape(tam_queries, tam_corpus)


nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(root_sentences_docs)

for i, sentences in enumerate(root_sentences_queries):

	
	distances, indices = nbrs.kneighbors(sentences)
	suspects = []
	sources = []
	for queries_i in indices:
		#suspects.append(mapa[ posicao[0] ])
		for k in queries_i:
			docs_results[i][mapa[k]] += 1
	'''
	for doc in set(suspects):
		#docs_results = np.array(len(root_sentences_docs[doc]), len(sentences))
		t = []
		for key, value in mapa.items():
			if value == doc:
				t.append(root_sentences_docs[key])
		s = 0
		for iter_doc, k in enumerate(t):
			
			for iter_querie, l in enumerate(sentences):
				#print(pairwise_distances(k.reshape(1,-1), l.reshape(1,-1), 'euclidean'))
				d = pairwise_distances(k.reshape(1,-1), l.reshape(1,-1), 'euclidean')
				if d > 1.2:
					s += d


		docs_results[i][doc] = s
	'''
#print(docs_results)
indices, distances = ranking(docs_results)
precision, recall, f1, hfm, separation, selectivity = metricas(targets, indices, distances)
print(recall)
print(selectivity)

plt.plot(recall, selectivity, marker='.')
plt.show()
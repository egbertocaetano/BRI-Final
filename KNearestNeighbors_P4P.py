from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from time import time

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
    targets_dense = targets.todense()
  	
    for i in range(len(targets_dense)):
        relevantes = 0

        rankingi = indices[i]

        #targeti = targets[i]
        targeti = np.transpose(targets_dense[i])
        total_relevantes = np.sum(targeti)

        for j in range(len(rankingi)):
            if (targeti[rankingi[j]] == 1):
               # print("relevante", targeti[j])
                
                relevantes += 1

            recalls[i, j] = relevantes / total_relevantes
        
    #print(np.transpose(np.nonzero(1-recalls)))
    #print(recalls[0,: 15])
    #print(recalls[9,: 15])
    #print(recalls)
    recall = np.mean(recalls[:, [10, 50, 100, 213, 427, 641, 855]], axis=0)
    recall_std = np.std(recalls[:, [10, 50, 100, 213, 427, 641, 855]], axis=0)
    selectivitys = [10, 50, 100, 213, 427, 641, 855]
    #print(recall)
    #print(recall_std)
    return recall, recall_std, selectivitys

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
print("\n-----------------------------------------------------------------------------------")

queries_indices = np.nonzero(targets.sum(axis=1))[0]
targets = targets[queries_indices, :]
#print(targets.shape)
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


docs_results = [0.0]*tam_corpus*tam_queries
docs_results = np.array(docs_results).reshape(tam_queries, tam_corpus)
#print(docs_results.shape)
k = 50
'''
brute = NearestNeighbors(n_neighbors=50, algorithm='brute', metric='euclidean').fit(root_sentences_docs)
y_brute = []
for i in queries_indices:
    
    sentences = root_sentences_queries[i]
    distances, indices = brute.kneighbors(sentences)
    a = set()
    for j, queries_i in enumerate(indices):
        for l, m in enumerate(queries_i):
            a.add(mapa[m])
            if len(a) > k:
                break
            docs_results[i][mapa[m]] += 1
            y_brute.append(distances[j, l])
            
        if len(a) > k:
            break
kdTree = NearestNeighbors(n_neighbors=50, algorithm='kd_tree', metric='euclidean').fit(root_sentences_docs)
y_kd = []
for i in queries_indices:
    
    sentences = root_sentences_queries[i]
    distances, indices = kdTree.kneighbors(sentences)
    a = set()
    for j, queries_i in enumerate(indices):
        for l, m in enumerate(queries_i):
            a.add(mapa[m])
            if len(a) > k:
                break
            docs_results[i][mapa[m]] += 1
            y_kd.append(distances[j, l])
            
        if len(a) > k:
            break
'''
fig, ax = plt.subplots()
for algorithm_name,alg_decorator in [("kd_tree","-*"),("ball_tree","-^"),("brute","-o")][::-1]:

    alg_time = time()


    ballTree = NearestNeighbors(n_neighbors=1000, algorithm=algorithm_name, metric='euclidean').fit(root_sentences_docs)

    alg_time = time() - alg_time
    
    y_ball = []

    ret_time = []
    for i in queries_indices:
        
        sentences = root_sentences_queries[i]


        rt_timei =time()
        distances, indices = ballTree.kneighbors(sentences)
        ret_time.append(time() - rt_timei)

        a = set()
        for j, queries_i in enumerate(indices):
            for l, m in enumerate(queries_i):
                a.add(mapa[m])
                if len(a) > k:
                    break
                docs_results[i][mapa[m]] += 1
                y_ball.append(distances[j, l])
                
            if len(a) > k:
                break

    docs_results_1 = docs_results[queries_indices, :]
    
    


    indices, distances = ranking(docs_results_1)
    
    del docs_results_1
    print("Algorithm - ", algorithm_name ," : construction time in %4.4f seconds | %4.4f retrievel time seconds"%(alg_time,np.sum(np.array(ret_time))) )
    recall, recall_std, selectivity = metricas(targets, indices)

    print("Recall : ", recall)
    print("Variância : ", recall_std)
    print("Selectivity : ", selectivity)
    print("\n-----------------------------------------------------------------------------------")

    title_graphic = algorithm_name + "_n=100_unfold=True"


    ax.errorbar(selectivity, recall, yerr=recall_std, fmt=alg_decorator)
    ax.set_title(algorithm_name)


    path = "/home/forrest/workspace/BRI/Final Work/final/image/short_"+title_graphic
    plt.savefig(path)
    
    #plt.plot(selectivity, recall, marker='o')

#plt.show()

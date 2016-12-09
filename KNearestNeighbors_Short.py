from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import numpy as np
import pickle
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from time import time

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


def metricas(targets, indices):
	indices = np.array(indices)

	recalls = np.zeros(targets.shape)
	#targets_dense = targets.todense()

	for i in range(len(targets)):
		relevantes = 0

		rankingi = indices[i]

		#targeti = targets[i]
		targeti = targets[i]
		total_relevantes = np.sum(targeti)

		for j in range(len(rankingi)):
			if (targeti[rankingi[j]] == 1):
			# print("relevante", targeti[j])

				relevantes += 1

			recalls[i, j] = relevantes / total_relevantes
		

	#print(np.transpose(np.nonzero(1-recalls)))
	#print(recalls[0,: 15])
	#print(recalls[9,: 15])
	recall = np.mean(recalls, axis=0)
	recall_std = np.std(recalls, axis=0)
	selectivitys = [0, 1, 2, 3, 4]
	#print(recall)
	#print(recall_std)
	return recall, recall_std, selectivitys


# all_texts = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/at.txt", "rb"))
# ft = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/ft.txt", "rb"))
# targets = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/targets.txt", "rb"))
# tam_queries = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/tam_queries.txt", "rb"))
# tam_corpus = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/tam_corpus.txt", "rb"))

# all_texts = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/at_200.txt", "rb"))
# ft = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/ft_200.txt", "rb"))
# targets = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/targets_200.txt", "rb"))
# tam_queries = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/tam_queries_200.txt", "rb"))
# tam_corpus = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/pan_2011/tam_corpus_200.txt", "rb"))

# #print(ft[0][0][0])
# all_texts = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/at.txt", "rb"))
# ft = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/ft.txt", "rb"))
# targets = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/targets.txt", "rb"))
# tam_queries = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/tam_queries.txt", "rb"))
# tam_corpus = pickle.load(open("/home/forrest/Documentos/resultados/hu0_n100/tam_corpus.txt", "rb"))



all_texts = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/result_100/T2-UNFOLD/at.txt", "rb"))
ft = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/result_100/T2-UNFOLD/ft.txt", "rb"))
targets = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/result_100/T2-UNFOLD/targets.txt", "rb"))
tam_queries = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/result_100/T2-UNFOLD/tam_queries.txt", "rb"))
tam_corpus = pickle.load(open("/home/forrest/workspace/BRI/Final Work/final/resultados/result_100/T2-UNFOLD/tam_corpus.txt", "rb"))


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


#nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(root_sentences_docs)
#btree = BallTree(root_sentences_docs, leaf_size=2)
k = 10

fig, ax = plt.subplots()
for algorithm_name,alg_decorator in [("kd_tree","-*"),("ball_tree","-^"),("brute","-o")][::-1]:

    alg_time = time()


    ballTree = NearestNeighbors(n_neighbors=10, algorithm=algorithm_name, metric='euclidean').fit(root_sentences_docs)

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
    ax.set_title(title_graphic)


    path = "/home/forrest/workspace/BRI/Final Work/final/image/short_"+title_graphic
    plt.savefig(path)
    #plt.plot(selectivity, recall, marker='o')

#plt.show()

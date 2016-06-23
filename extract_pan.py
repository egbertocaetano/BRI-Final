# create_extrinsic_partition.py
# by Marcus Huderle
# Creates a random partition of the extrinsic corpus that ensures the source files
# of plagiarised documents are also included in the partition.
#
# There are 14,428 total suspicious documents in the test pan-plagiarism-corpus-2009 directory
'''
import os
import random
import xml.etree.ElementTree as ET
import sys

def main():
    
    suspects_base_path = "/home/jones/Documentos/BRI/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents/"
    suspects_dirs = ["part1/", "part2/", "part3/", "part4/", "part5/", "part6/", "part7/", "part8/"]
    sources_base_path = "/home/jones/Documentos/BRI/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents/"
    sources_dirs = ["part1/", "part2/", "part3/", "part4/", "part5/", "part6/", "part7/", "part8/"]

     # Without extensions
    all_base_files = []
    all_files = [] # list of tuples where tuple[0] is the absolute path of the text document and tuple[1] is the absolute path of the xml file

    # Put all the suspect files in a list
    for d in suspects_dirs:
        p = os.path.join(suspects_base_path, d)
        for f in os.listdir(p):
            all_base_files.append(os.path.splitext(f)[0])

            if f[-4:] == ".txt":
                all_files.append((p+f, (p+f)[:-4]+".xml"))
    
    # Make sure all of these files actually exist
    worked = True
    for suspect in all_files:
        if not os.path.exists(suspect[0]):
            worked = False
            print(".txt file does not exist:", suspect[0])
        if not os.path.exists(suspect[1]):
            worked = False
            print(".xml file does not exist:", suspect[1])
    assert(worked)

    # shuffle and take files from the front of the list
    print('Shuffling ', len(all_files), 'suspect files...')
    random.shuffle(all_files)

    cutoff = int(len(all_files)*training_percent)
    print('Splitting suspect files into', cutoff, 'training files and', len(all_files)-cutoff, 'testing files...')
    training_suspect_partition = all_files[:cutoff]
    testing_suspect_partition = all_files[cutoff:]

    print('Writing partitions to disk...')
    suspect_training_file = open("extrinsic_training_suspect_files.txt", 'w')
    for suspect in training_suspect_partition:
        rel_path_start = suspect[0].index('/part')
        suspect_training_file.write(suspect[0][rel_path_start:-4] + '\n')
    suspect_training_file.close()

    suspect_testing_file = open("extrinsic_testing_suspect_files.txt", 'w')
    for suspect in testing_suspect_partition:
        rel_path_start = suspect[0].index('/part')
        suspect_testing_file.write(suspect[0][rel_path_start:-4] + '\n')
    suspect_testing_file.close()

    print('Determining source documents for training partition...')
    training_sources = {}
    num_files = 0
    for filenames in training_suspect_partition:
        tree = ET.parse(filenames[1])
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism" and feature.get("source_reference") and feature.get("source_reference")[:-4] not in training_sources:
                # figure out which partX the doc is in...so annoying...
                for p in sources_dirs:
                    if os.path.exists(sources_base_path + p + feature.get("source_reference")):
                        training_sources["/" + p + feature.get("source_reference")[:-4]] = 1
        num_files += 1
        if num_files%100 == 0:
            print(num_files,)
            sys.stdout.flush()
    print()
    print(len(training_sources.keys()), 'sources for the training partition were found...')

    print('Determining source documents for testing partition...')
    testing_sources = {}
    num_files = 0
    for filenames in testing_suspect_partition:
        tree = ET.parse(filenames[1])
        for feature in tree.iter("feature"):
            if feature.get("name") == "artificial-plagiarism" and feature.get("source_reference") and feature.get("source_reference")[:-4] not in training_sources:
                # figure out which partX the doc is in...so annoying...
                for p in sources_dirs:
                    if os.path.exists(sources_base_path + p + feature.get("source_reference")):
                        testing_sources["/" + p + feature.get("source_reference")[:-4]] = 1
        num_files += 1
        if num_files%100 == 0:
            print(num_files,)
            sys.stdout.flush()
    print()
    print(len(testing_sources.keys()), 'sources for the testing partition were found...')

    print('Writing source documents to disk...')
    source_training_file = file("extrinsic_training_source_files.txt", 'w')
    for filename in training_sources.keys():
        source_training_file.write(filename + '\n')
    source_training_file.close()

    source_testing_file = file("extrinsic_testing_source_files.txt", 'w')
    for filename in testing_sources.keys():
        source_testing_file.write(filename + '\n')
    source_testing_file.close()
    

if __name__ == '__main__':
    main()

'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import xml.etree.ElementTree as ET
from nltk import word_tokenize
import re
import codecs
import reader
import numpy as np

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


def metricas(targets, id_queries, id_sources, indices, distancias):
    
    precisions = np.zeros(len(id_sources))
    recalls = np.zeros(len(id_sources))
    f1 = np.zeros(len(id_sources))
    hfm = np.zeros(len(id_queries))
    separation = np.zeros(len(id_queries))

    for i in range(len(id_queries)):
        
        relevantes = 0
        retornados = 0
        
        rankingi = indices[i]
        targeti = targets[i]
        
        total_relevantes = len(rankingi)
        
        for j in range(len(rankingi)):
            
            retornados = j+1
        
            if id_sources[ rankingi[j] ] in targeti:
                relevantes += 1
                separation[i] = (distancias[i][ rankingi[j] ] * 100. / distancias[i][ rankingi[0] ]) - hfm[i]

            elif hfm[i] == 0:
                hfm[i] =  distancias[i][ rankingi[j] ] * 100. / distancias[i][ rankingi[0] ]
                
            precisions[j] = precisions[j] + relevantes/retornados
            recalls[j] = recalls[j] + relevantes/total_relevantes
            f1[j] = f1[j] + f1_score(precisions[j], recalls[j])
        
    precisions /= len(id_queries)
    recalls /= len(id_queries)  
    f1 /= len(id_queries)

   
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


def mount_vocab(l):
    queries = []
    corpus = []
    id_queries = []
    id_sources = []
    targets = []
    vocab = []
    for path in l[1:]:
        if(path[-4:] == ".txt"):
           
            file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
            text = file.read()
            if path.find("source") == -1:
                id_queries.append(path.split("/")[-1])
                vocab += remove_stopwords(tokenize_text(text))
                queries.append( text.lower() )

                tree = ET.parse(path[:-4]+".xml")
                target_aux = list()
                for feature in tree.iter("feature"):
                    if(feature.get("name") == "plagiarism") and feature.get("source_reference") not in target_aux:
                        target_aux.append(feature.get("source_reference"))
                targets.append(target_aux)

            else:
                id_sources.append(path.split("/")[-1])
                vocab += remove_stopwords(tokenize_text(text))
                corpus.append( text.lower() )
                            
    vocab = list(set(vocab))
    
    return vocab, queries, corpus, id_queries, id_sources, targets

    
r = reader.Reader("/home/jones/external-detection-corpus/")
l = r.get_paths()

print("Montando vocabulario")
vocab, queries, corpus, id_queries, id_sources, targets = mount_vocab(l)

cv = CountVectorizer(vocabulary = vocab)

print("Contruindo os vetores de frequencias das palavras das queries")
matrix_queries = cv.fit_transform(queries)

print("Contruindo os vetores de frequencias das palavras do corpus")
matrix_corpus = cv.fit_transform(corpus)

print("Rankeando as similaridades")
indices, distancias = ranking(matrix_queries, matrix_corpus)

print("Resultados")
precisions, recalls, f1, hfm, separation = metricas(targets, id_queries, id_sources, indices, distancias)

pickle.dump(precisions, open("/home/jones/resultados/precisions_bow.txt", "wb"))
pickle.dump(recalls, open("/home/jones/resultados/recalls_bow.txt", "wb"))
pickle.dump(f1, open("/home/jones/resultados/f1_bow.txt", "wb"))
pickle.dump(hfm, open("/home/jones/resultados/hfm_bow.txt", "wb"))
pickle.dump(separation, open("/home/jones/resultados/separation_bow.txt", "wb"))
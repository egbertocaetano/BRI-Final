from nltk.parse import stanford as st
from nltk.tree import Tree
from pprint import pprint
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing.data import normalize
from numpy import linalg, zeros, nonzero, ones, matrix, argmax, argsort
from scipy.sparse.csr import csr_matrix
import math
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from time import time
import os.path
import codecs
import pickle
from sklearn.externals.joblib.parallel import delayed, Parallel
from duartefellipe.datasets.extractors.short_plagiarised_answers_extractor import extract_short_plagiarized_answers_to_ranking
from duartefellipe.deep_learning.activation_functions import Tanh
from duartefellipe.deep_learning.word_embedding_transformers import ACL2010WordEmbedding
from duartefellipe.deep_learning import hadamard_product
from nltk.tokenize import sent_tokenize
from sklearn.linear_model.logistic import LogisticRegression
from duartefellipe.datasets.extractors.meter_extractor import extract_meter_to_corpus_ranking_task,\
    extract_meter_news_relations
from duartefellipe.datasets.extractors.pan_plagiarism_corpus_2011_extractor import load_pan_2011
from scipy.sparse.lil import lil_matrix
from duartefellipe.datasets.extractors.P4P_extractor import extract_p4p_to_ranking_task
import re
from duartefellipe.config import datasets_extractors, STANFORD_PARSER
from nltk.corpus import wordnet as wn, wordnet_ic


encoding_re = '[' + re.escape(''.join([chr(0xa0+i) for i in range(0, 5*16+15) ])) + ']'

def encoding_sanitization(input_text):
#     input_text = codecs.encode(input_text, encoding='cp1252', errors='ignore')
#     input_text = codecs.encode(input_text, encoding='utf-8')
#     input_text = codecs.decode(input_text, encoding='utf-8', errors='ignore')

#     return re.sub(encoding_re, '', input_text.replace(u'\xa0', u' ').replace("\n\n","\n").encode(' ISO_8859-1'))        
    return input_text.replace("\n\n","\n")

def load_MSRParaphraseCorpus():
    '''
        loading the sample file with 5801 paraphrase pairs. (train=4076 + test=1725)
    '''
    #path = "/home/fellipe/Documents/Datasets/MSRParaphraseCorpus/"
#     path = "C:/Users/Fellipe/Documents/Datasets/MSRParaphraseCorpus" 
    path = datasets_extractors['DATASETS_PATH']['MSRParaphrase']
       
    corpus_tuple = []
    corpus_size = []
    
    for file_name in ['msr_paraphrase_train.txt','msr_paraphrase_test.txt']:
        with codecs.open(os.path.join(path,file_name), 'r', encoding='utf-8') as msrp_file:
#         with open(os.path.join(path,file_name), 'r') as msrp_file:    
            for linei in msrp_file.readlines():

                linei_features = linei.split('\t')
                if linei_features[0] != '\ufeffQuality':
                    corpus_tuple.append([linei_features[0],encoding_sanitization(linei_features[3]),encoding_sanitization(linei_features[4])])
            corpus_size.append(len(corpus_tuple))

    corpus_tuple = np.array(corpus_tuple)
    corpus_tuple = corpus_tuple.reshape((corpus_size[1],3))
    
    return (corpus_tuple[:,1:],corpus_tuple[:,0]=='1',corpus_size[0],corpus_size[1]-corpus_size[0])
    

def load_pan_plagiarism_corpus_2011():
    #path = "/home/jones/Documentos/Datasets/pan-plagerism-corpus-2011/"
    queries, corpus_index, target, labels = load_pan_2011()
    
    dataset_documents = []
    dataset_target = []
    

    for i in range(len(queries)):
        for j in range(len(corpus_index)):
            dataset_documents.append((
                                      encoding_sanitization(queries[i]['content']),
                                      encoding_sanitization(corpus_index[j]['content'])
                                      ))
            dataset_target.append(target[i][j])

    del queries, corpus_index, target
    
    return (np.array(dataset_documents), np.array(dataset_target)==1, 0, 0)


def load_cpc11Corpus():
    '''
        loading the sample file with 7859 paraphrase pairs.
    '''
    #path = "/home/fellipe/Documents/Datasets/corpus-webis-cpc-11/"
#     path = "C:/Users/Fellipe/Documents/Datasets/corpus-webis-cpc-11" 
    path = datasets_extractors['DATASETS_PATH']['cpc-11'] 

    corpus_tuple = []
    corpus_target = []
    
    paraphrases_count = int(len(os.listdir(path))/3)
    
    for fi in range(1,paraphrases_count+1):
#         print("[%d/%d]"%(fi, paraphrases_count))
        with codecs.open(os.path.join(path,"%d-original.txt"%(fi)), 'r', encoding='utf-8', errors='ignore') as f:
            original_contenti = codecs.decode(codecs.encode(encoding_sanitization(f.read().replace('\n','')), encoding='cp1252', errors='ignore'), encoding='utf-8', errors='ignore')
#             original_contenti = ""
#             for linei in f.readlines():
#                 if linei != '\n':
#                     original_contenti += linei 
        with codecs.open(os.path.join(path,"%d-paraphrase.txt"%(fi)), 'r', encoding='utf-8', errors='ignore') as f:
            para_contenti = codecs.decode(codecs.encode(encoding_sanitization(f.read().replace('\n','')), encoding='cp1252', errors='ignore'), encoding='utf-8', errors='ignore')
#             para_contenti = ""
#             for linei in f.readlines():
#                 if linei != '\n':
#                     para_contenti += linei 
        with codecs.open(os.path.join(path,"%d-metadata.txt"%(fi)), 'r',encoding='utf-8', errors='ignore') as f:
            meta_contenti = f.read()[-4:-1]

            if meta_contenti == 'Yes' : 
                meta_contenti = 1 
            else:
                meta_contenti = 0
            
        corpus_tuple.append([original_contenti, para_contenti])
        corpus_target.append( meta_contenti)
#         print("====[",original_contenti,"]====")
#         print("====[",para_contenti,"]====")
#         exit()
            

    corpus_tuple = np.array(corpus_tuple)
    corpus_target = np.array(corpus_target)
#     print(corpus_tuple.shape, corpus_target.shape)
    
    return (corpus_tuple, corpus_target, 0, 0)
        
def load_short_plagiarized_answers_as_pairs():
    '''
        5 documents X 57 queries (plagiarism suspects) = 285 comparison pairs 
        
        285 paraphrase pairs to eval.
    '''
    queries, corpus_index, target, labels = extract_short_plagiarized_answers_to_ranking()
    
    dataset_documents = []
    dataset_target = []
    
    replace_fun = lambda x: x.replace(" '","").replace("' ","").replace(" `","").replace("` ","").replace(" \"","").replace("\" ","").replace("´´","").replace("``","").replace('\n','').replace('\\','')

    for i in range(len(queries)):
        for j in range(len(corpus_index)):
            dataset_documents.append((
                                      replace_fun(encoding_sanitization(queries[i]['content'])),
                                      replace_fun(encoding_sanitization(corpus_index[j]['content']))
                                      ))
            dataset_target.append(target[i][j])
    tam_queries = len(queries)
    tam_corpus = len(corpus_index)
    del queries, corpus_index, labels
    
    return (np.array(dataset_documents), np.array(dataset_target)==1, 0, 0, tam_queries, tam_corpus, target)
    
def load_meter_as_pairs(leave_out = None):
    '''
         123820 court pairs + 8100 showbiz pairs = 131920 pairs to eval
    '''
    
    news_relations = extract_meter_news_relations(leave_out)
    
    dataset_documents = np.empty((len(news_relations),2),dtype=np.ndarray)
    dataset_target = np.empty((len(news_relations)),dtype=np.int)
    
    for i in range(len(news_relations)):
            dataset_documents[i,0] = encoding_sanitization(news_relations[i][0])
            dataset_documents[i,1] = encoding_sanitization(news_relations[i][1])
            dataset_target[i] = news_relations[i][2]
#             print("**[%s]"%dataset_documents[k,0]) 

    del news_relations
        
    return (dataset_documents, dataset_target, 0, 0)   

def load_meter_showbiz_as_pairs():
    '''
         8100 showbiz pairs to eval
    '''

    return load_meter_as_pairs(leave_out = 'courts')

def load_meter_courts_as_pairs():
    '''
         123820 court pairs to eval
    '''

    return load_meter_as_pairs(leave_out = 'showbiz')


def load_p4p_pairs():
    '''
        856 documents and queries pairs (plagiarism suspects) = 1712 comparison pairs 
    '''
    
    """
        must compare query[i] with corpus_index[j]
    """
    queries, corpus_index, target, labels = extract_p4p_to_ranking_task()
    
    dataset_documents = np.empty((len(queries),2),dtype=np.ndarray)
    dataset_target = np.empty((len(queries)),dtype=np.bool)

    for i in range(dataset_documents.shape[0]):
            dataset_documents[i,0] = encoding_sanitization(queries[i])
            dataset_documents[i,1] = encoding_sanitization(corpus_index[i])
            dataset_target[i] = target[i,i] == 1

    del queries, corpus_index, target, labels
        
    return (dataset_documents, dataset_target, 0, 0)    

def wordnet_syntatic_semantic_vectors(sentence1_word_list,sentence2_word_list,ic = wordnet_ic.ic('ic-brown.dat')):
    '''
        Sentence Similarity Based on Semantic Nets and Corpus Statistics approach
    
        return (semantic-vectors, syntactic-vectors, word-set)
    '''    
    
    word_set = set()
    for sent1_wordi in sentence1_word_list:
        sent1_wss = wn.synsets(sent1_wordi)
        word_set = word_set.union(set(sent1_wss))
        for sent2_wordi in sentence2_word_list:
            sent2_wss = wn.synsets(sent2_wordi)
            word_set = word_set.union(set(sent2_wss))
            
    word_set = list(word_set)
    semantic_vector = zeros((len(word_set),2))
    syntactic_vector = zeros((len(word_set),2))
    
    for wpos,w_ss in enumerate(word_set):
        for sent_id, sent_lw in enumerate([sentence1_word_list,sentence2_word_list]):
            
            for sent_wi_pos,sent_wi in enumerate(sent_lw):
                sent_wiss = wn.synsets(sent_wi)
                if w_ss in sent_wiss:
                    semantic_vector[wpos,sent_id] = 1
                    syntactic_vector[wpos,sent_id] = sent_wi_pos + 1
                    break
                else:
                    words_similarities = []
                    words_similarities_pos = []
                    for sent_wissj in sent_wiss:
                        try:
                            words_similarities.append(w_ss.lin_similarity(sent_wissj,ic))
                            words_similarities_pos.append(sent_wi_pos)
                        except:
                            pass
                    if len(words_similarities) > 0:
                        max_pos = argmax(words_similarities)
                        semantic_vector[wpos,sent_id] = words_similarities[max_pos]
                        syntactic_vector[wpos,sent_id] = words_similarities_pos[max_pos] + 1
                    
#     print(lil_matrix(semantic_vector).nnz,'x',semantic_vector.shape[0]*semantic_vector.shape[1])
#     print(lil_matrix(syntactic_vector).nnz,'x',syntactic_vector.shape[0]*syntactic_vector.shape[1])
    return semantic_vector, syntactic_vector, word_set
    
    
def _tree_to_matrix(tree_to_parse):
    
    tree_to_parse.collapse_unary(collapseRoot=True,collapsePOS=True)
    tree_to_parse.chomsky_normal_form()
    nodes = [tree_to_parse]
    
    nonterminals_vocabulary = {}

    tree_vocabulary = {}

    '''
        generating unique ID's foreach node gramatical class (NP, NN,JJ) instance
    '''
    while(len(nodes) > 0):
        nodei = nodes.pop(0)
        
        if nodei.label() not in nonterminals_vocabulary.keys():
            nonterminals_vocabulary[nodei.label()] = []
            
        if nodei not in nonterminals_vocabulary[nodei.label()]:
            nonterminals_vocabulary[nodei.label()].append(nodei)

        nodei_classid = "%s_%d"%(nodei.label(),nonterminals_vocabulary[nodei.label()].index(nodei))
        nodei.set_label(nodei_classid)
        if nodei_classid not in tree_vocabulary.keys():
            tree_vocabulary[nodei_classid] = len(tree_vocabulary)
            
        for sonj in nodei:
            if isinstance(sonj, Tree):
                '''
                    transforming productions like NP-> NN VT and NN ->"book" to NP-> "book" VT  
                '''
                if len(sonj) > 1:
                    nodes.append(sonj)
                else:
                    if sonj[0] not in tree_vocabulary.keys():
                        tree_vocabulary[sonj[0]] = len(tree_vocabulary)
            else:
                '''
                    sonj is a string
                '''
                if sonj not in tree_vocabulary.keys():
                    tree_vocabulary[sonj] = len(tree_vocabulary)
        
    del nonterminals_vocabulary
    
    nodes = [tree_to_parse]
    productions_matrix = np.zeros((len(2*tree_to_parse.leaves())-1,2),dtype=np.int8) - np.ones((len(2*tree_to_parse.leaves())-1,2),dtype=np.int8)
    
#     pprint(tree_to_parse)
#     pprint(tree_vocabulary)
    
    while(len(nodes) > 0):
        nodei = nodes.pop(0)
        i = tree_vocabulary[nodei.label()]
        for j in range(len(nodei)):
            if isinstance(nodei[j], Tree):
                if len(nodei[j]) > 1:
                    nodes.append(nodei[j])
#                     print("(%d, %s) -> (%d,%s)"%(i,nodei.label(),tree_vocabulary[nodei[j].label()],nodei[j].label()))
                    productions_matrix[i,j] = tree_vocabulary[nodei[j].label()]
                else:
                    productions_matrix[i,j] = tree_vocabulary[nodei[j,0]]
            else:
#                 print("(%d, %s) -> (%d,%s)"%(i,nodei.label(),tree_vocabulary[nodei[j]],nodei[j]))
                productions_matrix[i,j] = tree_vocabulary[nodei[j]]

    nonterminal_ids = nonzero(productions_matrix[:,0] > 0)[0]

    return productions_matrix, tree_vocabulary,nonterminal_ids,tree_to_parse.leaves()     

class SentenceTreeTokenizer(BaseEstimator, VectorizerMixin):
    
    def __init__(self,parser_model_path="/home/jones/Documentos/Copy of retrieval_pipeline/libsenglishPCFG.ser.gz", parser_path_to_jar='/home/jones/Documentos/Copy of retrieval_pipeline/libsstanford-parser.jar', parser_path_to_models_jar='/home/jones/Documentos/Copy of retrieval_pipeline/libsstanford-parser-3.5.2-models.jar', log_steps = False):
        self.parser_model_path, self.parser_path_to_jar, self.parser_path_to_models_jar, self.log_steps = parser_model_path, parser_path_to_jar, parser_path_to_models_jar, log_steps 
        self.parser = st.StanfordParser(
            model_path = self.parser_model_path, 
            path_to_jar = self.parser_path_to_jar, 
            path_to_models_jar = self.parser_path_to_models_jar,
            java_options=STANFORD_PARSER['java_options'],
            encoding=STANFORD_PARSER['encoding']
        ) 
        
    def fit(self, raw_documents, y=None):
        pass

    def fit_transform(self, raw_documents, y=None):    
        return self.transform(raw_documents)
    
    def transform(self, raw_documents):
        '''
            each document from raw_documents can have sentences.
            
            return for each document a list of  tuples(p,vocabulary) where :
            
            * p is a matrix that maps each node to its offsprings (y1 -> x1,x2)
            * vocabulary maps the node index to the node tag/ word (0 maps to ROOT label). 
        '''
        parse_results = []
        for i in range(len(raw_documents)):
            raw_documents_i = raw_documents[i]
            
            parse_results_i = []
            for sentencej in list(sent_tokenize(raw_documents_i)):
                parse_results_i.append(self.parser.raw_parse_sents([sentencej]))                                      
            
            parse_results.append([])
            
            for treei_list_of_list in parse_results_i:
                for treei_list in list(iter(treei_list_of_list)):
                    for treei in list(iter(treei_list)):
                        parse_results[-1].append(_tree_to_matrix(treei))
                        del treei
#                         treei.draw()
            del parse_results_i
            
            parse_results[-1] = np.array(parse_results[-1])
            
            if self.log_steps:
                print("%d-th document(total: %d) with %d sentences"%(i+1,len(raw_documents),len(parse_results[i])))
                
        return np.array(parse_results)    
      
class SentenceDeepRecursiveAutoEncoder(BaseEstimator, VectorizerMixin):
    
    def __init__(self,word_embeddings_path,unfold = False, n = 3, hidden_units = 200,eta = 0.01,reg_rate = 0.00001, epochs = 10, error_threshold = 0.001, activation_function = Tanh(), debug = False, log_steps = False):
        self.word_embeddings_path,self.unfold,self.n,self.hidden_units, self.eta,self.reg_rate, self.epochs,self.error_threshold, self.debug, self.log_steps = (word_embeddings_path, unfold, n,hidden_units, eta, reg_rate, epochs, error_threshold, debug, log_steps)
        self.activation_function = activation_function
        
    def weight_init(self,axis1,axis2):
        """
            http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.randn.html
        """
        dist_mean, dist_std = 0, 1.0/math.sqrt(axis1)
        return dist_std*np.random.randn(axis1,axis2) + dist_mean

    def _boostrap(self,sentence_tree_nodes):
        if self.hidden_units == None:
            Wh,We,Wd = None,self.weight_init(self.n,2*self.n),self.weight_init(2*self.n,self.n)
    #         be,bd = self.weight_init(self.n), self.weight_init(2*self.n)
            bh,be,bd = None,np.zeros(self.n), np.zeros(2*self.n)
            Ch,Ce,Cd = None,self.weight_init(len(sentence_tree_nodes),self.n),self.weight_init(len(sentence_tree_nodes),2*self.n)
        else:                
            Wh, We, Wd = self.weight_init(self.hidden_units,2*self.n),self.weight_init(self.n,self.hidden_units), self.weight_init(2*self.n,self.n)
    #         be,bd = self.weight_init(self.n), self.weight_init(2*self.n)
            bh, be, bd = np.zeros(self.hidden_units), np.zeros(self.n), np.zeros(2*self.n)
            Ch, Ce, Cd = self.weight_init(len(sentence_tree_nodes),self.hidden_units),self.weight_init(len(sentence_tree_nodes),self.n),self.weight_init(len(sentence_tree_nodes),2*self.n) 
        
        return Wh, We, Wd, bh, be, bd, Ch, Ce, Cd

    def _encode(self, biases, weights, z, activation_function = None):
        """ 
            neuron activation function
            
            return y = neuron activation results vector for z inputs vectors   
        """
#         print(biases.shape,"+",weights.eshape,"*", z.shap,end="")
        y = np.dot(weights,z) + biases
        
        if activation_function != None:
            y = activation_function(y)
#         print(' =',y.shape)
        return y
        
    def _decode(self, biases, weights, z, activation_function = None):
        """ 
            neuron activation function
            
            return y = neuron activation results vector for z inputs vectors   
        """
#             print(biases.shape,"+",weights.shape,"*", z.shape,end="")
        y = np.dot(weights,z) + biases
        
        if activation_function != None:
            y = activation_function(y)
#         print(' =',y.shape)
        return y.reshape(2,self.n)    
    
        
    def fit(self, sentence_tuples, y=None):
        pass

    def fit_transform(self, sentence_tuples, y=None):    
        return self.transform(sentence_tuples)
    
        
    def transform(self, sentence_tuples):
        
        sti_productions,sti_vocab,sti_nonterminals, sti_leaves = sentence_tuples
        we_function = ACL2010WordEmbedding(self.n,self.word_embeddings_path) 

        pooling_order, Ce, reg_rate_min_error  = None, None, None
        
#         for reg_ratei in [self.reg_rate*10**(-i) for i in np.arange(0, 6)]:

        continue_training = True
        reg_ratei = self.reg_rate
        while continue_training:
            
            Wh, We, Wd, bh, be, bd, Ch, Ce, Cd = self._boostrap(sti_vocab)
                 
            pooling_order = []
                
            for sti_leavei,sti_leavei_vector in zip(sti_leaves,we_function(sti_leaves)):
                pooling_order.append(sti_vocab[sti_leavei])
                Ce[sti_vocab[sti_leavei],:] = sti_leavei_vector
    #           print(sti_leavei_vector)
                
    #       print(pooling_order,'+',sti_nonterminals)
            pooling_order.extend(sti_nonterminals)
    #         print(pooling_order)
            
            epochs_Ce = []    
            epochs_error_sum = []
            for epochi in range(0,self.epochs):
                if self.log_steps:
                    print("Training epoch %d/%d :[reg_rate=%1.2e]"%(epochi+1,self.epochs,reg_ratei))
    
                for i in sti_nonterminals[::-1]:
                    """
                        encoding structure vectors (forward fase)
                    """
                    c1c2 = np.hstack(Ce[sti_productions[i,:]])
                    if self.hidden_units == None:
                        Ce[i,:] = self._encode(be, We, c1c2, self.activation_function)
                    else:
                        Ch[i,:] = self._encode(bh, Wh, c1c2, self.activation_function)
                        Ce[i,:] = self._encode(be, We, Ch[i,:], self.activation_function)
        
                    Ce[i,:] = normalize(matrix(Ce[i,:]), axis=1, norm='l2')
        
                    """
                        decoding structure vectors (backward fase)
                    """
                    p = Ce[i,:]
                    Cd[i,:] = (self._decode(bd, Wd, p, self.activation_function)).flatten()
                
                """
                    backpropagation
                """ 
                reconstruction_error = [] # all nonterminals trees
                for i in sti_nonterminals:
                    c1c2 = np.hstack(Ce[sti_productions[i,:]])
                    p = Ce[i,:]
                    c1c2_error = (1.0/Cd.shape[0])*(Cd[i,:] - c1c2)
                        
                    if (self.unfold and sti_productions[i,:][1] in sti_nonterminals):
                        y_error = (1.0/Cd.shape[0])*(Cd[i,self.n:] - c1c2[self.n:])
                        c1c2_error -= np.hstack([zeros(self.n),y_error])
#                         print("\t unfolding %d!"%(i))                
                        
                    """
                        backpropagating reconstruction error sigma_p 
                    """
                    sigma_p = hadamard_product(c1c2_error,self.activation_function.derivative(np.dot(Wd, p)))
                    """
                        outer(u,v) == u(v.T)
                    """
                    Wd = Wd + self.eta*np.outer(sigma_p,p) + (reg_ratei/Cd.shape[0])*Wd    
                            
                    if self.hidden_units == None:
                        """
                            backpropagating p error to c1 and c2
                        """
                        sigma_we = hadamard_product(np.dot(Wd.T, sigma_p),self.activation_function.derivative(np.dot(We, c1c2)))
                        We = We + self.eta*np.outer(sigma_we,c1c2)
                    else:
                        
                        """
                            backpropagating p error (sigma_p) to activation layer (p)
                        """
                        h = Ch[i,:]
                        sigma_we = hadamard_product(np.dot(Wd.T, sigma_p),self.activation_function.derivative(np.dot(We, h)))
                        We = We + self.eta*np.outer(sigma_we,h)
        
                        """
                            backpropagating activation layer error (sigma_we) to hidden layer (h) 
                        """
                        Wh = Wh + self.eta*np.outer(hadamard_product(np.dot(We.T, sigma_we),self.activation_function.derivative(np.dot(Wh, c1c2))),c1c2)
                        
                    reconstruction_error.append(linalg.norm(sigma_p))
                
                epochs_error_sum.append(np.array(reconstruction_error).sum())
                print(epochs_error_sum)
                epochs_Ce.append(Ce)    
            
                if self.log_steps:
                    print('tree nodes reconstruction_error[reg_rate=%2.4e]: mean = %4.4e([+/-]%4.4e)'%(reg_ratei, np.array(reconstruction_error).mean(),np.array(reconstruction_error).std()))
                    
                if self.error_threshold >= np.array(reconstruction_error).mean():
                    min_error = np.array(epochs_error_sum).argmin() 
#                     print('**********************',epochi)
                    if self.log_steps:
                        print('Early stop! error <= threshold[reg_rate=%1.2e]: mean = %4.4e <= %1.2e '%(reg_ratei, np.array(reconstruction_error).mean(),self.error_threshold))
                    
                    continue_training = False
                    break

                if len(epochs_error_sum) >= 0.1*self.epochs: 
                    min_error = np.array(epochs_error_sum).argmin()
                    if min_error == 0:
                        if self.log_steps:
                            print(' * [reg_rate=%1.2e] Non-convergence until %d-th epoch error: mean = %4.4e([+/-]%4.4e)'%(reg_ratei, epochi, np.array(reconstruction_error).mean(),np.array(reconstruction_error).std()))
                        
                        break # don't stop regularization loop 
                    else:
                        #del epochs_Ce[0], epochs_error_sum[0]
                        epochs_Ce, epochs_error_sum = epochs_Ce[1:], epochs_error_sum[1:]
                
                del reconstruction_error
            
            min_error = np.array(epochs_error_sum).argmin()
            Ce = epochs_Ce[min_error]
            del epochs_Ce
                    
            self.eary_stop_epoch = epochi - min_error
        
            if reg_rate_min_error == None:
                reg_rate_min_error = epochs_error_sum[min_error]
            elif reg_rate_min_error <= epochs_error_sum[min_error]:
#                 print('!!!!!!!!!!!!!!!!!!!!!!!!!!![%1.2e]->%4.4e <= %4.4e'%(reg_ratei,reg_rate_min_error, epochs_error_sum[min_error]))
                if self.log_steps:
                    print('Early stop on [reg_rate=%4.4e] error sum = %4.4e'%(reg_ratei, epochs_error_sum[min_error]))

                continue_training = False
            
            elif reg_ratei <= self.reg_rate*10**(-6):
                if self.log_steps:
                    print('Early stop on [reg_rate=%1.2e] '%(reg_ratei))
                
                continue_training = False

#             else:
#                 print('[%1.2e] <= [%1.2e] : '%(reg_ratei,self.reg_rate*10**(-6)),reg_ratei <= self.reg_rate*10**(-6))
            
            
            reg_ratei *= 10**(-1)
        
            del epochs_error_sum
            
            epochs_error_sum = []
            
        self.eary_stop_reg_rate = reg_ratei
        sentence_vectors = (sti_vocab,pooling_order,Ce)
        del epochs_error_sum
        
        return sentence_vectors     


class TreeDeepRecursiveAutoEncoder(BaseEstimator, VectorizerMixin):
    
    def __init__(self,word_embeddings_path,unfold = False, n = 3, hidden_units = 200,eta = 0.01,reg_rate = 0.00001, epochs = 10, error_threshold=0.001, activation_function = Tanh(), debug = False, log_steps = False):
        self.word_embeddings_path,self.unfold,self.n,self.hidden_units, self.eta,self.reg_rate, self.epochs, self.error_threshold, self.debug, self.log_steps = (word_embeddings_path, unfold, n,hidden_units, eta, reg_rate, epochs, error_threshold, debug, log_steps)
        self.activation_function = activation_function
        
    def fit(self, documents_sentence_tuples, y=None):
        pass

    def fit_transform(self, documents_sentence_tuples, y=None):    
        return self.transform(documents_sentence_tuples)
    
        
    def transform(self, documents_sentence_tuples):
        
        documents_sentence_vectors = []
        
        for sentences_tuples in documents_sentence_tuples:
            documents_sentence_vectors.append([])
            
            for sentence_tuplei in sentences_tuples:
                
                sdrea = SentenceDeepRecursiveAutoEncoder(self.word_embeddings_path, self.unfold, self.n, self.hidden_units, self.eta, self.reg_rate, self.epochs, self.error_threshold, self.activation_function, self.debug, self.log_steps)
                sti_vocab, pooling_order, Ce = sdrea.fit_transform(sentence_tuplei, None)
                
                documents_sentence_vectors[-1].append((sti_vocab,pooling_order,Ce))
                
                if self.log_steps:
                    print('sentence vectors:',documents_sentence_vectors[-1][-1][-1].shape)
#                     for node_label,node_id in documents_sentence_vectors[-1][-1][0].items():
#                         print("[%d] %s ="%(node_id,node_label),documents_sentence_vectors[-1][-1][-1][node_id,:])  
        
        return np.array(documents_sentence_vectors)     

#---------------------------------------------------------------------------------------------------------------------------------------------------

class SentenceTreeDistancesPooling(BaseEstimator, VectorizerMixin):
    def __init__(self,resultant_matrix_size, just_root = False,sentence_stats=False, log_steps = False):
        self.resultant_matrix_size, self.just_root,self.sentence_stats, self.log_steps = resultant_matrix_size,just_root, sentence_stats, log_steps

    def fit(self, raw_documents, y=None):
        pass

    def fit_transform(self, raw_documents, y=None):    
        return self.transform(raw_documents)
    
    def transform(self, raw_documents):
        '''
            return for each document pair (d1,d2) : a tensor with all pooled distances between d1 and d2 sentences trees 

            * if d1 has m sentences and d2 n sentences the tensor will have m X n matrices
        '''
        
        pooled_distances = np.empty(len(raw_documents), dtype=np.ndarray)
        
        for i in range(0,len(raw_documents)):
            if self.log_steps:
                print('[%d/%d] - pairwise comparison and pooling!'%(i,len(raw_documents)))
            s_list = self.pairwise_document_sentences_trees_distance(raw_documents[i,0],raw_documents[i,1],self.just_root)

            s_pooled_list = np.array([[None]*s_list.shape[1]]*s_list.shape[0])
            for doc1_s in range(s_pooled_list.shape[0]):
                for doc2_s in range(s_pooled_list.shape[1]):
#                     print("s[%d][%d]"%(doc1_s,doc2_s),s_list[doc1_s][doc2_s])
                    s_pooled_list[doc1_s][doc2_s] = self(s_list[doc1_s][doc2_s])
#                     if self.log_steps:                    
#                         print("s[%d][%d]"%(doc1_s,doc2_s),s_list[doc1_s][doc2_s])
#                         print("s_pooled[%d][%d]"%(doc1_s,doc2_s),csr_matrix(s_pooled_list[doc1_s][doc2_s]))
            del s_list
            
            pooled_distances[i] = s_pooled_list

        return pooled_distances
        
    def split_axis(self,i, axis_size):
        if axis_size < self.resultant_matrix_size:
            raise Exception("axis_size (%d) < self.resultant_matrix_size (%d)!"%(axis_size, self.resultant_matrix_size))
        elif ((i < self.resultant_matrix_size) or ((axis_size % self.resultant_matrix_size) == 0)):
            return math.floor(axis_size/self.resultant_matrix_size)
        else:
            return math.floor(axis_size/self.resultant_matrix_size) + 1
    
    def window_min_distance(self,i,j,s):
        
        try:
            ri = self.split_axis(i,s.shape[0])
            ri1 = self.split_axis(i+1,s.shape[0])
        except Exception as e:
            if i > 0:
                raise e
            else:
                ri = ri1 = s.shape[0]
        try:
            cj = self.split_axis(j,s.shape[1])
            cj1 = self.split_axis(j+1,s.shape[1])
        except Exception as e:
            if j > 0:
                raise e
            else:
                cj = cj1 = s.shape[1]
                
        window_matrix = s[(i*ri):(i*ri + ri1),(j*cj):(j*cj + cj1)]
        
        return window_matrix.min()
        
    def __call__(self,similarity_matrix):

        if self.sentence_stats:
            s_pooled = (similarity_matrix.max())*ones((self.resultant_matrix_size+2,self.resultant_matrix_size+2))
        else:
            s_pooled = (similarity_matrix.max())*ones((self.resultant_matrix_size,self.resultant_matrix_size))
            
        i_range = self.resultant_matrix_size if self.resultant_matrix_size < similarity_matrix.shape[0] else 1
        j_range = self.resultant_matrix_size if self.resultant_matrix_size < similarity_matrix.shape[1] else 1
        
        if self.sentence_stats:
#             i_range += 2
#             j_range += 2
            
#             s_pooled[-2,:] = similarity_matrix.shape[0] - similarity_matrix.shape[1]
#             s_pooled[:,-2] = similarity_matrix.shape[0] - similarity_matrix.shape[1]
            equal_words = 0
        
        for i in range(0, i_range):
            for j in range(0, j_range):
                s_pooled[i,j] = self.window_min_distance(i, j, similarity_matrix)
#                 print("s_pooled[%d,%d] = %4.2f"%(i,j,s_pooled[i,j]))

                if self.sentence_stats and similarity_matrix[i,j] == 0:
                    equal_words += 1

                    # sentence length difference
                    s_pooled[-2,j] = max(i - j, j - i)
                    s_pooled[-1,j] = equal_words / (similarity_matrix.shape[1])
            
            if self.sentence_stats:
                # sentence length difference
                s_pooled[i,-2] = max(i - j, j - i)
                s_pooled[i,-1] = equal_words / (similarity_matrix.shape[0])

#         if self.sentence_stats:
#             s_pooled[-1,:] = equal_words / (similarity_matrix.shape[1])
#             s_pooled[:,-1] = equal_words / (similarity_matrix.shape[0])
#             print('equal_words:',equal_words,'/',(similarity_matrix.shape[0],similarity_matrix.shape[1]),'=',s_pooled[-1,:] ,s_pooled[:,-1] )
                
        return s_pooled 
                

    def pairwise_document_sentences_trees_distance(self,document1_tree_list,document2_tree_list,just_root):
        '''    
            document1_tree_list = list of all sentences trees structures from document1 
            document2_tree_list = list of all sentences trees structures from document2 
            
            evaluates euclidean distances between all sentences trees structures from document1 against all document2 sentences trees structures 
            
            return a tensor with all pairwise euclidean distances between the sentences trees of document1 and document2    
        '''
        
        sentence_trees_distances = []
        
#         print('===================')
        for d1_treei_vocab,d1_treei_pooling_order,d1_treei_Ce in document1_tree_list:
            del d1_treei_vocab
            sentence_trees_distances.append([])
            for d2_trej_vocab,d2_treej_pooling_order,d2_treej_Ce in document2_tree_list:
                del d2_trej_vocab
                sentence_trees_distances[-1].append(self.pairwise_sentence_vectors_distance(d1_treei_Ce[d1_treei_pooling_order,:],d2_treej_Ce[d2_treej_pooling_order,:],just_root))
#                 print('-->',sentence_trees_distances[-1][-1].shape)
#             print('-------')
                
        return np.array(sentence_trees_distances)
    
    
    def pairwise_sentence_vectors_distance(self,sentence1_vectors,sentence2_vectors, just_root = False):
        '''    
            sentence1_vectors = list of sentence1 structure nodes(tree + words) vectors
            sentence2_vectors = list of sentence1 structure nodes(tree + words) vectors
            
            evaluates euclidean distances between all sentence1_vectors vectors against sentence2_vectors vectors
            
            return a (sentence1_vectors.shape[0]) x (sentence2_vectors.shape[0]) euclidean distance matrix    
        '''
        
        if just_root:
            pd = pairwise_distances(X=sentence1_vectors[0:1,:], Y=sentence2_vectors[0:1,:])
        else:
            pd = pairwise_distances(X=sentence1_vectors, Y=sentence2_vectors)
        
        # nao consigo usar .min em lil_matrix.
        return csr_matrix(pd)    
        
class SentenceTreePairsFlattener(BaseEstimator, VectorizerMixin):
    
    def fit(self, sentences_trees, y=None):
        pass

    def fit_transform(self, sentences_trees, y=None):    
        return self.transform(sentences_trees)
    
    def transform(self, sentences_trees):
        shape_array = [sentences_trees.shape[0] + sentences_trees.shape[0]]
        shape_array = shape_array+list(sentences_trees.shape)[2:]
        
        sentences_trees = np.reshape(sentences_trees,tuple(shape_array ))
        
        return sentences_trees   
    
class SentenceTreeFlattentoPairs(BaseEstimator, VectorizerMixin):
    """
        reshapes dataset tree list to dataset tree pairs
    """
    
    def fit(self, sentences_trees, y=None):
        pass

    def fit_transform(self, sentences_trees, y=None):    
        return self.transform(sentences_trees)
    
    def transform(self, sentences_trees):
        shape_array = [sentences_trees.shape[0]/2,2]
        shape_array = shape_array+list(sentences_trees.shape)[1:]
        sentences_trees = np.reshape(sentences_trees,tuple(shape_array ))
        
        return sentences_trees  
    
class PooledDistancesKMinSelector(BaseEstimator, VectorizerMixin):
    
    def __init__(self, log_steps = False, k = 1):
        self.log_steps, self.k = log_steps, k
    
    def fit(self, similarity_trees, y=None):
        pass

    def fit_transform(self, similarity_trees, y=None):    
        return self.transform(similarity_trees)
    
    def transform(self, similarity_trees):
        min_distances = np.empty((similarity_trees.shape[0], self.k), dtype=np.ndarray)
        for i in range(similarity_trees.shape[0]):
            sum_distances = []
            matrices_indexes = []
            
            for j in range(similarity_trees[i].shape[0]):
                for k in range(similarity_trees[i].shape[1]):
                    
                    sum_distances.append(similarity_trees[i][j,k].sum())
                    matrices_indexes.append((j,k))
                    
                    if self.log_steps:
                        print('sum_distances[%d,%d,%d]=%4.2f'%(i,j,k,sum_distances[-1]))
            
            for j,ranking_index in enumerate(argsort(sum_distances)[0:self.k]):

                x, y = matrices_indexes[ranking_index]
                min_distances[i,j] = similarity_trees[i][x, y]
                
                if self.log_steps:
                    print('min_distances[%d,%d]=%4.2f'%(i,j,min_distances[i,j].sum()))
            
            # if exist less sentences comparisons then k (repeat the min distance sum)
            for j in range(len(sum_distances), self.k):
                min_distances[i,j] = min_distances[i,0] 
                
            del sum_distances, matrices_indexes
        
        return min_distances

#     def transform(self, similarity_trees):
#         min_distances = np.empty(similarity_trees.shape[0], dtype=np.ndarray)
#         for i in range(similarity_trees.shape[0]):
#             sum_distances = np.empty(similarity_trees[i].shape, dtype='d')
#             j_min, k_min, min_sum = -1, -1, np.inf
#             
#             for j in range(similarity_trees[i].shape[0]):
#                 for k in range(similarity_trees[i].shape[1]):
#                     
#                     sum_distances[j,k] = similarity_trees[i][j,k].sum()
# 
#                     if sum_distances[j,k] < min_sum:
#                         j_min, k_min, min_sum = j, k, sum_distances[j,k] 
#                     if self.log_steps:
#                         print('sum_distances[%d,%d]=%4.2f'%(j,k,sum_distances[j,k]))
#                         print('min_sum[%d,%d]=%4.2f'%(j_min,k_min,min_sum))
#             
#             min_distances[i] = similarity_trees[i][j_min, k_min]
#         return min_distances
    
class SimilaritiesTreesFlattener(BaseEstimator, VectorizerMixin):
    
    def __init__(self, log_steps = False):
        self.log_steps = log_steps
    
    def fit(self, similarity_trees, y=None):
        pass

    def fit_transform(self, similarity_trees, y=None):    
        return self.transform(similarity_trees)
    
    def transform(self, similarity_trees):
        
        flatten_arrays = np.empty((similarity_trees.shape[0],similarity_trees[0].shape[0]*similarity_trees[0][0].shape[0]**2))
        for i in range(similarity_trees.shape[0]):
            flatten_arrays[i] = np.array((np.hstack(similarity_trees[i])).flatten())
            if self.log_steps:
                print('flatten_array[%d] with %d elements! with %d nnz elements!'%(i,len(flatten_arrays[i]),csr_matrix(flatten_arrays[i]).nnz))
                
        return flatten_arrays    

class DatasetFlattener(BaseEstimator, VectorizerMixin):
    
    def fit(self, raw_document_pair, y=None):
        pass

    def fit_transform(self, raw_document_pair, y=None):    
        return self.transform(raw_document_pair)
    
    def transform(self, raw_document_pair):
        return raw_document_pair.flatten()
    
    
class WordnetSentenceSSWVectors(BaseEstimator, VectorizerMixin):
    '''
        using WordNet Information content to eval sentences pairs semantic-vectors and syntactic-vectors (this approach meansures distances)
    '''
    
    def __init__(self, wordnet_ic_filename='ic-brown.dat', alpha=0.5, log_steps=False):
        self.wordnet_ic_filename,self.alpha,self.log_steps = wordnet_ic_filename,alpha,log_steps
        self.wordnet_ic = wordnet_ic.ic(wordnet_ic_filename)
    
    def fit(self, raw_document_pair, y=None):
        pass

    def fit_transform(self, raw_document_pair, y=None):    
        return self.transform(raw_document_pair)
    
    def transform(self, raw_document_pair):
        '''
            ssw_distances  #d1 sentences X #d2 sentences array of 1x1 matrices to be compatible with sentence tree pooling results
            
#             return 1 - ssw_distances
        '''
        ssw_distances = np.empty(raw_document_pair.shape[0], dtype=np.ndarray)

        for i in range(raw_document_pair.shape[0]):
            if self.log_steps:
                print('[%d/%d] - pairwise comparison, syntactic and semantic vectors similarities!'%(i,raw_document_pair.shape[0]))
            
            d1_sentences = raw_document_pair[i,0]
            d2_sentences = raw_document_pair[i,1]
            ssw_distances[i] = np.array([[None]*len(d2_sentences)]*len(d1_sentences))
            
            for a in range(ssw_distances[i].shape[0]):
                for b in range(ssw_distances[i].shape[1]):
                    semantic_vector, syntactic_vector, word_set = wordnet_syntatic_semantic_vectors(d1_sentences[a][3],d2_sentences[b][3],self.wordnet_ic)
                    
                    if lil_matrix(semantic_vector).nnz > 0 :
                        sem_dist = pairwise_distances(semantic_vector.T, metric='cosine', n_jobs=1)[0,1]
                    else:
                        sem_dist = 1

                    if lil_matrix(syntactic_vector).nnz > 0 :
                        sy_v1 = semantic_vector[:,0].T
                        sy_v2 = semantic_vector[:,1].T
                        syn_dist = pairwise_distances(X=[sy_v1,sy_v2], metric='euclidean', n_jobs=1)[0,1]   # a-b
                        syn_dist /= pairwise_distances(X=[sy_v1,-sy_v2], metric='euclidean', n_jobs=1)[0,1] # a+b
                        
                        del sy_v1, sy_v2
                    else:
                        syn_dist = 1
                    
                    d1_d2_dist = self.alpha*(sem_dist) + (1-self.alpha)*syn_dist
                    ssw_distances[i][a][b] = d1_d2_dist 
#                     if self.log_steps:                           
#                         print('ssw_distances[%d][%d][%d] = (%4.2f)%4.2f + (1 - %4.2f)%4.2f = %4.2f'%(i,a, b,self.alpha, sem_dist,self.alpha, syn_dist, ssw_distances[i][a][b]))
                    
        return 1 - ssw_distances
    
class PDLKEstimator(BaseEstimator):
    def __init__(self, threshold, log_steps=False):
        self.threshold, self.log_steps = threshold, log_steps
    
    def fit(self,X,y):
        pass
    
    def predict_proba(self, X):
        
        probas = np.zeros((X.shape[0],2))
        for pos,value in enumerate(X):
            if (value >= self.threshold).any():
                probas[pos,1] = 1
            else:    
                probas[pos,0] = 1
         
        return probas
    
    def predict(self,X):
        
        probas = self.predict_proba(X)
        y_predict = probas[:,1] ==  np.argmax(probas[:,:])
        return y_predict
    
    
if __name__ == '__main__':
    
#     msrp_corpus, msrp_target, msrp_train_size, msrp_test_size = load_MSRParaphraseCorpus()
    spa_corpus, spa_target, spa_train_size, spa_test_size, tam_queries, tam_corpus, targets = load_short_plagiarized_answers_as_pairs()
#    cpc_corpus, cpc_target, cpc_train_size, cpc_test_size = load_cpc11Corpus()
#     spa_corpus, spa_target, spa_train_size, spa_test_size = load_meter_as_pairs()
#     p4p_corpus, p4p_target, p4p_train_size, p4p_test_size = load_p4p_pairs()
#    pan_corpus, pan_target, pan_train_size, pan_test_size = load_pan_plagiarism_corpus_2011()
    
    
    all_texts,all_target = spa_corpus, spa_target
#     all_texts,all_target = msrp_corpus,msrp_target
#    all_texts,all_target = cpc_corpus,cpc_target
#     all_texts,all_target = p4p_corpus,p4p_target
#    all_texts,all_target = pan_corpus,pan_target
    
#   all_texts,all_target = all_texts[:995],all_target[:995]
#   all_texts,all_target = all_texts[900:1000],all_target[900:1000]
    #all_texts,all_target = all_texts[:5],all_target[:5]
    
    
    #all_texts,all_target = np.array([["Grandpa saw Grandma","Grandpa saw Grandma and Grandma saw Grandpa"],["Fries tastes good","bacon tastes better"]]),np.array([1,0])
    
    word_embeddings_path = "/home/jones/Documentos/BRI/WordEmbedding/"
    
    print('all_texts,all_target : ',len(all_texts),len(all_target))
    
    parser_model_path="/home/jones/Documentos/retrieval_pipeline/libs/stanford-parser/englishPCFG.ser.gz" 
    parser_path_to_jar='/home/jones/Documentos/retrieval_pipeline/libs/stanford-parser/stanford-parser.jar' 
    parser_path_to_models_jar='/home/jones/Documentos/retrieval_pipeline/libs/stanford-parser/stanford-parser-3.5.2-models.jar' 
    log_steps = False

    pipeline = Pipeline([
        ("DatasetFlattener",DatasetFlattener()),
        ("Tokenizer",SentenceTreeTokenizer(parser_model_path=parser_model_path,parser_path_to_jar=parser_path_to_jar,parser_path_to_models_jar=parser_path_to_models_jar,log_steps=log_steps )),
        ("RAE",TreeDeepRecursiveAutoEncoder(word_embeddings_path, unfold = False, n = 25,hidden_units=2000,epochs=20,eta = 0.015,log_steps=True)),
    ])

    ft = pipeline.fit_transform(all_texts, all_target)
    pickle.dump(all_texts, open("/home/jones/resultados/at.txt", "wb"))
    pickle.dump(ft, open("/home/jones/resultados/ft.txt", "wb"))
    pickle.dump(tam_queries, open("/home/jones/resultados/tam_queries.txt", "wb"))
    pickle.dump(tam_corpus, open("/home/jones/resultados/tam_corpus.txt", "wb"))
    pickle.dump(targets, open("/home/jones/resultados/targets.txt", "wb"))

    
    



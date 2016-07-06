import os.path
import pickle
import numpy as np

class ACL2010WordEmbedding():
    
    FILE_NAMES = {200 : "model-1750000000.LEARNING_RATE=1e-09.EMBEDDING_LEARNING_RATE=1e-06.EMBEDDING_SIZE=200.txt",
                  100 : "model-2030000000.LEARNING_RATE=1e-09.EMBEDDING_LEARNING_RATE=1e-06.EMBEDDING_SIZE=100.txt",
                   50 : "model-2280000000.LEARNING_RATE=1e-08.EMBEDDING_LEARNING_RATE=1e-07.EMBEDDING_SIZE=50.txt",
                   25 : "model-2280000000.LEARNING_RATE=1e-08.EMBEDDING_LEARNING_RATE=1e-07.EMBEDDING_SIZE=25.txt",
                }

    def __init__(self,embedding_size,file_path):
        self.embedding_size, self.file_path = embedding_size,file_path
        
    def __call__(self,word_list):
        
        if os.path.exists(os.path.join(self.file_path,"embedding_%s.plk"%self.embedding_size)):
            with open(os.path.join(self.file_path,"embedding_%s.plk"%self.embedding_size),"rb") as f: 
                embedding_dict = pickle.load(f)
        else:
            with open(os.path.join(self.file_path,self.FILE_NAMES[self.embedding_size]),'r') as json_data:
                embedding_dict = {None:np.zeros(self.embedding_size)}
                
                for linei in json_data.readlines():
                    linei_tokens = linei.replace('\n','').split(' ')
                    embedding_dict[linei_tokens[0]] = np.array([ float(linei_tokens[i]) for i in range(1,self.embedding_size+1)])
                    embedding_dict[None] += embedding_dict[linei_tokens[0]] 
                
                with open(os.path.join(self.file_path,"embedding_%s.plk"%self.embedding_size),'wb') as f:
                    pickle.dump(embedding_dict,f)
             
        resultant_matrix = np.zeros((len(word_list),self.embedding_size),)
        for i in range(len(word_list)):
            if word_list[i] in embedding_dict.keys():
                resultant_matrix[i,:] = embedding_dict[word_list[i]]
            else: 
                '''
                    out-of-vocabulary word is the sum vector from all words 
                '''
                 
                resultant_matrix[i,:] = embedding_dict[None]
                
        del embedding_dict
        
        return resultant_matrix  
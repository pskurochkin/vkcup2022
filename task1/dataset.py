import numpy as np

import torch
from torch.utils import data

from sklearn.model_selection import train_test_split

from tokenizer import MystemTokenizer
from gensim.models import KeyedVectors

from tqdm import tqdm


CATEGORIES = ['athletics', 'autosport', 'basketball',
              'boardgames','esport','extreme','football',
              'hockey','martial_arts','motosport',
              'tennis','volleyball','winter_sport']


class GroupsDataset(data.Dataset):
    def __init__(self, word2vec_file):
        self.cat2id = {c: id_c for id_c, c in enumerate(CATEGORIES)}
        self.id2cat = {id_c: c for id_c, c in enumerate(CATEGORIES)}
        
        self.tokenizer = MystemTokenizer()
        
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_file)
        self.word2vec.fill_norms()
        
        self.training_mode = True
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
    
    def __len__(self):
        if self.training_mode:
            return len(self.X_train)
        else:
            return len(self.X_test)
    
    def __getitem__(self, idx):
        if self.training_mode:
            embedding, label = self.X_train[idx], self.y_train[idx]
            
            return torch.tensor(embedding), torch.tensor(label)
        else:
            if self.y_test is None:
                embedding = self.X_test[idx]
                
                return torch.tensor(embedding)
            else:
                embedding, label = self.X_test[idx], self.y_train[idx]
                
                return torch.tensor(embedding), torch.tensor(label)
    
    def train(self):
        self.training_mode = True
    
    def eval(self):
        self.training_mode = False
    
    def preprocess_train(self, dataframe, split_train_test=False):
        df_groupby = dataframe.groupby(['oid', 'category'], as_index = False)
        df_groupby = df_groupby.agg({'text': ' '.join}).reset_index(drop=True)
        
        labels, vectors = [], []
        for oid, category, text in tqdm(df_groupby.values):
            embeddings = [self.word2vec.get_vector(token) \
                          for token in self.tokenizer(text) if token in self.word2vec]
            
            labels.append(self.cat2id[category])
            vectors.append(np.mean(embeddings, axis=0))
            
        labels = np.array(labels, dtype=np.int32)
        vectors = np.array(vectors, dtype=np.float32)
        
        if split_train_test:
            self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(vectors, labels, test_size=0.3, stratify=labels)
        else:
            self.X_train, self.y_train = vectors, labels
    
    def preprocess_test(self, dataframe):
        if self.X_test is not None:
            raise RuntimeError('X_test is already exist!')
        
        df_groupby = dataframe.groupby(['oid'], as_index = False)
        df_groupby = df_groupby.agg({'text': ' '.join}).reset_index(drop=True)
        
        vectors = []
        for oid, text in tqdm(df_groupby.values):
            embeddings = [self.word2vec.get_vector(token) \
                          for token in self.tokenizer(text) if token in self.word2vec]
            
            vectors.append(np.mean(embeddings, axis=0))
        
        vectors = np.array(vectors, dtype=np.float32)
        
        self.X_test, self.y_test = vectors, None

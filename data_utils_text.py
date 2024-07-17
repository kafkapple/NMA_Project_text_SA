import os
import numpy as np
import torch
import torchaudio
import requests
import zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from config_text import config
import pandas as pd
import nltk
import spacy
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# preprocess the data (common across all models)
from sentence_transformers import SentenceTransformer

# word2vec
from gensim.models import Word2Vec
# # url = "https://raw.githubusercontent.com/ataislucky/Data-Science/main/dataset/emotion_train.txt"

class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        return features, label
    
def collate_batch(batch):
    features, labels = zip(*batch)
    features_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.float32) for f in features], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return features_padded, labels

def prepare_dataloaders(data, labels, BATCH_SIZE):

    full_dataset = TextDataset(data, labels)
    train_size = int(config.ratio_train * len(full_dataset))
    val_size = int(config.ratio_test * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(config.SEED))
    print(f"\nTrain/Val/Test set splitted with batch size {config.BATCH_SIZE}: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}\n")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    
    
    return train_loader, val_loader, test_loader

def preprocess_text(text):
      # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word.lower() not in stop_words]

    # Rejoin tokens to create the cleaned sentence
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def data_prep_text():
    
    
    st_encoder = SentenceTransformer('all-MiniLM-L12-v2')
      # Download necessary NLTK data
    nltk.download('stopwords')
    nltk.download('wordnet')

    # # Load spaCy model
    # nlp = spacy.load('en_core_web_sm')

    # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    data_1 = pd.read_csv(os.path.join(config.DATA_DIR,"val.txt"), sep=";")
    data_1.columns = ["Text", "Emotions"]
    data_3 = pd.read_csv(os.path.join(config.DATA_DIR,"validation.csv")).rename(columns={"text":"Text", "label":"Emotions"})
    data_3["Emotions"] = data_3["Emotions"].map({0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"})

    # Train Word2Vec model
    data_1_3 = pd.concat([data_1, data_3])
    sentences = [sentence.split() for sentence in data_1_3['Text']]

    VECTOR_SIZE = 100
    MIN_COUNT = 5
    WINDOW = 3
    SG = 1

    w2v_model = Word2Vec(sentences, vector_size=VECTOR_SIZE, min_count=MIN_COUNT, window=WINDOW, sg=SG)

    # encode the data
    data_1_3['Cleaned_Text'] = data_1_3['Text'].apply(preprocess_text).dropna()
    data_1_3["hf_embed"] = data_1_3['Cleaned_Text'].apply(lambda x: st_encoder.encode(x))

    # Obtain word embeddings for data_1.Text and train a svm model on it with class being data_1.Emotion and measure accuracy

    # Apply preprocessing to the text data
    data_1_3['Cleaned_Text'] = data_1_3['Text'].apply(preprocess_text).dropna()

    # Get word embeddings for the cleaned text
    X = data_1_3['Cleaned_Text'].apply(lambda sent: w2v_model.wv.get_mean_vector([word for word in sent.split()]))
    X = np.stack(X.values)

    # Continue with mapping emotions to numerical labels, and splitting the data
    y = data_1_3['Emotions'].map({'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}).values
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get word embeddings for data.Text
    data = data_1_3.copy()
    X = data['hf_embed']
    X = np.stack(X.values)
    # Map emotions to numerical labels
    y = data['Emotions'].map({'sadness':0, 'anger':1, 'love':2, 'surprise':3, 'fear':4, 'joy':5}).values
    print(f'Data size: {X.shape, y.shape}\nPreparing Text dataloader...')
    # Split data into train and test sets
    
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    
    train_loader, val_loader, test_loader=prepare_dataloaders(X, y, config.BATCH_SIZE)
    return train_loader, val_loader, test_loader#_train, X_val, X_test, y_train, y_val, y_test

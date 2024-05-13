import pandas as pd
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from wordcloud import WordCloud
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import re
import numpy as np

# Define a dictionary of contractions and their expanded forms
contractions_dict = {
    "i'm": "I am",
    "i'll": "I will",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not"

}

# Function to replace contractions in a string with all lowercase letters
def expand_contractions(text, contractions_dict):
    # Regular expression for finding contractions
    contractions_re = re.compile('(%s)' % '|'.join(map(re.escape, contractions_dict.keys())))
    
    def replace(match):
        return contractions_dict[match.group(0)]
    
    return contractions_re.sub(replace, text)

def split_text_into_sentence_force(text):
    # Remove all characters that are not alphanumeric or spaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Split using period followed by whitespace or end of text to denote sentence boundaries
    sentences = re.split(r'\.\s+|\.$', cleaned_text)
    
    # Join all sentences into one single text
    all_words = []
    for sentence in sentences:
        words = sentence.split()
        if words:  # Check if the sentence is not empty
            all_words.extend(words)
    
    # Now split all_words into chunks of 50 words each
    processed_sentences = []
    for i in range(0, len(all_words), 50):
        chunk = all_words[i:i + 50]
        # Convert chunk of words back into a single sentence string
        if chunk:  # Ensure the chunk is not empty
            processed_sentences.append(' '.join(chunk))
    
    return processed_sentences

def split_text_into_sentence(text):
    # Split using period followed by whitespace or end of text to denote sentence boundaries
    sentences = re.split(r'\.\s+|\.$', text)
    
    # Process each sentence to handle long sentences, remove empty sentences, and skip short sentences
    processed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        # Skip any empty sentences or sentences shorter than 10 words
        if len(words) < 20:
            continue
        
        # If the sentence is longer than 100 words, chunk it into parts of 50 words each
        if len(words) > 100:
            for i in range(0, len(words), 50):
                chunk = words[i:i + 50]
                # Ensure the chunk is not empty and meets the minimum word count
                if len(chunk) >= 20:
                    processed_sentences.append(' '.join(chunk))
        else:
            # Add the sentence if it meets the minimum word count
            joined_sentence = ' '.join(words)
            if joined_sentence:
                processed_sentences.append(joined_sentence)
    
    return processed_sentences

#pre-padding
def convert_tokens_to_indices(documents, vocab):
    max_sentences = max(len(doc) for doc in documents)
    max_len = max(len(sentence) for doc in documents for sentence in doc)

    indexed_documents = []
    for doc in documents:
        indexed_sentences = []
        for sentence in doc:
            indexed_sentence = [vocab[token] for token in sentence]
            # Pre-pad each sentence to max_len
            indexed_sentence = [vocab["<pad>"]] * (max_len - len(indexed_sentence)) + indexed_sentence
            indexed_sentences.append(indexed_sentence)
        # Pre-pad the number of sentences in each document
        while len(indexed_sentences) < max_sentences:
            indexed_sentences = [[vocab["<pad>"]] * max_len] + indexed_sentences
        indexed_documents.append(indexed_sentences)

    indexed_documents_tensor = torch.tensor(indexed_documents)
    return indexed_documents_tensor

def apply_word_embeddings(vocab):
    # Load the pre-trained Word2Vec model
    word2vec = KeyedVectors.load_word2vec_format('data_preprocess/GoogleNews-vectors-negative300.bin', binary=True)
    embedding_dim = word2vec.vector_size  # Typically 300 for Google's Word2Vec

    # Initialize the embedding matrix
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.get_stoi().items():
        if word in word2vec:
            embedding_matrix[idx] = word2vec[word]
        else:
            # Initialize with a random vector
            embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

    # Convert the embedding matrix to a tensor
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    return embedding_matrix

def yield_tokens(data):
    for document in data:
        for sentence in document:
            yield sentence

class EssaysDataset(Dataset):
    def __init__(self, documents, labels):
        self.documents = documents
        self.labels = labels
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return torch.tensor(self.documents[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def preprocess_data():
    # Load dataset
    data = pd.read_csv('data_preprocess/essays.csv', encoding='mac_roman')  #Ensure the correct encoding is specified

    # Clean the text
    data['TEXT'] = data['TEXT'].str.lower().apply(lambda x: expand_contractions(x, contractions_dict))
    data['TEXT'] = data['TEXT'].str.replace(r'[^a-zA-Z0-9\s\.]', '',regex=True)
    #print(data['TEXT'][0])
    data['PROCESSED_TEXT']= data['TEXT'].apply(lambda x: split_text_into_sentence(x))

    # Tokenization
    tokenizer = get_tokenizer('basic_english')
    tokenized_texts = [[tokenizer(text) for text in sentence]for sentence in data['PROCESSED_TEXT']]
    data['TOKENIZED_TEXT'] = tokenized_texts

    # Build vocabulary
    vocab = build_vocab_from_iterator(yield_tokens(data['TOKENIZED_TEXT']), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    # Numericalize and pad the tokenized text
    numericalized_texts = convert_tokens_to_indices(data['TOKENIZED_TEXT'], vocab)

    # Initialize Labels
    labels = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].replace({'y': 1, 'n': 0}).values
    #labels = data[['cEXT']].replace({'y': 1, 'n': 0}).values
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
 
    # Initialize the dataset
    full_dataset = EssaysDataset(numericalized_texts, labels_tensor)

    # Calculate split sizes
    train_size = int(0.7 * len(full_dataset))
    val_size = (len(full_dataset) - train_size) // 2
    test_size = len(full_dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    return train_dataset,  val_dataset, test_dataset, vocab

def prepare_data_for_bert(model, texts, max_length):
    tokenizer = BertTokenizer.from_pretrained(model)

    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,                      # Text to encode
            add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
            max_length=max_length,     # Pad & truncate all sentences
            padding='max_length',      # Temporarily pad all to max length
            return_attention_mask=True,# Construct attn. masks
            truncation=True,           # Ensure below max_length
            return_tensors='pt',       # Return pytorch tensors
        )
        
        # Manually shift padding to the front for pre-padding
        input_id = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        
        # Count the number of padding tokens
        pad_count = (input_id == tokenizer.pad_token_id).sum().item()
        
        # Create pre-padded input_id and attention_mask
        pre_padded_input_id = torch.cat([torch.full((pad_count,), tokenizer.pad_token_id, dtype=torch.long),
                                         input_id[input_id != tokenizer.pad_token_id]])
        pre_padded_attention_mask = torch.cat([torch.zeros(pad_count, dtype=torch.long),
                                               attention_mask[attention_mask == 1]])
        
        input_ids.append(pre_padded_input_id)
        attention_masks.append(pre_padded_attention_mask)

    # Convert lists to tensors
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    return input_ids, attention_masks

def plot_wordcloud(text):
    # Flatten the list of lists into a single string
    words = ' '.join([' '.join(sublist) for sublist in text])
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig('wordcloud.png')

#train_dataset,  val_dataset, test_dataset, vocab = preprocess_data()
#print(len(train_dataset), len(val_dataset), len(test_dataset), len(vocab))
#preprocess_data()
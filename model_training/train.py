import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from data_preprocess.preprocess import apply_word_embeddings, preprocess_data


class HierarchicalTextModel(nn.Module):
    def __init__(self, embedding_matrix, sentence_hidden_dim, document_hidden_dim, output_dim, num_layers=1):
        super(HierarchicalTextModel, self).__init__()
        # Create the embedding layer pre-initialized
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = True  # Set False to freeze embeddings
        
        # Sentence-level LSTM
        self.sentence_lstm = nn.LSTM(embedding_dim, sentence_hidden_dim, batch_first=True, num_layers=num_layers)
        
        # Document-level LSTM
        self.document_lstm = nn.LSTM(sentence_hidden_dim, document_hidden_dim, batch_first=True, num_layers=num_layers)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(document_hidden_dim, output_dim)
        
    def forward(self, documents):
        batch_size, num_sentences, sentence_length = documents.shape
        
        documents = documents.view(-1, sentence_length)
        embedded_sentences = self.embedding(documents)
        
        _, (sentence_hidden, _) = self.sentence_lstm(embedded_sentences)
        sentence_hidden = sentence_hidden[-1]
        
        sentence_hidden = sentence_hidden.view(batch_size, num_sentences, -1)
        
        _, (document_hidden, _) = self.document_lstm(sentence_hidden)
        document_hidden = document_hidden[-1]
        
        output = self.fc(document_hidden)
        return output

def train():
    # Define the model
    sentence_hidden_dim = 128
    document_hidden_dim = 256
    output_dim = 5
    num_layers = 1

    train_dataset,  val_dataset, test_dataset, vocab = preprocess_data()
    embedding_matrix = apply_word_embeddings(vocab)

    model = HierarchicalTextModel(embedding_matrix, sentence_hidden_dim, document_hidden_dim, output_dim, num_layers)

    # Set batch size
    batch_size = 2

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Assume the model and other components are set up as before
    for epoch in range(1):
        # Training Phase
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")
        
        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

    # Testing Phase
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for texts, labels in test_loader:
            predictions = model(texts)
            predictions = torch.sigmoid(predictions).round()  # Convert to binary predictions
            y_pred.extend(predictions.numpy())
            y_true.extend(labels.numpy())

    # Calculate accuracy for each trait
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    trait_names = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
    for i, trait in enumerate(trait_names):
        trait_accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        print(f'Accuracy for {trait}: {trait_accuracy:.4f}')

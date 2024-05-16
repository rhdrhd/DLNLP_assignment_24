import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torch.nn.functional as F
import wandb

from data_preprocess.preprocess import apply_word_embeddings, preprocess_data, prepare_data_for_bert


def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

class Customized_LSTM(nn.Module):
    def __init__(self, embedding_matrix, sentence_hidden_dim, document_hidden_dim, output_dim, num_layers=1, bidirectional=False, dropout=0.1):
        super(Customized_LSTM, self).__init__()
        # Create the embedding layer pre-initialized
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = True  # Set False to freeze embeddings
        
        # Sentence-level LSTM
        self.sentence_lstm = nn.LSTM(
            embedding_dim, 
            sentence_hidden_dim, 
            batch_first=True, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Document-level LSTM
        self.document_lstm = nn.LSTM(
            sentence_hidden_dim * (2 if bidirectional else 1), 
            document_hidden_dim, 
            batch_first=True, 
            num_layers=num_layers, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(document_hidden_dim * (2 if bidirectional else 1), output_dim)
        
    def forward(self, documents):
        batch_size, num_sentences, sentence_length = documents.shape
        
        documents = documents.view(-1, sentence_length)
        embedded_sentences = self.embedding(documents)


        # Normalize the embeddings
        embedded_sentences = F.normalize(embedded_sentences, p=2, dim=2)
        #embedded_sentences = embedded_sentences[:,:8]
        _, (sentence_hidden, _) = self.sentence_lstm(embedded_sentences)
        
        if self.sentence_lstm.bidirectional:
            sentence_hidden = torch.cat((sentence_hidden[-2], sentence_hidden[-1]), dim=1)
        else:
            sentence_hidden = sentence_hidden[-1]
        
        sentence_hidden = sentence_hidden.view(batch_size, num_sentences, -1)
        #sentence_hidden = sentence_hidden[:,:12]
        _, (document_hidden, _) = self.document_lstm(sentence_hidden)
        
        if self.document_lstm.bidirectional:
            document_hidden = torch.cat((document_hidden[-2], document_hidden[-1]), dim=1)
        else:
            document_hidden = document_hidden[-1]
        
        document_hidden = self.dropout(document_hidden)
        
        output = self.fc(document_hidden)
        output = torch.sigmoid(output)

        return output

def run_lstm_sweep():
    sweep_config = {
        'method': 'random',  # 'grid', 'random', or 'bayes'
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-6,
                'max': 1e-4
            },
            'batch_size': {
                'values': [4, 8, 16, 32]
            },
            'epochs': {
                'values': [30, 50]  # Shorter for quick sweeps
            },
            'sentence_hidden_dim': {
                'values': [128, 256, 512]
            },
            'document_hidden_dim': {
                'values': [128, 256, 512]
            },
            'optimizer': {
                'values': ['adam', 'adamw']
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="nlp-personality-prediction")
    wandb.agent(sweep_id, lstm_sweep_setup, count=20)

def lstm_sweep_setup():
    wandb.init(project="nlp-personality-prediction")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Use wandb config
    sentence_hidden_dim = wandb.config.sentence_hidden_dim
    document_hidden_dim = wandb.config.document_hidden_dim
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    optimizer_type = wandb.config.optimizer
    trait_num = 5
    num_layers = 1

    train_dataset,  val_dataset, test_dataset, vocab = preprocess_data()
    embedding_matrix = apply_word_embeddings(vocab)

    model = Customized_LSTM(embedding_matrix, sentence_hidden_dim, document_hidden_dim, trait_num, num_layers)
    model.to(device)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training phase
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss_epoch_avg = total_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss_epoch_avg})
        
        # Validation phase
        model.eval()
        total_eval_correct = 0
        total_eval = 0
        total_eval_loss = 0
        total_eval_correct_col = np.zeros(trait_num)
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_eval_loss += loss.item()

                # Convert probabilities to predicted classes based on threshold
                preds = (outputs >= 0.5).long() 
                correct = (preds == targets).cpu().numpy()
                total_eval_correct += correct.sum()
                total_eval_correct_col += correct.sum(axis=0)
                total_eval += correct.size
            
            avg_val_accuracy = total_eval_correct / total_eval
            avg_val_accuracy_col = total_eval_correct_col / (total_eval/trait_num)
            avg_val_loss = total_eval_loss / len(val_loader)

            wandb.log({"val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy})

    wandb.finish()

def train_lstm(epoch=50):
    seed_everything(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Define the model
    sentence_hidden_dim = 256
    document_hidden_dim = 512
    trait_num = 5
    num_layers = 1
    batch_size = 32
    learning_rate = 0.00005964911225837496
    epochs = epoch

    train_dataset,  val_dataset, test_dataset, vocab = preprocess_data()
    embedding_matrix = apply_word_embeddings(vocab)

    model = Customized_LSTM(embedding_matrix, sentence_hidden_dim=sentence_hidden_dim, document_hidden_dim=document_hidden_dim, output_dim=trait_num, num_layers=num_layers, bidirectional=True, dropout=0.1)
    model_save_path_loss = "model_weights/best_model_lstm_loss.pth"
    model_save_path_acc = "model_weights/best_model_lstm_acc.pth"
    model.to(device)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    wandb_on = False
    if wandb_on:
        wandb.init(
            project="nlp-personality-prediction",
            config={
                "model": model,
                "epochs": epochs,
                "batch_size": batch_size,
                "sentence_hidden_dim": sentence_hidden_dim,
                "document_hidden_dim": document_hidden_dim,
                "optimizer": optimizer,
                "criterion": criterion,
                "trait_number": trait_num,
            },
            notes="None",
            )
        
    best_val_loss = float('inf')
    best_val_acc = 0.0
    # Assume the model and other components are set up as before
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        
        train_loss_epoch_avg = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss_epoch_avg}")
        
        # Validation Phase
        model.eval()
        total_eval_correct = 0
        total_eval = 0
        total_eval_loss = 0
        total_eval_correct_col = np.zeros(trait_num)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_eval_loss += loss.item()

                # Convert probabilities to predicted classes based on threshold
                preds = (outputs >= 0.5).long() 
                correct = (preds == targets).cpu().numpy()
                total_eval_correct += correct.sum()
                total_eval_correct_col += correct.sum(axis=0)
                total_eval += correct.size
                
            avg_val_accuracy = total_eval_correct / total_eval
            avg_val_accuracy_col = total_eval_correct_col / (total_eval/trait_num)
            avg_val_loss = total_eval_loss / len(val_loader)

        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}, Validation Accuracy per trait: {avg_val_accuracy_col}")
        #if wandb_on: wandb.log({"epoch": epoch, "train loss": train_loss_epoch_avg, "val loss": avg_val_loss})
        # Check if this is the best model based on validation accuracy
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model
            torch.save(model.state_dict(), model_save_path_loss)
            print(f"New best model with min loss saved with validation loss: {avg_val_loss}, and accuracy {avg_val_accuracy}")
        if avg_val_accuracy > best_val_acc:
            best_val_acc = avg_val_accuracy
            torch.save(model.state_dict(), model_save_path_acc)
            print(f"New best model with max acc saved with validation loss: {avg_val_loss}, and accuracy {avg_val_accuracy}")



def test_lstm(model_save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    # Define the model
    sentence_hidden_dim = 256
    document_hidden_dim = 512
    trait_num = 5
    num_layers = 1
    batch_size = 32

    _,  _, test_dataset, vocab = preprocess_data()
    embedding_matrix = apply_word_embeddings(vocab)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = Customized_LSTM(embedding_matrix, sentence_hidden_dim=sentence_hidden_dim, document_hidden_dim=document_hidden_dim, output_dim=trait_num, num_layers=num_layers, bidirectional=True, dropout=0.1)
    # Test phase
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()
    total_test_correct = 0
    total_test = 0
    total_test_correct_col = np.zeros(trait_num)
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            texts, labels = batch
            
            outputs = model(texts)
            outputs = outputs.squeeze()

            # Convert probabilities to predicted classes based on threshold
            preds = (outputs >= 0.5).long() 
            correct = (preds == labels).cpu().numpy()
            total_test_correct += correct.sum()
            total_test_correct_col += correct.sum(axis=0)
            total_test += correct.size
    
    avg_test_accuracy = total_test_correct / total_test
    avg_test_accuracy_col = total_test_correct_col / (total_test/trait_num)

    print(f"### Conducting LSTM model test (with path {model_save_path}): ")
    print(f"Test Accuracy: {avg_test_accuracy}, Test Accuracy per trait: {avg_test_accuracy_col}")


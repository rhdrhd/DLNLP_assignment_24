# Standard library imports
import os
import random

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
import wandb

# Local module imports
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
        embedded_sentences = embedded_sentences[:,:8]
        _, (sentence_hidden, _) = self.sentence_lstm(embedded_sentences)
        
        if self.sentence_lstm.bidirectional:
            sentence_hidden = torch.cat((sentence_hidden[-2], sentence_hidden[-1]), dim=1)
        else:
            sentence_hidden = sentence_hidden[-1]
        
        sentence_hidden = sentence_hidden.view(batch_size, num_sentences, -1)
        sentence_hidden = sentence_hidden[:,:12]
        _, (document_hidden, _) = self.document_lstm(sentence_hidden)
        
        if self.document_lstm.bidirectional:
            document_hidden = torch.cat((document_hidden[-2], document_hidden[-1]), dim=1)
        else:
            document_hidden = document_hidden[-1]
        
        document_hidden = self.dropout(document_hidden)
        
        output = self.fc(document_hidden)
        output = torch.sigmoid(output)

        return output

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, output_dim):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        # BERT's pooled output is of the hidden size of the model
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output for classification
        pooled_output = outputs.pooler_output
        probs = self.fc(pooled_output)
        preds = torch.sigmoid(probs)
        return preds

def train_bert():
    data = pd.read_csv('data_preprocess/essays.csv', encoding='mac_roman')
    raw_texts = data['TEXT'].tolist()

    # Model config
    max_length = 512
    model_name = 'prajjwal1/bert-small'
    trait_number = 5
    batch_size = 16
    epochs = 1

    # Prepare data for BERT
    input_ids, attention_masks = prepare_data_for_bert(model_name,raw_texts, max_length)

    if trait_number == 1:
        labels = data[['cEXT']].replace({'y': 1, 'n': 0}).values
        labels = torch.tensor(labels.tolist(), dtype=torch.float32).squeeze()
    else:
        labels = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].replace({'y': 1, 'n': 0}).values
        labels = torch.tensor(labels.tolist(), dtype=torch.float32)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    bert_model = BertModel.from_pretrained(model_name)
    model = BERTClassifier(bert_model, output_dim=trait_number)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)

    
    wandb_on = False
    if wandb_on:
        wandb.init(
            project="nlp-personality-prediction",
            config={
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_length": max_length,
                "model_name": model_name,
                "optimizer": optimizer,
                "criterion": criterion,
                "trait_number": trait_number,
            },
            notes="None",
            )
    
    # Training
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch

            model.zero_grad()
            outputs = model(b_input_ids, b_attention_mask)
            outputs = outputs.squeeze()
            #print(outputs)
            loss = criterion(outputs, b_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_train_loss / len(train_dataloader)}")

        # Validation phase
        model.eval()
        total_eval_correct = 0
        total_eval = 0
        total_eval_loss = 0
        total_eval_correct_col = np.zeros(5)
        with torch.no_grad():
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_attention_mask, b_labels = batch
                
                outputs = model(b_input_ids, b_attention_mask)
                outputs = outputs.squeeze()

                loss = criterion(outputs, b_labels)
 
                total_eval_loss += loss.item()

                # Convert probabilities to predicted classes based on threshold
                preds = (outputs >= 0.5).long() 
                correct = (preds == b_labels).cpu().numpy()
                total_eval_correct += correct.sum()
                total_eval_correct_col += correct.sum(axis=0)
                total_eval += correct.size
                
        avg_val_accuracy = total_eval_correct / total_eval
        avg_val_accuracy_col = total_eval_correct_col / (total_eval/trait_number)
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}, Validation Accuracy per trait: {avg_val_accuracy_col}")

        # Test phase
        model.eval()
        total_test_correct = 0
        total_test = 0
        total_test_correct_col = np.zeros(5)
        with torch.no_grad():
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_attention_mask, b_labels = batch
                
                outputs = model(b_input_ids, b_attention_mask)
                outputs = outputs.squeeze()

                # Convert probabilities to predicted classes based on threshold
                preds = (outputs >= 0.5).long() 
                correct = (preds == b_labels).cpu().numpy()
                total_test_correct += correct.sum()
                total_test_correct_col += correct.sum(axis=0)
                total_test += correct.size
        
        avg_test_accuracy = total_test_correct / total_test
        avg_test_accuracy_col = total_test_correct_col / (total_test/trait_number)

        print(f"Test Accuracy: {avg_test_accuracy}, Test Accuracy per trait: {avg_test_accuracy_col}")

def run_bert_sweep():
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
                'values': [8, 16]
            },
            'epochs': {
                'values': [30, 50]  # Shorter for quick sweeps
            },
            'model_name': {
                'values': ['prajjwal1/bert-small','google-bert/bert-base-uncased','distilbert/distilbert-base-uncased']
            },
            'max_length': {
                'values': [128, 256, 512]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="nlp-personality-prediction")
    wandb.agent(sweep_id, bert_sweep_setup, count=50)

def bert_sweep_setup():


    wandb.init(project="nlp-personality-prediction")

    data = pd.read_csv('data_preprocess/essays.csv', encoding='mac_roman')
    raw_texts = data['TEXT'].tolist()

    # Model config
    max_length = wandb.config.max_length
    model_name = wandb.config.model_name
    trait_number = 5
    batch_size = wandb.config.batch_size  # Use wandb config
    epochs = wandb.config.epochs  # Use wandb config

    # Prepare data for BERT
    input_ids, attention_masks = prepare_data_for_bert(model_name,raw_texts, max_length)

    if trait_number == 1:
        labels = data[['cEXT']].replace({'y': 1, 'n': 0}).values
        labels = torch.tensor(labels.tolist(), dtype=torch.float32).squeeze()
    else:
        labels = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].replace({'y': 1, 'n': 0}).values
        labels = torch.tensor(labels.tolist(), dtype=torch.float32)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    bert_model = BertModel.from_pretrained(model_name)
    model = BERTClassifier(bert_model, output_dim=trait_number)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)

    
    # Training
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch

            model.zero_grad()
            outputs = model(b_input_ids, b_attention_mask)
            outputs = outputs.squeeze()
            #print(outputs)
            loss = criterion(outputs, b_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
        # Log training loss to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": total_train_loss / len(train_dataloader)})

        # Validation phase
        model.eval()
        total_eval_correct = 0
        total_eval = 0
        total_eval_loss = 0
        total_eval_correct_col = np.zeros(5)
        with torch.no_grad():
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_attention_mask, b_labels = batch
                
                outputs = model(b_input_ids, b_attention_mask)
                outputs = outputs.squeeze()

                loss = criterion(outputs, b_labels)
 
                total_eval_loss += loss.item()

                # Convert probabilities to predicted classes based on threshold
                preds = (outputs >= 0.5).long() 
                correct = (preds == b_labels).cpu().numpy()
                total_eval_correct += correct.sum()
                total_eval_correct_col += correct.sum(axis=0)
                total_eval += correct.size
                
        avg_val_accuracy = total_eval_correct / total_eval
        avg_val_accuracy_col = total_eval_correct_col / (total_eval/trait_number)
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Log validation metrics to wandb
        wandb.log({"val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy})

    # Finish the wandb run
    wandb.finish()

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
                'values': [30,50]  # Shorter for quick sweeps
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

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Define the model
    sentence_hidden_dim = 256
    document_hidden_dim = 512
    trait_num = 5
    num_layers = 1
    batch_size = 16
    learning_rate = 0.005
    epochs = 20

    train_dataset,  val_dataset, test_dataset, vocab = preprocess_data()
    embedding_matrix = apply_word_embeddings(vocab)

    model = Customized_LSTM(embedding_matrix, sentence_hidden_dim=128, document_hidden_dim=128, output_dim=trait_num, num_layers=2, bidirectional=True, dropout=0.1)
    model.to(device)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
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

    if False:
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

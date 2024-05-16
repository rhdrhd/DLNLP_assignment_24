import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertModel
import wandb

from data_preprocess.preprocess import prepare_data_for_bert

# Function to set seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the BERT-based classifier model
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

# Function to train the BERT model
def train_bert():
    # Model config
    max_length = 512
    model_name = 'prajjwal1/bert-small'
    trait_number = 5
    batch_size = 8
    epochs = 50

    # Prepare data for BERT
    train_dataset, val_dataset, test_dataset = prepare_data_for_bert(model_name, max_length, trait_number)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Load BERT model
    bert_model = BertModel.from_pretrained(model_name)
    model = BERTClassifier(bert_model, output_dim=trait_number)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)

    # Initialize Weights & Biases (wandb) logging
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

    # Initialize best validation loss
    best_val_loss = float('inf')
    # Define model saving path
    model_save_path = 'best_model.pth'

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch

            model.zero_grad()
            outputs = model(b_input_ids, b_attention_mask)
            outputs = outputs.squeeze()
            loss = criterion(outputs, b_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")
        
        if wandb_on:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})
        
        # Validation phase
        model.eval()
        total_eval_correct = 0
        total_eval = 0
        total_eval_loss = 0
        total_eval_correct_col = np.zeros(trait_number)
        with torch.no_grad():
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_attention_mask, b_labels = batch
                
                outputs = model(b_input_ids, b_attention_mask)
                outputs = outputs.squeeze()

                loss = criterion(outputs, b_labels)
                total_eval_loss += loss.item()

                preds = (outputs >= 0.5).long() 
                correct = (preds == b_labels).cpu().numpy()
                total_eval_correct += correct.sum()
                total_eval_correct_col += correct.sum(axis=0)
                total_eval += correct.size
                
        avg_val_accuracy = total_eval_correct / total_eval
        avg_val_accuracy_col = total_eval_correct_col / (total_eval / trait_number)
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}, Validation Accuracy per trait: {avg_val_accuracy_col}")
        
        if wandb_on:
            wandb.log({"val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy})
        
        # Check if this is the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with validation loss: {best_val_loss}, and accuracy {avg_val_accuracy}")

# Function to run a hyperparameter sweep with Weights & Biases
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
                'values': ['prajjwal1/bert-small', 'google-bert/bert-base-uncased', 'distilbert/distilbert-base-uncased']
            },
            'max_length': {
                'values': [128, 256, 512]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="nlp-personality-prediction")
    wandb.agent(sweep_id, bert_sweep_setup, count=50)

# Function to set up BERT model training for hyperparameter sweep
def bert_sweep_setup():
    wandb.init(project="nlp-personality-prediction")

    # Load dataset
    data = pd.read_csv('data_preprocess/essays.csv', encoding='mac_roman')
    raw_texts = data['TEXT'].tolist()

    # Model config from wandb
    max_length = wandb.config.max_length
    model_name = wandb.config.model_name
    trait_number = 5
    batch_size = wandb.config.batch_size  # Use wandb config
    epochs = wandb.config.epochs  # Use wandb config
    learning_rate = wandb.config.learning_rate  # Use wandb config

    # Prepare data for BERT
    input_ids, attention_masks = prepare_data_for_bert(model_name, raw_texts, max_length)

    if trait_number == 1:
        labels = data[['cEXT']].replace({'y': 1, 'n': 0}).values
        labels = torch.tensor(labels.tolist(), dtype=torch.float32).squeeze()
    else:
        labels = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].replace({'y': 1, 'n': 0}).values
        labels = torch.tensor(labels.tolist(), dtype=torch.float32)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Split dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Load BERT model
    bert_model = BertModel.from_pretrained(model_name)
    model = BERTClassifier(bert_model, output_dim=trait_number)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop for hyperparameter sweep
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch

            model.zero_grad()
            outputs = model(b_input_ids, b_attention_mask)
            outputs = outputs.squeeze()
            loss = criterion(outputs, b_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Validation phase
        model.eval()
        total_eval_correct = 0
        total_eval = 0
        total_eval_loss = 0
        total_eval_correct_col = np.zeros(trait_number)
        with torch.no_grad():
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_attention_mask, b_labels = batch
                
                outputs = model(b_input_ids, b_attention_mask)
                outputs = outputs.squeeze()

                loss = criterion(outputs, b_labels)
                total_eval_loss += loss.item()

                preds = (outputs >= 0.5).long()
                correct = (preds == b_labels).cpu().numpy()
                total_eval_correct += correct.sum()
                total_eval_correct_col += correct.sum(axis=0)
                total_eval += correct.size
                
        avg_val_accuracy = total_eval_correct / total_eval
        avg_val_accuracy_col = total_eval_correct_col / (total_eval / trait_number)
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        wandb.log({"val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy})

    # Finish the wandb run
    wandb.finish()

# Function to test the trained BERT model
def test_bert(model_save_path):
    # Load dataset
    # Model config
    max_length = 512
    model_name = 'prajjwal1/bert-small'
    trait_number = 5
    batch_size = 8

    # Prepare data for BERT
    _, _, test_dataset = prepare_data_for_bert(model_name, max_length, trait_number)

    # Create dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # Load BERT model
    bert_model = BertModel.from_pretrained(model_name)
    model = BERTClassifier(bert_model, output_dim=trait_number)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)

    # Test phase
    model.eval()
    total_test_correct = 0
    total_test = 0
    total_test_correct_col = np.zeros(trait_number)
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch
            
            outputs = model(b_input_ids, b_attention_mask)
            outputs = outputs.squeeze()

            preds = (outputs >= 0.5).long()
            correct = (preds == b_labels).cpu().numpy()
            total_test_correct += correct.sum()
            total_test_correct_col += correct.sum(axis=0)
            total_test += correct.size
    
    avg_test_accuracy = total_test_correct / total_test
    avg_test_accuracy_col = total_test_correct_col / (total_test / trait_number)
    print(f"### Conducting BERT model {model_name} testing:")
    print(f"Test Accuracy: {avg_test_accuracy}, Test Accuracy per trait: {avg_test_accuracy_col}")

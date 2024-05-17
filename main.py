import os

# Unset LD_LIBRARY_PATH to avoid cuDNN conflicts
if 'LD_LIBRARY_PATH' in os.environ:
    del os.environ['LD_LIBRARY_PATH']

# Set CUBLAS_WORKSPACE_CONFIG to ensure deterministic behavior with CuBLAS
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Import modules after setting the seed to ensure reproducibility in those imports
from model_training.LSTM import *
from data_preprocess.preprocess import *
from model_training.chatgpt_prompt import *
from model_training.bag_of_word_logistic_regression import *
from model_training.BERT_finetune import *

train_mode = False
openai_api = "sk-proj-EwRsyEt1chy2ZSdlumsRT3BlbkFJqgPaeKVzUDyOawgAUFKH" #Paste OpenAI api key here

# Download weights from google drive
download_files()
download_word2vector()
print("\n")

# Test Bag-of-word with Logistic Regression
test_bag_of_words()
print("\n")

# Test Customized LSTM Model
if train_mode: train_lstm(epoch=50)
test_lstm("model_weights/best_model_lstm_loss.pth")
test_lstm("model_weights/best_model_lstm_acc.pth")
print("\n")

# Test Finetuned BERT Model
if train_mode: train_bert(epoch=50)
test_bert("model_weights/best_model_bert_loss.pth")
test_bert("model_weights/best_model_bert_acc.pth")
print("\n")

# Test ChatGPT Prompt Learning
if openai_api:
    test_chatgpt(fraction=0.001,example_count=1,iterations=1,api_key=openai_api)
else:
    print("please input your openai apikey for testing")


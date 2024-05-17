from data_preprocess.preprocess import download_files,download_word2vector
from model_training.bag_of_word_logistic_regression import test_bag_of_words
from model_training.LSTM import train_lstm, test_lstm
from model_training.BERT_finetune import train_bert, test_bert
from model_training.chatgpt_prompt import test_chatgpt


train_mode = False
openai_api = "" # Paste OpenAI api key here

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
    print("### Please input your openai apikey for testing")


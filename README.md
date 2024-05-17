# ELEC0141_NLP_Project
This repo presents the development and evaluation of various NLP machine learning models for the task of personality trait prediction over Essays Dataset. The dataset is a collection of essays written by students and annotated with the Big Five personality traits. The project is part of the ELEC0141 Natural Language Processing course at University College London.

Models used in the project are:
1. Bag of Words Logistic Regression
2. Customized hierarchical LSTM
3. BERT Fine-tuning
4. ChatGPT Prompt

## Repository Structure
The current project structure is shown below
```
├── data_preprocess
│   ├── init.py
│   ├── preprocess.py
│   ├── EDA.ipynb
|   ├── essay.csv
|   ├── word2vector.bin
├── model_training
│   ├── init.py
│   ├── bag_of_word_logistic_regression.py
│   ├── BERT_finetune.py
│   ├── chatgpt_prompt.py
│   ├── llama_prompt.py
│   ├── LSTM.py
├── model_weights
│   ├── best_model_bert_acc.pth
│   ├── best_model_bert_loss.pth
│   ├── best_model_lstm_acc.pth
│   ├── best_model_lstm_loss.pth
├── environment.yml
├── requirements.txt
├── README.md
└── main.py
```

## How to start
1. Create a new conda environment from environment.yml file.
```
conda env create -f environment.yml
```
2. Activate this conda virtual environment. 
```
conda activate nlp-project
```
3. Run main.py if all the dependencies required for the current project are already installed. 
```
python main.py
```
### Notes
Model weights trained in project and Word2Vector model weights are downloaded automatically.

In case the download does not work, please manually download the pretrained weights from all these [link1](https://drive.google.com/file/d/1_RwOEto-l-ra2ApAFXvi-KFO4-o_ZnXk/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) [link2](https://drive.google.com/file/d/1XScpzWarASBsqiCqkKtGmsjwr26b6mKP/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) [link3](https://drive.google.com/file/d/1u-UZMd5hwVPIR_sTGmHzldpeuRfuvLqY/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) [link4](https://drive.google.com/file/d/1nBIYwQf4iAwEYFYqqdyKgnzQCjnmEhgd/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g). And put them in model_weights folder.

Word2Vector Model from google at [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) and put it in data_preprocess folder

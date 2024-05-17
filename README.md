# ELEC0141_NLP_Project_SN23039407
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
|   ├── images
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
conda activate nlp-final
```
3. Run main.py if all the dependencies required for the current project are already installed. **In main.py file, train mode is by default set as False, PLease adjust to True to start Training**
```
python main.py
```
## Notes
Model weights trained in project and Word2Vector model weights are downloaded automatically. You can also obtain the same weights by turning on train_mode.

In case the download does not work, please manually download the pretrained weights from all these [link1](https://drive.google.com/file/d/1_RwOEto-l-ra2ApAFXvi-KFO4-o_ZnXk/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) [link2](https://drive.google.com/file/d/1XScpzWarASBsqiCqkKtGmsjwr26b6mKP/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) [link3](https://drive.google.com/file/d/1u-UZMd5hwVPIR_sTGmHzldpeuRfuvLqY/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) [link4](https://drive.google.com/file/d/1nBIYwQf4iAwEYFYqqdyKgnzQCjnmEhgd/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g), and put them in model_weights folder.

Word2Vector Model from google at [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) and put it in data_preprocess folder


## Accuracy Metrics for All Models Tested

| Model                                | Overall | cEXT         | cNEU         | cAGR          | cCON         | cOPN         |
|--------------------------------------|---------|--------------|--------------|---------------|--------------|--------------|
| GPT-4o                               | 54.04   | **56.64**    | 50.41        | 50.68         | 54.20        | 58.27        |
| GPT-3.5-turbo (zero-shot)            | 54.00   | 56.08        | 49.46        | 54.73         | 51.35        | 58.38        |
| GPT-3.5-turbo (1-shot)               | 55.26   | 55.82        | 53.20        | 55.73         | 53.56        | 57.98        |
| GPT-3.5-turbo (16-shot)              | 56.76   | 54.12        | 51.18        | **58.62**     | 61.65        | 58.40        |
| BERT-small (loss)                    | 51.79   | 45.38        | 51.09        | 51.63         | 52.45        | 58.42        |
| BERT-small (acc)                     | **56.68** | 49.73        | 52.99        | 57.61         | **62.23**    | **60.87**    |
| Bag-of-Words Logistic Regression     | 54.78   | 51.82        | **55.06**    | 55.06         | 54.05        | 57.89        |
| LSTM (acc)                           | 51.82   | 53.41        | 51.70        | 52.84         | 52.84        | 48.30        |
| LSTM (loss)                          | 47.61   | 45.45        | 51.99        | 45.17         | 47.73        | 47.73        |
| GPT-3.5-turbo (zero-shot) (by Ji et al.) | **57.40** | 60.90        | 56.00        | 50.80         | **58.90**    | 60.50        |
| RNN (by Ji et al.)                   | 47.30   | 43.30        | **60.00**    | 33.30         | 43.30        | 56.70        |
| RoBERTa (by Ji et al.)               | 55.30   | **63.30**    | 53.30        | **53.30**     | 40.00        | **66.70**    |

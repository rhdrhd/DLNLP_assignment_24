# ELEC0141_NLP_Project
This repo presents the development and evaluation of a customized LSTM NLP machine learning models for the task of personality trait prediction over Essays Dataset. The dataset is a collection of essays written by students and annotated with the Big Five personality traits. The project is part of the ELEC0141 Natural Language Processing course at University College London.

## Repository Structure
The current project structure is shown below
```
├── data_preprocess
│   ├── init.py
│   ├── preprocess.py
│   ├── draft.ipynb
|   ├── essay.csv
├── model_training
│   ├── init.py
│   ├── train.py
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
3. Run main.py if all the dependencies required for the current project are already installed. 
```
python main.py
```
### Notes
To run Customized_LSTM Model, first download the Word2Vector Model from google at [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=share_link&resourcekey=0-wjGZdNAUop6WykTtMip30g) and put it in data_preprocess folder

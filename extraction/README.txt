In order to extract, clean the data and create a cross validation dataset, please execute the scripts in this order : 

1. extract_data.py
2. process_data.py
3. cross_validation.py

In order to prepare data for differents models we have tested, please execute one of this script (or all if need all models preparation) :

- models_preprocess/cnn_preprocess (pre process data for cnn and save them in "project_directory/data/models_prepared/cnn_formated/"
- models_preprocess/rnn_preprocess (pre process data for rnn and save them in "project_directory/data/models_prepared/rnn_formated/"
- models_preprocess/random_forest_preprocess (pre process data for random forest and save them in "project_directory/data/models_prepared/random_forest_formated/"
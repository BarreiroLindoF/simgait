Before executing any extraction script, make sure to have a directory called "data" in the root of the project. Inside this directory make sure to have a directory called "raw_structured".
"raw_structured" directory may have four subdirectory :
- CP_Gait_1.0
- Healthy
- ITW_RETRACTION_SOL_TRICEPS_Gait_1.0
- ITW_RETRACTION_TRICEPS_Gait_1.0

containing .c3d files in them.

In order to extract, clean the data and create a cross validation dataset, please execute the scripts in this order : 

1. extract_data.py
2. process_data.py
3. cross_validation.py

In order to prepare data for differents models we have tested, please execute one of this script (or all if need all models preparation) :

- models_preprocess/cnn_preprocess (pre process data for cnn and save them in "project_directory/data/models_prepared/cnn_formated/"
- models_preprocess/rnn_preprocess (pre process data for rnn and save them in "project_directory/data/models_prepared/rnn_formated/"
- models_preprocess/random_forest_preprocess (pre process data for random forest and save them in "project_directory/data/models_prepared/random_forest_formated/"
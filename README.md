# Radiology Report Generation
# Resources
1. OpenFlamingo Repository (https://github.com/mlfoundations/open_flamingo): This repository was referenced to mimic the process of training a Flamingo model. Scripts to train are present in the train.py and train_utils.py folder.
2. MedFlamingo Model (https://github.com/snap-stanford/med-flamingo): This model was used as the base model to finetune as it's already been trained on medical data
3. Flamingo-CXR Paper (https://www.nature.com/articles/s41591-024-03302-1): This paper was used as a reference to finetune the Flamingo model for radiology report generation.Details on finetuning the model on the dataset can be found in the paper

# Setup Details
1. Install the required library packages using the requirements.txt file with the command: "pip install -r requirements.txt"
2. Run the script using the command: "python main.py"

# Guide to files in source folder
1. _main.py_: Orchestrate the pipeline: Loading model, preprocessing data and finetuning the model
2. _model.py_: Defines a model class and is used to load the MedFlamingo model
3. _preprocessing.py_: Defines a class that is used to preprocess the data (Read the data, load the images and split the data into training, testing and validation datasets)
4. _dataset.py_: Defines the Torch Dataset class that is used to preprocess the images (Apply image transformations, generate embeddings and normalize embeddings) and return data in the format expected by the DataLoader
5. _finetune.py_: Defines method to preprocess text and generate input ids and attention masks and defines a method to finetune the data

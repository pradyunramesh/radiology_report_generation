import pandas as pd
import numpy as np
from PIL import Image
from logger import logger
from sklearn.model_selection import train_test_split

class CheXpertPreprocessing():
    '''Class perfroms preprocessing on a dataset given the path to the CSV file of the dataset'''
    def __init__(self, path):
        self.path = path

    def read_data(self):
        '''Read data from path and filter the dataset'''
        logger.info(f"Reading data from path: {self.path}")
        chexpert_plus = pd.read_csv(self.path)
        chexpert_plus = chexpert_plus[chexpert_plus['section_impression'].notna()]
        chexpert_plus = chexpert_plus[chexpert_plus['frontal_lateral'] == 'Frontal']
        logger.info("Successfully read data!")
        return chexpert_plus

    def generate_embeddings(self, data):
        '''Generate embeddings for the dataset'''
        logger.info("Generating embeddings for the dataset")
        images = []
        for filename in data['path_to_image']:
            if filename.startswith('train'):
                filename = '/data2/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/' + filename
                images.append(Image.open(filename))
            elif filename.startswith('valid'):
                filename = "/data2/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/" + filename
                images.append(Image.open(filename))
        logger.info("Successfully generated embeddings!")
        return images

    def split_and_save_data(self, data, images):
        '''Split the data into training, validation and test sets and save to paths'''
        logger.info("Splitting data into training, validation and test sets")
        train_chexpert, temp_chexpert = train_test_split(data, test_size=0.3, random_state=42)
        valid_chexpert, test_chexpert = train_test_split(temp_chexpert, test_size=0.5, random_state=42)
        train_chexpert = train_chexpert.reset_index(drop=True)
        valid_chexpert = valid_chexpert.reset_index(drop=True)
        test_chexpert = test_chexpert.reset_index(drop=True)
        train_embeddings, temp_embeddings = train_test_split(images, test_size=0.3, random_state=42)
        valid_embeddings, test_embeddings = train_test_split(temp_embeddings, test_size=0.5, random_state=42)
        logger.info("Successfully split data into training, validation and test sets")
        return train_chexpert, valid_chexpert, test_chexpert, train_embeddings, valid_embeddings, test_embeddings

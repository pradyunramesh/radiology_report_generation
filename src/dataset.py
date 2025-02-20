import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CheXpertDataset(Dataset):
    '''This dataset is compatible with the CheXpert Dataset
    Arguments:
    data: Images/Chest X-rays loaded from the CheXpert dataset
    labels: Radiology Reports (Impressions Section)
    image_processor: Image Processor to process and conver images to tensors
    tokenizer: Tokenizer to encode the radiology reports'''

    def __init__(self, data, input_ids, attention_masks, image_processor):
        '''Initialize the dataset'''
        self.data = data
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.image_processor = image_processor
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomRotation(15)
                                             ])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        '''Return the length of the dataset'''
        return len(self.data)

    def __getitem__(self, idx):
        '''Get the image data and text data for the given index'''
        image_data = self.preprocess_images([self.data[idx]]) #Preprocess the image
        input_ids = self.input_ids[idx]
        attention_masks = self.attention_masks[idx]
        return image_data, input_ids, attention_masks

    def preprocess_images(self, images):
        '''Preprocess images and convert them to tensors'''
        transformed_images = [self.transform(image) for image in images] #Apply image transformations
        processed_images = [self.image_processor(sample).unsqueeze(0) for sample in transformed_images] #Process each image and add a batched dimension
        normalized_images = [self.normalize(image) for image in processed_images] #Normalize the images
        final_images = torch.cat(normalized_images, dim = 0) #Concatenate along the first axis
        return final_images

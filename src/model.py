import torch
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from logger import logger

class MedFlamingoModel():
    '''Class to load the Med-Flamingo model and associated processors/tokenizers'''
    def __init__(self, lang_encoder_path, vision_encoder_path):
        '''Initialize configurations for the model'''
        self.lang_encoder_path = lang_encoder_path
        self.vision_encoder_path = vision_encoder_path

    def load_model(self):
        '''Method to load the Med-Flamingo model'''
        logger.info("Loading the Med-Flamingo model")
        #Set up resources
        torch.cuda.empty_cache()
        accelerator = Accelerator()
        device = accelerator.device
        #Load model and processors
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path = self.vision_encoder_path,
            clip_vision_encoder_pretrained = "openai",
            lang_encoder_path = self.lang_encoder_path,
            tokenizer_path = self.lang_encoder_path,
            cross_attn_every_n_layers = 4,
            freeze_lm_embeddings = True
        )
        #Download the med-flamingo checkpoint
        checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt") 
        model.load_state_dict(torch.load(checkpoint_path, map_location = device), strict=False)
        logger.info("Model loaded successfully")
        return model, image_processor, tokenizer

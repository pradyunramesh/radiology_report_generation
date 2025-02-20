from model import MedFlamingoModel
from preprocessing import CheXpertPreprocessing
from finetune import finetune_data, load_data_and_embeddings
from logger import logger

def main():
    '''Main function to load and finetune the med-flamingo model'''
    #Load the model
    model_obj = MedFlamingoModel(lang_encoder_path = "meta-llama/Llama-2-7b-hf", vision_encoder_path = "ViT-L-14")
    model, image_processor, tokenizer = model_obj.load_model()
    #Preprocess the data
    path = '/data2/chexpert/chexpertplus/df_chexpert_plus_240401.csv'
    chexpert_obj = CheXpertPreprocessing(path)
    data = chexpert_obj.read_data()
    images = chexpert_obj.generate_embeddings(data)
    train_chexpert, valid_chexpert, test_chexpert, train_embeddings, valid_embeddings, test_embeddings = chexpert_obj.split_and_save_data(data, images)
    #Finetune the model
    finetune_data(model, image_processor, tokenizer, train_chexpert, train_embeddings)

if __name__ == "__main__":
    main()
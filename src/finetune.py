import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import CheXpertDataset
from logger import logger
from einops import rearrange
from accelerate import Accelerator

def load_data_and_embeddings():
    '''Method to load the CheXpert dataset and embeddings'''
    logger.info("Loading data and embeddings!")
    train_chexpert = pd.read_csv('/data/mchome/pr2762/radiology_data/train_chexpert_plus.csv')
    val_chexpert = pd.read_csv('/data/mchome/pr2762/radiology_data/valid_chexpert_plus.csv')
    test_chexpert = pd.read_csv('/data/mchome/pr2762/radiology_data/test_chexpert_plus.csv')
    logger.info("Data and embeddings loaded successfully!")
    return train_chexpert, val_chexpert, test_chexpert

def process_text(tokenizer, text_data):
    '''Method to process the text data'''
    logger.info("Processing text data!")
    tokenizer.padding_side = "right"
    processed_text = [(f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in text_data]
    encoded_text = tokenizer(processed_text, padding = "longest", max_length = 1024, truncation = "only_first") #Only text following the image is truncated
    input_ids, attention_masks = encoded_text["input_ids"], encoded_text["attention_mask"]
    logger.info("Text data processed successfully!")
    return torch.tensor(input_ids), torch.tensor(attention_masks)

def finetune_data(model, image_processor, tokenizer, train_chexpert, train_embeddings):
    '''Method to finetune the model on the CheXpert dataset'''
    logger.info("Finetuning the model!")
    input_ids, attention_masks = process_text(tokenizer, train_chexpert['section_impression'])
    train_dataset = CheXpertDataset(train_embeddings, input_ids, attention_masks, image_processor)
    training_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    image_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    #endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    accelerator = Accelerator()
    torch.cuda.empty_cache()
    device = accelerator.device
    optimizer = torch.optim.AdamW(params = filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)
    model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    for param in model.lang_encoder.parameters():
        param.requires_grad = False
    model.train()

    torch.cuda.empty_cache()
    epochs = 1
    for epoch in range(epochs):
        for batch_idx, (images, input_ids, attention_masks) in enumerate(training_dataloader):
            #Load Text and Image Data
            images = images.to(device, non_blocking = True)
            images = rearrange(images, "(b t) f c h w -> b t f c h w", t=1) #Rearrange images to add dimensions
            input_ids = input_ids.to(device, non_blocking = True)
            attention_masks = attention_masks.to(device, non_blocking = True)
            #Evaluate labels for the model
            labels = input_ids.clone()
            labels[labels == image_token_id] = -100 #Ignores tokens during loss computation
            labels[labels == tokenizer.pad_token_id] = -100
            labels = labels.to(device)
            #Compute the model loss
            model_loss = model(images, input_ids, attention_masks, labels = labels)[0] #Minimize the sum of image-to-text loss and text-to-image loss
            #Perform backpropagation
            optimizer.zero_grad()
            accelerator.backward(model_loss)
            optimizer.step()
        logger.info(f"Epoch: {epoch}, Loss: {model_loss.item()}")

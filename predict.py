from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences 

import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification, AdamW
from transformers.optimization import WarmupLinearSchedule 
from argparse import ArgumentParser
from utils import build_segment_ids, MAX_LEN, ROBERTA_MAX_LEN

import re 
import os
import json
import logging
from tqdm import tqdm 
from pprint import pformat

logger = logging.getLogger(__file__)
PREDICTION_BATCH_SIZE = 256


def get_data_loader(args, input_data, tokenizer):

    cache_fp = args.data_path[:args.data_path.rfind('.')] + '_cache'
    if os.path.isfile(cache_fp) and False: 
        logger.info("Loading tokenized data from cache...")
        input_ids = torch.load(cache_fp)

    else:
    # format sentence with BERT special tokens
        sentences = []
        for sample in input_data: 
            sentence = '[CLS] {} [SEP] {} [SEP]'.format(sample['p'], sample['r'])
            sentences.append(sentence)

        # encode to BERT tokens

        logger.info("Tokenize input data...")
        input_ids = [tokenizer.encode(sentence) for sentence in sentences]
        torch.save(input_ids, cache_fp)

    # dialogue ids for tracking purposes 
    dialogue_idx = [[sample['idx']]*MAX_LEN for sample in input_data] 

    prev_len = len(input_ids)
    # remove sequences longer than MAX_LEN
    for i in reversed(range(len(input_ids))): 
        if len(input_ids[i]) > MAX_LEN: 
            input_ids.pop(i)
            dialogue_idx.pop(i) 

    logger.info("{} samples were removed as it exceeded the maximum sequence length of {}.".format(prev_len - len(input_ids), MAX_LEN))

    # pad to MAX_LEN
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, truncating="post", padding="post")

    # get attention mask
    attention_masks = [] 
    for seq in input_ids: 
        attention_masks.append([float(i>0) for i in seq])

    # get segment information
    segment_ids = build_segment_ids(input_ids)

    # wrap as tensors
    dialogue_idx = torch.tensor(dialogue_idx)
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    segment_ids = torch.tensor(segment_ids)

    # create dataloader
    prediction_data = TensorDataset(dialogue_idx, input_ids, attention_masks, segment_ids)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler = prediction_sampler, batch_size = PREDICTION_BATCH_SIZE)

    return prediction_dataloader

def predict_label(args, model, prediction_dataloader, data_to_predict): 
  
    # deactivate dropout, etc. 
    if torch.cuda.is_available(): 
        model.cuda()
        model.eval() 

    # for convenient retrieval 
    idx_to_data = {sample['idx']: {'p': sample['p'], 'r': sample['r']} for sample in data_to_predict}

    predictions = [] 
    for batch in tqdm(prediction_dataloader): 
    # add batch to GPU
        if torch.cuda.is_available(): 
            batch = tuple(t.to(args.device).to(torch.int64) for t in batch)

        #unpack input 
        b_dialogue_idx, b_input_ids, b_attention_masks, b_segment_ids = batch

        # don't store gradients
        with torch.no_grad(): 
            # forward pass
            logits = model(b_input_ids, b_attention_masks, b_segment_ids)

            softmax_logits = torch.nn.functional.softmax(logits[0], dim=1).cpu().numpy()

            # labels = softmax_logits[:,1] > args.threshold
            # labels = labels.astype(int).flatten()

            # TODO: modify how you want to store your prediction results
            for dialogue_idx, input_id, softmax_logit in zip(b_dialogue_idx, b_input_ids, softmax_logits): 
                idx = int(dialogue_idx[0].cpu().numpy())
                result = {'idx': idx, 'p': idx_to_data[idx]['p'], 'r': idx_to_data[idx]['r'], 'confidence': {'yesand': round(softmax_logit[1]*100, 2) , 'nonyesand': round(softmax_logit[0]*100, 2)}}
                predictions.append(result)

    return predictions

def decode_input_id(input_id, tokenizer): 
    decoded_text = tokenizer.decode(input_id.to('cpu').numpy(), clean_up_tokenization_spaces=True)
    decoded_text_split = decoded_text[:2]
    decoded_text_split = [re.sub('(\[CLS\]|[-*+_])', '', split).strip() for split in decoded_text_split]
    decoded_text_split = [re.sub('\ \ ', ' ', split).strip() for split in decoded_text_split]

    return decoded_text_split[:2]

def get_cornell_data(data_path): 
    # Load Cornell movie dialogue corpus and and format them in to a list  
    # Each sample has the following format: {'id': int, 'p': str, 'r': str}

    with open(data_path, 'r') as f: 
        data_to_predict = json.load(f)

    return data_to_predict

def predict(): 
    """Determine which are yes-ands are not from a given dialogue data set with a finetuned BERT yes-and classifier"""
    parser = ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased", help="Provide pretrained model type that is consisten with BERT model that was fine-tuned.")
    parser.add_argument("--model_checkpoint", default="runs/yesand_cornell_bert_base_iter1", help="Provide a directory for a pretrained BERT model.")
    parser.add_argument("--data_path", default="data/reformatted_cornell.json", help="Provide a datapath for which predictions will be made.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--predictions_folder", default="data/", help="Provide a folderpath for which predictions will be saved to.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: {}".format(pformat(args)))

    logger.info("Loading model and tokenizer.")
    model = BertForSequenceClassification.from_pretrained(args.model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    logger.info("Loading data to predict: {}".format(args.data_path))
    data_to_predict = get_cornell_data(args.data_path)

    logger.info("Building data loader...")
    prediction_dataloader = get_data_loader(args, data_to_predict, tokenizer)

    logger.info("Making predictions...")
    predictions = predict_label(args, model, prediction_dataloader, data_to_predict)
    logger.info("Predictions complete for {} dialogue pairs. ".format(len(predictions)))


    logger.info("Saving predictions...")
    predictions_fp = args.predictions_folder + 'predictions_{}.json'.format(re.sub('runs/', '', args.model_checkpoint))
    with open(predictions_fp, 'w') as f: 
        json.dump(predictions, f, indent=4)
    logger.info("Predictions saved to {}.".format(predictions_fp))

if __name__ == "__main__": 
    predict() 
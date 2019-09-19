from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences 

import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification, AdamW
from pytorch_transformers.optimization import WarmupLinearSchedule 
from argparse import ArgumentParser
from utils import build_segment_ids 

import re 
import os
import json
import logging
from tqdm import tqdm 
from pprint import pformat

logger = logging.getLogger(__file__)
CONFIDENCE_THRESHOLD = 0.70
MAX_LEN = 128 
PREDICTION_BATCH_SIZE = 256


def get_data_loader(args, input_data, tokenizer):

    cache_fp = args.data_path[:args.data_path.rfind('.')] + '_cache'
    if os.path.isfile(cache_fp): 
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
        torch.save(cache_fp)

    prev_len = len(input_ids)
    # remove sequences longer than MAX_LEN
    input_ids = [seq for seq in input_ids if len(seq) < MAX_LEN]
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
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    segment_ids = torch.tensor(segment_ids)

    # create dataloader
    prediction_data = TensorDataset(input_ids, attention_masks, segment_ids)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler = prediction_sampler, batch_size = PREDICTION_BATCH_SIZE)

    return prediction_dataloader

def predict_label(args, model, prediction_dataloader): 
  
    # deactivate dropout, etc. 
    if torch.cuda.is_available(): 
        model.cuda()
        model.eval() 

    yesands = []
    non_yesands = []
    for batch in tqdm(prediction_dataloader): 
    # add batch to GPU
        if torch.cuda.is_available(): 
            batch = tuple(t.to(args.device).to(torch.int64) for t in batch)

        #unpack input 
        b_input_ids, b_attention_masks, b_segment_ids = batch

        # don't store gradients
        with torch.no_grad(): 
            # forward pass
            logits = model(b_input_ids, b_attention_masks, b_segment_ids)

            softmax_logits = torch.nn.functional.softmax(logits[0], dim=1).cpu().numpy()
            #   labels = np.argmax(logits, axis =1).flatten() 

            labels = softmax_logits[:,1] > args.threshold
            labels = labels.astype(int).flatten()

            # TODO: modify how you want to store your prediction results
            for input_id, label, softmax_logit in zip(b_input_ids, labels, softmax_logits): 
                decoded_text_split= decode_input_id(input_id)
                result = {'p': decoded_text_split[0].strip(), 'r': decoded_text_split[1].strip(), 'confidence': {'yesand': softmax_logit[1]*100, 'nonyesand': softmax_logit[0]*100}}
                if label == 1: 
                    yesands.append(result)
                else:
                    non_yesands.append(result)

        
    return yesands, non_yesands 

def decode_input_id(input_id, tokenizer): 
    decoded_text = tokenizer.decode(input_id.to('cpu').numpy(), clean_up_tokenization_spaces=True)
    decoded_text_split = decoded_text[:2]
    decoded_text_split = [re.sub('(\[CLS\]|[-*+_])', '', split).strip() for split in decoded_text_split]
    decoded_text_split = [re.sub('\ \ ', ' ', split).strip() for split in decoded_text_split]

    return decoded_text_split[:2]

def get_cornell_data(data_path): 
    # Load Cornell movie dialogue corpus and and format them in to a list  
    # Each sample has the following format: {'id': int, 'p': str, 'r': str}
    with open(data_path, 'r', encoding='iso8859-15') as f: 
        data = f.readlines() 

    data = sorted(data, key = lambda x: int(x.split('+++$+++')[0][1:]))
    dialogues = [re.sub('\n', '', l.split('+++$+++')[-1]).strip() for l in data] 

    data_to_predict = [] 
    for i in range(0,len(dialogues) -1): 
        dialogue = {'id': i,
                    'p': dialogues[i],
                    'r': dialogues[i+1],}
        data_to_predict.append(dialogue)

    return data_to_predict

def predict(): 
    """Determine which are yes-ands are not from a given dialogue data set with a finetuned BERT yes-and classifier"""
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", default="runs/yesand_bert_classifier", help="Provide a directory for a pretrained BERT model.")
    parser.add_argument("--data_path", default="data/cornell_movies.txt", help="Provide a datapath for which predictions will be made.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--threshold", type=str, default=CONFIDENCE_THRESHOLD, help="Set the confidence threshold for yes-ands")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: {}".format(pformat(args)))

    logger.info("Loading model and tokenizer.")
    model = BertForSequenceClassification.from_pretrained(args.model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    logger.info("Loading data to predict: {}".format(args.data_path))
    data_to_predict = get_cornell_data(args.data_path)

    logger.info("Building data loader")
    prediction_dataloader = get_data_loader(data_to_predict, tokenizer)

    logger.info("Making predictions")
    yesands, non_yesands = predict_label(args, model, prediction_dataloader)
    logger.info("Remove single word prompt/response yes-ands")
    yesands_filtered = [yesand for yesand in yesands if len(yesand['p'].split()) > 1 and len(yesand['r'].split()) > 1]
    logger.info("Predictions complete. ")

    yesands_fp = "{}_{:.0f}_{}".format(args.data_path[:args.data_path.rfind('.')], args.threshold*100,'yesands.json')
    logger.info("Saving yesands to: {}".format(yesands_fp))
    yesand_dict = {'threshold': args.threshold, 'yesands': yesands_filtered}
    with open(yesands_fp, 'r') as f: 
        json.dump(yesand_dict, f)
    
    print("Predicted and filtered yesands: {}\nPredicted non-yesands: {}".format(len(yesands_filtered), len(non_yesands)))
    print("Proportion of yesands: {:.2f}%".format(len(yesands_filtered) / (len(data_to_predict)) * 100))

if __name__ == "__main__": 
    predict() 
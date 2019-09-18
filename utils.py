import numpy as np 
from keras.preprocessing.sequence import pad_sequences 
import logging
import json 
import random

logger = logging.getLogger(__file__)

YESAND_DATAPATH = '../yes-and-data.json'
MAX_LEN = 128 


def calc_metrics(pred, labels): 
    """Function to calculate the accuracy of predictions vs labels """
    pred_flat = np.argmax(pred, axis = 1).flatten()
    labels_flat = labels.flatten()
  
    flat_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
  
    # sklearn takes first parameter as the true label
    precision = precision_score(labels_flat, pred_flat)
    recall = recall_score(labels_flat, pred_flat)
  
    return flat_accuracy, precision, recall

def calc_f1(pred,labels): 
    pred_flat = np.argmax(pred, axis = 1).flatten()
    labels_flat = labels.flatten()
  
    # f1_score from sklearn.metrics take first parameter as the true label
    return f1_score(labels_flat, pred_flat)

def build_segment_ids(input_ids):
    """ Create segment ids to differentiate sentence1 and sentence2 """ 
    segment_ids = [] 
    for seq in input_ids: 
        segment_id = []
        id_ = 0
        for token_id in seq: 
            segment_id.append(id_)
            # 102 : [SEP]
            if token_id == 102: 
                id_ +=1 
                id_ %= 2 
        segment_ids.append(segment_id)
    return segment_ids 

def build_attention_mask(input_ids): 

    """ Create attention masks to differentiate from valid input and pads""" 
    attention_masks = [] 

    # 1 for input and 0 for pad
    for seq in input_ids: 
        attention_masks.append([float(i>0) for i in seq])

    return attention_masks 

def get_data(data_path=None):

    data_path = data_path or YESAND_DATAPATH

    logger.info("Loading data from: {}".format(data_path))
    with open(data_path, 'r') as f: 
        data = json.load(f) 
    logger.info("Loaded data from: {}".format(data_path))

    # make sure data set is balanced
    total_yes_ands = 0
    for k in data['yes-and'].keys(): 
        total_yes_ands += len(data['yes-and'][k])
    
    logger.info("Total number of yes-ands: {}".format(total_yes_ands))

    return data 

def build_bert_input(data, tokenizer): 

    """
    Format data as BERT input 
    sequence: "[CLS] <sentence1> [SEP] <sentence2> [SEP]"
    """
    all_samples = [] 
    for non_yesand in data['non-yes-and']['cornell']: 
        seq = "[CLS] {} [SEP] {} [SEP]".format(non_yesand['p'], non_yesand['r'])
        all_samples.append([0, seq])
    
    for k in data['yes-and'].keys(): 
        for yesand in data['yes-and'][k]: 
            seq = "[CLS] {} [SEP] {} [SEP]".format(yesand['p'], yesand['r'])
            all_samples.append([1, seq])
        
    random.shuffle(all_samples)

    sentences = [x[1] for x in all_samples]
    labels = [x[0] for x in all_samples]

    # tokenize with BERT tokenizer 
    tokenized_texts = [tokenizer.encode(sentence) for sentence in sentences]

    # pad input to MAX_LEN
    input_ids = pad_sequences(tokenized_texts, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # get attention masks and segment ids 
    attention_masks = build_attention_mask(input_ids)
    segment_ids = build_segment_ids(input_ids)

    return input_ids, attention_masks, segment_ids, labels



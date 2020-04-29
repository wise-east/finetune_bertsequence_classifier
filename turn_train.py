import os 
import numpy as np 
import json
import random 
import logging
from pprint import pformat
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path 

import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, AdamW, WEIGHTS_NAME, CONFIG_NAME
from transformers.optimization import WarmupLinearSchedule
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage, Average, Accuracy, Precision, Recall
from ignite.contrib.metrics import GpuInfo
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from utils import  get_data, build_bert_input, build_roberta_input, ROBERTA_MAX_LEN
from train import get_data_loaders

logger = logging.getLogger(__file__)

def train(): 
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default='data/yesands_train_iter4.json', help="Set data path")    
    parser.add_argument("--valid_path", type=str, default='data/yesands_valid.json', help="Set data path")     

    parser.add_argument("--correct_bias", type=bool, default=False, help="Set to true to correct bias for Adam optimizer")
    parser.add_argument("--lr", type=float, default=2e-5, help="Set learning rate")
    parser.add_argument("--n_epochs", type=int, default=4, help="Set number of epochs")
    parser.add_argument("--num_warmup_steps", type=float, default=1000, help="Set number of warm-up steps")
    parser.add_argument("--num_total_steps", type=float, default=10000, help="Set number of total steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Set maximum gradient normalization.")
    parser.add_argument("--pretrained_path", type=str, default='roberta-base', help="Choose which pretrained model to use (bert-base-uncased, roberta-base, roberta-large, roberta-large-mnli)")    
    parser.add_argument("--batch_size", type=int, default=32, help="Provide the batch size")    
    parser.add_argument("--random_seed", type=int, default=42, help="Set the random seed")
    parser.add_argument("--test", action='store_true', help="If true, run with small dataset for testing code")
    parser.add_argument("--base", action='store_true', help="If true, run with base experiment configuration (training with spont only) for comparison")

    args = parser.parse_args() 

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: {}".format(pformat(args)))

    if 'roberta' in args.pretrained_path: 
        # initialize tokenizer and model 
        logger.info("Initialize model and tokenizer.")
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_path, cache_dir = '../pretrained_models')
        model = RobertaForSequenceClassification.from_pretrained(args.pretrained_path, cache_dir='../pretrained_models')

        ### START MODEL MODIFICATION
        # Pretrained model was not trained with token type ids. 
        # fix token type embeddings for finetuning. Without this, the model can only take 0s as valid input for token_type_ids 
        model.config.type_vocab_size = 2 
        model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, model.config.hidden_size)
        model.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

        ### END MOD
    elif 'bert' in args.pretrained_path: 
        model = BertForSequenceClassification.from_pretrained(args.pretrained_path, cache_dir='../pretrained_models')
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_path, cache_dir='../pretrained_models')

    model.to(args.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01}, 
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=args.lr,
                        correct_bias = args.correct_bias)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.num_warmup_steps, t_total=args.num_total_steps) 

    logger.info("Prepare datasets")

    train_data = get_data(args.train_path)
    valid_data = get_data(args.valid_path)

    if arg.test: 
        train_data = train_data[:100]
        valid_data = valid_data[:100]

    logger.info("Loading train set...")
    train_loader, train_sampler = get_data_loaders(args, train_data, args.train_path, tokenizer)
    
    logger.info("Loading validation set...")
    valid_loader, valid_sampler = get_data_loaders(args, valid_data, args.valid_path,  tokenizer)



    # Training function and trainer 
    def update(engine, batch): 
        model.train() 

        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        b_input_ids, b_input_mask, b_input_segment, b_labels = batch

        optimizer.zero_grad()
        #roberta has issues with token_type_ids 
        loss, logits = model(b_input_ids, token_type_ids=b_input_segment, attention_mask=b_input_mask, labels=b_labels)
        # loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)


        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step() 
        scheduler.step() 

        return loss.item(), logits, b_labels

    trainer = Engine(update)     

    val_result_f = Path('turn_val_prediction.txt').open('w')

    # Evaluation function and evaluator 
    def inference(engine, batch): 
        model.eval() 

        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        b_input_ids, b_input_mask, b_input_segment, b_labels = batch
        
        with torch.no_grad(): 
            #roberta has issues with token_type_ids 
            # loss, logits = model(b_input_ids, token_type_ids = None, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = model(b_input_ids, token_type_ids = b_input_segment, attention_mask=b_input_mask, labels=b_labels)
            label_ids = b_labels



        return logits, label_ids, loss.item()
    evaluator = Engine(inference)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss") 
    RunningAverage(Accuracy(output_transform=lambda x: (x[1], x[2]))).attach(trainer, "accuracy")
    if torch.cuda.is_available(): 
        GpuInfo().attach(trainer, name='gpu')

    recall = Recall(output_transform=lambda x: (x[0], x[1]))
    precision = Precision(output_transform=lambda x: (x[0], x[1]))
    F1 = (precision * recall * 2 / (precision + recall)).mean()
    accuracy = Accuracy(output_transform=lambda x: (x[0], x[1]))
    metrics = {"recall": recall, "precision": precision, "f1": F1, "accuracy": accuracy, "loss": Average(output_transform=lambda x: x[2])}

    for name, metric in metrics.items(): 
        metric.attach(evaluator, name) 


    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['loss', 'accuracy'])
    pbar.attach(trainer, metric_names=['gpu:0 mem(%)', 'gpu:0 util(%)'])
    
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation metrics:\n %s" % pformat(evaluator.state.metrics)))


    tb_logger = TensorboardLogger(log_dir=None)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
    tb_logger.attach(evaluator, log_handler=OutputHandler(tag="valid", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)


    # tb_logger.writer.log_dir -> tb_logger.writer.logdir (this is the correct attribute name as seen in: https://tensorboardx.readthedocs.io/en/latest/_modules/tensorboardX/writer.html#SummaryWriter)
    checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=5)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

    torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
    tokenizer.save_vocabulary(tb_logger.writer.logdir)

    trainer.run(train_loader, max_epochs = args.n_epochs)

    if args.n_epochs > 0: 
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.logdir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


    val_result_f.close()

if __name__ =="__main__": 
    train() 


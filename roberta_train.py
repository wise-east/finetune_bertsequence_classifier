import os 
import numpy as np 
import json
import random 
import logging
from pprint import pformat
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, WEIGHTS_NAME, CONFIG_NAME
from transformers.optimization import WarmupLinearSchedule
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage, Precision, Recall
from ignite.contrib.metrics import GpuInfo
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

from utils import  get_data, YESAND_DATAPATH, build_roberta_input, ROBERTA_MAX_LEN

logger = logging.getLogger(__file__)

# recommended settings: batch size = 32, max length = 128 

def get_data_loaders(args, tokenizer):

    data = get_data(args.data_path)
    all_samples = build_roberta_input(data, args.data_path, tokenizer)
    random.shuffle(all_samples)
    train_samples, validation_samples = train_test_split(all_samples, random_state=args.random_seed, test_size=args.test_size)

    train_inputs, train_labels, train_masks, train_token_types = [s['input_ids'] for s in train_samples], [s['label'] for s in train_samples], [s['attention_mask'] for s in train_samples], [s['token_type_ids'] for s in train_samples] 
    validation_inputs, validation_labels, validation_masks, validation_token_types = [s['input_ids'] for s in validation_samples], [s['label'] for s in validation_samples], [s['attention_mask'] for s in validation_samples], [s['token_type_ids'] for s in validation_samples] 

    # wrap the data as tensors 
    train_inputs, train_labels, train_masks, train_token_types = [torch.tensor(x) for x in [train_inputs, train_labels, train_masks, train_token_types]]
    validation_inputs, validation_labels, validation_masks, validation_token_types = [torch.tensor(x) for x in [validation_inputs, validation_labels, validation_masks, validation_token_types]]

    # group as tensor datasets
    train_data = TensorDataset(train_inputs, train_masks, train_token_types, train_labels)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_token_types, validation_labels)

    # build dataloaders
    train_sampler = RandomSampler(train_data) 
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = args.batch_size)

    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = args.batch_size)

    return train_dataloader, validation_dataloader, train_sampler, validation_sampler

def train(): 
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=YESAND_DATAPATH, help="Set data path")    
    parser.add_argument("--correct_bias", type=bool, default=False, help="Set to true to correct bias for Adam optimizer")
    parser.add_argument("--lr", type=float, default=5e-6, help="Set learning rate")
    parser.add_argument("--n_epochs", type=int, default=5, help="Set number of epochs")
    parser.add_argument("--num_warmup_steps", type=float, default=1000, help="Set number of warm-up steps")
    parser.add_argument("--num_total_steps", type=float, default=10000, help="Set number of total steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Set maximum gradient normalization.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Set proportion of validation size split")
    parser.add_argument("--pretrained_path", type=str, default='roberta-base', help="Choose which pretrained Roberta to use (roberta-base, roberta-large, roberta-large-mnli)")    
    parser.add_argument("--batch_size", type=int, default=32, help="Provide the batch size")    
    parser.add_argument("--random_seed", type=int, default=42, help="Set the random seed")

    args = parser.parse_args() 

    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: {}".format(pformat(args)))

    # initialize tokenizer and model 
    logger.info("Initialize model and tokenizer.")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_path, cache_dir = '../')
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model = RobertaForSequenceClassification.from_pretrained(args.pretrained_path, cache_dir='../')
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
    train_loader, valid_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer 
    def update(engine, batch): 
        model.train() 
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        b_input_ids, b_input_mask, b_input_segment, b_labels = batch

        optimizer.zero_grad()
        #roberta has issues with token_type_ids 
        # loss, logits = model(b_input_ids, token_type_ids=b_input_segment, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)


        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step() 
        scheduler.step() 

        return loss.item(), logits, b_labels

    trainer = Engine(update)     

    # Evaluation function and evaluator 
    def inference(engine, batch): 
        model.eval() 

        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        b_input_ids, b_input_mask, b_input_segment, b_labels = batch
        
        with torch.no_grad(): 
            #roberta has issues with token_type_ids 
            logits = model(b_input_ids, token_type_ids = None, attention_mask=b_input_mask)
            # logits = model(b_input_ids, token_type_ids = b_input_segment, attention_mask=b_input_mask)
            logits = logits[0]
            label_ids = b_labels

        return logits, label_ids
    evaluator = Engine(inference)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss") 
    RunningAverage(Accuracy(output_transform=lambda x: (x[1], x[2]))).attach(trainer, "accuracy")
    if torch.cuda.is_available(): 
        GpuInfo().attach(trainer, name='gpu')

    metrics = {"recall": Recall(output_transform=lambda x: (x[0], x[1])), "precision": Precision(output_transform=lambda x: (x[0], x[1])), "accuracy": Accuracy(output_transform=lambda x: (x[0], x[1]))}

    for name, metric in metrics.items(): 
        metric.attach(evaluator, name) 

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['loss', 'accuracy'])
    pbar.attach(trainer, metric_names=['gpu:0 mem(%)', 'gpu:0 util(%)'])
    
    evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation %s" % pformat(evaluator.state.metrics)))

    tb_logger = TensorboardLogger(log_dir=None)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
    tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

    # tb_logger.writer.log_dir -> tb_logger.writer.logdir (this is the correct attribute name as seen in: https://tensorboardx.readthedocs.io/en/latest/_modules/tensorboardX/writer.html#SummaryWriter)
    checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

    torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
    tokenizer.save_vocabulary(tb_logger.writer.logdir)

    trainer.run(train_loader, max_epochs = args.n_epochs)

    if args.n_epochs > 0: 
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.logdir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ =="__main__": 
    train() 


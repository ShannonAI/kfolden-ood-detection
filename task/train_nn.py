#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: train_nn.py
# desc: text_classification_task for both train and evaluate

import os
import re
import argparse
from collections import namedtuple
from utils.random_seed import set_random_seed
set_random_seed(2333)

# https://github.com/PyTorchLightning/pytorch-lightning/issues/2757
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from transformers import AdamW, AutoTokenizer, BertTokenizer, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from datasets.twenty_newsgroups import TwentyNewsDataset
from datasets.collate_functions import collate_20newsgroups_to_max_length
from utils.get_parser import get_parser, add_cnn_configurations, add_rnn_configurations, add_basic_configurations
from models.rnn import RNNForTextClassification
from models.cnn import CNNForTextClassification
from models.model_config import RNNTextClassificationConfig, CNNTextClassificationConfig


class TextClassificationTask(pl.LightningModule):
    """Model Trainer for GLUE tasks."""
    def __init__(self, args: argparse.Namespace):
        """initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            print(f"DEBUG INFO -> save hyperparameters")
            self.save_hyperparameters(args)
            self.args = args
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.data_dir = args.data_dir
        self.optimizer = args.optimizer
        self.train_batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size
        self.num_classes = self.args.num_labels
        self.tokenizer = None if self.args.model_type.lower() != "bert" else AutoTokenizer.from_pretrained(self.args.bert_config_dir, use_fast=False, do_lower_case=self.args.do_lower_case)

        if args.model_type == "cnn":
            config = CNNTextClassificationConfig(vocab_size=args.vocab_size, embedding_size=args.embedding_size,
                                                 hidden_size=args.hidden_size, num_layers=args.num_layers,
                                                 num_kernels=args.num_kernels, conv_stride=args.conv_stride, kernel_size=args.kernel_size,
                                                 pooling_strategy=args.pooling_strategy, dropout=args.dropout,
                                                 num_labels=self.num_classes, padding_idx=args.padding_idx,
                                                 init_word_embedding=args.init_word_embedding, freeze_word_embedding=args.freeze_word_embedding,
                                                 classifier_type=args.classifier_type, activate_func=args.activate_func)
            self.model = CNNForTextClassification(config)
        elif args.model_type == "rnn":
            config = RNNTextClassificationConfig(vocab_size=args.vocab_size, embedding_size=args.embedding_size,
                                                 hidden_size=args.hidden_size, num_layers=args.num_layers,
                                                 bidirectional=args.bidirectional, batch_first=args.batch_first, rnn_cell_type=args.rnn_cell_type,
                                                 dropout=args.dropout, rnn_dropout=args.rnn_dropout, rnn_activate_func=args.rnn_activate_func,
                                                 num_labels=self.num_classes, padding_idx=args.padding_idx,
                                                 init_word_embedding=args.init_word_embedding, freeze_word_embedding=args.freeze_word_embedding,
                                                 classifier_type=args.classifier_type, activate_func=args.activate_func)
            self.model = RNNForTextClassification(config)
        else:
            raise ValueError("the input model_type does not exist.")

        output_logging_file = os.path.join(self.args.output_dir, self.args.log_file)
        self.result_logger = open(output_logging_file, "a")
        self.gpus = args.gpus.split(",") if "," in str(args.gpus) else args.gpus
        self.metric_accuracy = pl.metrics.Accuracy(ddp_sync_on_step=True)
        self.num_gpus = 1
        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.result_logger.write(f"INFO -> NUMBER OF PARAMS {self.pytorch_total_params}")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--dataset_name", type=str, default="20news", help=" The name of the task to  train.")
        parser.add_argument("--model_type", type=str, default="bert")
        parser = add_basic_configurations(parser)
        parser = add_rnn_configurations(parser)
        parser = add_cnn_configurations(parser)
        return parser

    def configure_optimizers(self, ):
        """prepare optimizer and lr scheduler (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.999),  # according to RoBERTa paper
                              lr=self.args.lr, eps=self.args.adam_epsilon, )
        elif self.optimizer == "torch.adam":
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr,
                                          eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        elif self.optimizer == "adadelta":
            optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad(optimizer_grouped_parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError

        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps / t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=self.args.lr / 4.0)
        else:
            raise ValueError("lr_scheduler doesnot exists.")

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if self.args.model_type.lower() == "bert":
            return self.model(input_ids, token_type_ids, attention_mask)
        else:
            return self.model(input_ids)

    def compute_loss(self, logits, labels):
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = {"lr": self.trainer.optimizers[0].param_groups[0]['lr']}

        input_ids, gold_labels = batch[0], batch[1]
        output_logits = self.model(input_ids)
        loss = self.compute_loss(output_logits, gold_labels)

        tf_board_logs[f"loss"] = loss
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}
        input_ids, gold_labels = batch[0], batch[1]
        output_logits = self.model(input_ids)
        loss = self.compute_loss(output_logits, gold_labels)
        pred_labels = _transform_logits_to_labels(output_logits)
        self.metric_accuracy.update(pred_labels, gold_labels)

        output[f"val_loss"] = loss
        return output

    def validation_epoch_end(self, outputs, prefix="dev"):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = self.metric_accuracy.compute()
        tensorboard_logs = {"val_loss": avg_loss}
        tensorboard_logs[f"acc"] = avg_acc

        self.result_logger.write(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} \n")
        print(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, val_acc is: {avg_acc} \n")

        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_acc": avg_acc}

    def test_step(self, batch, batch_idx):
        output = {}
        input_ids, gold_labels = batch[0], batch[1]
        output_logits = self.model(input_ids)
        pred_labels = _transform_logits_to_labels(output_logits)
        batch_acc = self.metric_accuracy.forward(pred_labels, gold_labels)

        output[f"test_acc"] = batch_acc
        return output

    def test_epoch_end(self, outputs, prefix="test"):
        tensorboard_logs = {}
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean() / self.num_gpus

        confusion_matrix = torch.sum(torch.stack([x[f"stats_confusion_matrix"] for x in outputs], dim=0), 0, keepdim=False)
        tensorboard_logs[f"test_acc"] = avg_acc

        self.result_logger.write(f"TEST INFO -> test_acc is: {avg_acc} \n")

        return {"test_log": tensorboard_logs, "test_acc": avg_acc}

    def train_dataloader(self, ):
        return self.get_dataloader(prefix="train")

    def val_dataloader(self, ):
        return self.get_dataloader(prefix="dev")

    def test_dataloader(self, ):
        return self.get_dataloader(prefix="test")

    def get_dataloader(self, prefix="train", ):
        """read vocab and dataset files"""
        dataset = TwentyNewsDataset(self.args.data_dir, prefix, max_length=self.args.max_length,
                                    tokenizer=self.tokenizer, allow_ood=False,
                                    vocab_file=self.args.vocab_file,
                                    remove=["headers", "footers", "quotes"],
                                    label_files=self.args.label_file, do_lower_case=self.args.do_lower_case)

        if prefix == "train":
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
            batch_size = self.train_batch_size
        else:
            data_sampler = SequentialSampler(dataset)
            batch_size = self.eval_batch_size

        dataloader = DataLoader(dataset=dataset,
                                sampler=data_sampler,
                                batch_size=batch_size,
                                num_workers=self.args.workers,
                                collate_fn=collate_20newsgroups_to_max_length)

        return dataloader


def _transform_logits_to_labels(output_logits):
    # output_logits -> [batch_size, num_labels]
    output_probs = F.softmax(output_logits, dim=-1)
    pred_labels = torch.argmax(output_probs, dim=1)

    return pred_labels


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt", only_keep_the_best_ckpt: bool = True):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = ""
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(
            re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(
            " as top", "")

        if current_f1 >= best_f1_on_dev:
            if only_keep_the_best_ckpt and len(best_checkpoint_on_dev) != 0:
                os.remove(best_checkpoint_on_dev)
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def main():
    parser = get_parser()
    parser = TextClassificationTask.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    text_classification_task = TextClassificationTask(args)

    if len(args.pretrained_checkpoint) > 1:
        text_classification_task.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=torch.device("cpu"))["state_dict"])

    checkpoint_callback = ModelCheckpoint(filepath=args.output_dir, save_top_k=args.max_keep_ckpt, save_last=False, monitor="val_acc", verbose=True, mode='max', period=-1)

    task_trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, deterministic=True, auto_select_gpus=True,)

    task_trainer.fit(text_classification_task)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.output_dir,
                                                                          log_file=args.log_file,
                                                                          only_keep_the_best_ckpt=args.only_keep_the_best_ckpt_after_training)
    text_classification_task.result_logger.write(f"{'=&' * 20} \n")
    text_classification_task.result_logger.write(f"Best F1 on DEV is {best_f1_on_dev} \n")
    text_classification_task.result_logger.write(f"Best checkpoint on DEV set is {path_to_best_checkpoint} \n")
    text_classification_task.result_logger.write(f"{'=&' * 20} \n")


if __name__ == "__main__":
    main()
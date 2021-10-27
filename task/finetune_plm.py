#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: task/finetune_plm.py

import os
import re
import sys
import csv
import argparse
import logging
from collections import namedtuple
from utils.random_seed import set_random_seed
from utils.get_parser import get_plm_parser

# https://github.com/PyTorchLightning/pytorch-lightning/issues/2757
import warnings
warnings.filterwarnings('ignore')
csv.field_size_limit(sys.maxsize)

import torch
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from data.datasets.label_fields import get_labels
from data.datasets.plm_doc_dataset import PLMDocDataset
from data.datasets.collate_functions import collate_plm_to_max_length
from models.plm import BertForSequenceClassification, RobertaForSequenceClassification
from models.model_config import BertForSequenceClassificationConfig, RobertaForSequenceClassificationConfig


class FinetunePLMTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, keep_label_lst="", save_output_dir=""):
        """initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            print(f"DEBUG INFO -> save hyperparameters")
            self.save_hyperparameters(args)
            self.args = args
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.keep_label_lst = keep_label_lst
        self.num_classes = len(keep_label_lst)
        self.save_output_dir = save_output_dir
        self.loss_name = self.args.loss_name
        self.model_path = self.args.bert_config_dir
        self.data_dir = self.args.data_dir
        self.optimizer = self.args.optimizer
        self.train_batch_size = self.args.train_batch_size
        self.eval_batch_size = self.args.eval_batch_size

        if args.model_type == "bert":
            bert_config = BertForSequenceClassificationConfig.from_pretrained(self.model_path, num_labels=self.num_classes,
                                                                              hidden_dropout_prob=self.args.hidden_dropout_prob,)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path, config=bert_config)
        elif args.model_type == "roberta":
            roberta_config = RobertaForSequenceClassificationConfig.from_pretrained(self.model_path, num_labels=self.num_classes,
                                                                                    hidden_dropout_prob=self.args.hidden_dropout_prob,)
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_path, config=roberta_config)
        else:
            raise ValueError
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, do_lower_case=self.args.do_lower_case)

        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.metric_accuracy = pl.metrics.Accuracy(num_classes=self.num_classes)
        self.num_gpus = 1

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_type", type=str, default="bert")
        parser.add_argument("--data_name", type=str, default="20news_6s", help=" The name of the task to  train.")
        parser.add_argument("--model_scale", type=str, default="single", choices=["kfolden", "single", "ensemble"])
        parser.add_argument("--num_of_ensemble", type=int, default=0)
        parser.add_argument("--loss_name", type=str, default="ce", help=" The name of the task to  train.")
        parser.add_argument("--pad_to_max_length", action="store_false", help="Whether to pad all samples to ' max_seq_length'.")
        return parser

    def configure_optimizers(self,):
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
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
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
        else:
            raise ValueError("lr_scheduler doesnot exists.")

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        return self.model(input_ids, token_type_ids, attention_mask)

    def compute_loss(self, logits, labels, id_label_mask, eps=1e-16):
        if self.loss_name == "ce":
            ce_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
            data_loss = ce_loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            avg_loss = torch.sum(data_loss * id_label_mask) / torch.sum(id_label_mask.float() + eps)
        elif self.loss_name == "kfolden":
            ce_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
            ce_data_loss = ce_loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            ce_avg_loss = torch.sum(ce_data_loss * id_label_mask) / (torch.sum(id_label_mask.float()) + eps)
            ood_label_mask = torch.tensor(id_label_mask==0, dtype=torch.long)
            ood_single_target = 1 / float(self.num_classes)
            normed_logits = F.softmax(logits, dim=-1)
            ood_target = torch.zeros_like(logits).fill_(ood_single_target).to('cuda')
            kl_loss = F.kl_div(normed_logits, ood_target, reduction='none', log_target=False)
            kl_avg_loss = torch.sum(kl_loss) / (torch.sum(ood_label_mask) + eps)
            avg_loss = ce_avg_loss + self.args.lambda_loss * kl_avg_loss
        return avg_loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = {"lr": self.trainer.optimizers[0].param_groups[0]['lr']}
        input_ids, token_type_ids, attention_mask, id_label_mask, gold_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
        output_logits = self.model(input_ids, token_type_ids, attention_mask)
        loss = self.compute_loss(output_logits, gold_labels, id_label_mask)
        tf_board_logs[f"loss"] = loss
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}
        input_ids, token_type_ids, attention_mask, id_label_mask, gold_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
        output_logits = self.model(input_ids, token_type_ids, attention_mask)
        loss = self.compute_loss(output_logits, gold_labels, id_label_mask)
        pred_label_probs, pred_labels = self._transform_logits_to_labels(output_logits)
        norm_pred_labels = pred_labels * id_label_mask
        norm_gold_labels = gold_labels * id_label_mask
        batch_acc = self.metric_accuracy.forward(norm_pred_labels, norm_gold_labels)
        output[f"val_loss"] = loss
        output[f"val_acc"] = batch_acc
        return output

    def validation_epoch_end(self, outputs, prefix="dev"):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs = {"val_loss": avg_loss}
        tensorboard_logs[f"acc"] = avg_acc
        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_acc": avg_acc}

    def test_step(self, batch, batch_idx):
        output = {}
        input_ids, token_type_ids, attention_mask, id_label_mask, gold_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
        output_logits = self.model(input_ids, token_type_ids, attention_mask)
        pred_label_probs, pred_labels = self._transform_logits_to_labels(output_logits)
        batch_acc = self.metric_accuracy.forward(pred_labels, gold_labels)
        output[f"test_acc"] = batch_acc
        return output

    def test_epoch_end(self, outputs, prefix="test"):
        tensorboard_logs = {}
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean() / self.num_gpus
        tensorboard_logs[f"test_acc"] = avg_acc
        self.result_logger.info(f"TEST INFO -> test_acc is: {avg_acc}")
        return {"test_log": tensorboard_logs, "test_acc": avg_acc}

    def train_dataloader(self, ):
        return self.get_dataloader(prefix="train")

    def val_dataloader(self, ):
        return self.get_dataloader(prefix="dev")

    def test_dataloader(self, ):
        return self.get_dataloader(prefix="test")

    def get_dataloader(self, prefix="train", limit: int = None):
        """read vocab and dataset files"""
        dataset = PLMDocDataset(self.args, self.tokenizer, mode=prefix, keep_label_lst=self.keep_label_lst)
        if prefix == "train":
            # define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
            batch_size = self.train_batch_size
        else:
            data_sampler = SequentialSampler(dataset)
            batch_size = self.eval_batch_size

        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=collate_plm_to_max_length)
        return dataloader

    def _transform_logits_to_labels(self, output_logits):
        # output_logits -> [batch_size, num_labels]
        pred_probs = F.softmax(output_logits, dim=-1)
        pred_labels = torch.argmax(pred_probs, dim=1)
        return pred_probs, pred_labels

def find_best_checkpoint_on_dev(output_dir: str, path_prefix: str, log_file: str = "eval_result_log.txt", only_keep_the_best_ckpt: bool = False):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    ACC_PATTERN = re.compile(r"val_acc reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_acc_on_dev = 0
    best_checkpoint_on_dev = ""
    for checkpoint_info_line in checkpoint_info_lines:
        if path_prefix not in checkpoint_info_line:
            continue
        current_acc = float(
            re.findall(ACC_PATTERN, checkpoint_info_line)[0].replace("val_acc reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(" as top", "")

        if current_acc >= best_acc_on_dev:
            if only_keep_the_best_ckpt and len(best_checkpoint_on_dev) != 0:
                os.remove(best_checkpoint_on_dev)
            best_acc_on_dev = current_acc
            best_checkpoint_on_dev = current_ckpt
    return best_acc_on_dev, best_checkpoint_on_dev


def finetune_model(args, save_output_dir, keep_label_lst):

    task_model = FinetunePLMTask(args, keep_label_lst=keep_label_lst, save_output_dir=save_output_dir)
    if len(args.pretrained_checkpoint) > 1:
        task_model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=torch.device("cpu"))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=save_output_dir,
        save_top_k=args.max_keep_ckpt,
        save_last=False,
        monitor="val_acc",
        verbose=True,
        mode='max',
        period=-1)

    task_trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, deterministic=True)
    task_trainer.fit(task_model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_acc_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.output_dir, save_output_dir+"/", only_keep_the_best_ckpt=args.only_keep_the_best_ckpt_after_training)
    task_model.result_logger.info("=&" * 20)
    task_model.result_logger.info(f"saved output dir is : {save_output_dir}")
    task_model.result_logger.info(f"Best ACC on DEV is {best_acc_on_dev}")
    task_model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    task_model.result_logger.info("=&" * 20)
    file_saving_best_ckpt = os.path.join(save_output_dir, "best_ckpt_on_dev.txt")
    with open(file_saving_best_ckpt, "w") as f:
        f.write(f"{path_to_best_checkpoint}\n")


def save_label_to_file(label_lst, label_file):
    with open(label_file, "w") as f:
        for label_item in label_lst:
            f.write(f"{label_item}\n")

def main():
    set_random_seed(2333)
    parser = get_plm_parser()
    parser = FinetunePLMTask.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    full_label_lst = get_labels(args.data_name, dist_sign="id")
    format = '%(asctime)s - %(name)s - %(message)s'
    logging.basicConfig(format=format, filename=os.path.join(args.output_dir, "eval_result_log.txt"), level=logging.INFO)

    if args.enable_leave_label_out:
        save_label_to_file(full_label_lst, os.path.join(args.output_dir, "labels.txt"))
        for idx in range(0, len(full_label_lst), args.num_of_left_label):
            print("$"*30)
            print(f">>> ENABLE LEAVE OUT TRAINING ...")
            left_label_lst = [item for item_idx, item in enumerate(full_label_lst) if item_idx in range(idx, idx+args.num_of_left_label)]
            keep_label_lst = [item for item_idx, item in enumerate(full_label_lst) if item_idx not in range(idx, idx+args.num_of_left_label)]
            sub_output_dir = os.path.join(args.output_dir, f"{idx}")
            os.makedirs(sub_output_dir, exist_ok=True)
            os.system(f"chmod -R 777 {sub_output_dir}")
            save_label_to_file(keep_label_lst, os.path.join(sub_output_dir, "keep_labels.txt"))
            save_label_to_file(left_label_lst, os.path.join(sub_output_dir, "left_labels.txt"))
            print(f">>> target labels:  {keep_label_lst}")
            finetune_model(args, sub_output_dir, keep_label_lst)
    else:
        if args.model_scale == "single":
            print("$"*30)
            print(f">>> ENABLE ALL LABEL TRAINING ...")
            print(f">>> target labels:  {full_label_lst}")
            os.makedirs(args.output_dir, exist_ok=True)
            os.system(f"chmod -R 777 {args.output_dir}")
            save_label_to_file(full_label_lst, os.path.join(args.output_dir, "labels.txt"))
            finetune_model(args, args.output_dir, full_label_lst,)
        elif args.model_scale == "ensemble":
            if args.num_of_ensemble == 0:
                num_of_ensemble = len(full_label_lst)
            else:
                num_of_ensemble = args.num_of_ensemble

            for iteration_idx in range(num_of_ensemble):
                set_random_seed(2334+iteration_idx)
                args.seed = 2334+iteration_idx
                print("$" * 30)
                print(f"seed in args: {args.seed}; seed in torch: {torch.initial_seed()}")
                print(f">>> ENABLE ALL LABEL TRAINING ...")
                print(f">>> target labels:  {full_label_lst}")
                sub_output_dir = os.path.join(args.output_dir, f"{iteration_idx}")
                os.makedirs(sub_output_dir, exist_ok=True)
                os.system(f"chmod -R 777 {sub_output_dir}")
                with open(os.path.join(sub_output_dir, "seed.txt"), "w") as f:
                    f.write(f"torch seed: {torch.initial_seed()}")
                    f.write(f"args seed: {args.seed}")
                finetune_model(args, sub_output_dir, full_label_lst)
        else:
            raise ValueError("Please append --enable_leave_label_out in you command . ")



if __name__ == "__main__":
    main()



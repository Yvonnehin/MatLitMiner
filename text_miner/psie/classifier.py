#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import re
from psie.data import get_entities
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2  # 输入维度、隐藏层维度和输出维度

        bert_path = r'/pretrained_models/m3rg-iitd/matscibert'
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型，并用该模型参数初始化(m3rg-iitd/matscibert)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

    def evaluate(self, val_loader, device):
        self.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        val_loss = 0

        criterion = FocalLoss().to(device)

        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device, dtype=torch.long)
                mask = batch['attention_mask'].to(device, dtype=torch.long)
                labels = batch['isrelevant'].to(device, dtype=torch.long).view(-1)
                predictions = self(input_ids=ids, attention_mask=mask)
                _, predicted = torch.max(predictions, 1)

                val_loss += criterion(predictions, labels).item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        val_loss = val_loss / len(val_loader)

        return val_loss, accuracy, precision, recall, f1

    def finetuning(self, train_loader, val_loader, device, max_norm, optimizer, weight):
        total_loss_tr = 0
        nb_tr_steps = 0
        tr_labels = []
        tr_predictions = []

        criterion = FocalLoss(weight=weight).to(device)
        self.train()

        for idx, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['isrelevant'].to(device, dtype=torch.long).view(-1)

            predictions = self(input_ids=ids, attention_mask=mask).view(-1, 2)

            batch_loss = criterion(predictions, labels)
            total_loss_tr += batch_loss.item()

            nb_tr_steps += 1

            _, predicted = torch.max(predictions, 1)
            tr_labels.extend(labels.cpu().numpy())
            tr_predictions.extend(predicted.cpu().numpy())

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=max_norm)

            # backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                loss_step = total_loss_tr / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

        epoch_loss_tr = total_loss_tr / nb_tr_steps
        tr_accuracy = accuracy_score(tr_labels, tr_predictions)

        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.evaluate(val_loader, device)

        print(f"Training loss epoch: {epoch_loss_tr}")
        print(f"Training accuracy epoch: {tr_accuracy}")
        print(f"Validation loss: {val_loss}")
        print(f"Validation accuracy: {val_accuracy}")
        print(f"Validation F1: {val_f1}")
        print(f"Validation Recall: {val_recall}")

        return epoch_loss_tr, tr_accuracy, val_loss, val_accuracy, val_f1, val_recall


        # 用于在测试数据集上评估模型。接收测试数据集的 test_loader 和设备 device，返回预测标签和真实标签。
    def testLabeledData(self, test_loader, device):
        self.eval()
        eval_preds, eval_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)                
                labels = batch['isrelevant'].to(device, dtype = torch.long).view(-1)
                
                predictions = self(input_ids=ids, attention_mask=mask).view(-1, 2)
                
                eval_labels.extend(labels)
                eval_preds.extend(predictions)
                
        return eval_labels, eval_preds

    # 在未标记的数据集上进行预测
    def predict(self, dataloader, device):
        self.eval()
        eval_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                
                predictions = self(input_ids=ids, attention_mask=mask).view(-1, 2)           
                eval_preds.extend(predictions)
                
        return eval_preds
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
from transformers import BertForTokenClassification

from psie.utils import toBertNer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from psie.data import preprocess_text
'''
BertForNer 类继承自 Hugging Face 的 BertForTokenClassification 类，
用于在 BERT 模型的基础上进行 NER 的微调。
它包含了微调方法 finetuning、测试标注数据的方法 testLabeledData 以及预测的方法 predict。
'''
class BertForNer(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)

    

class BertForNer(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)

    def finetuning(self, train_loader, val_loader, device, max_norm, optimizer, scheduler):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        self.train()

        for idx, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["label"].to(device, dtype=torch.long)

            outputs = self(input_ids=ids, attention_mask=mask, labels=labels)

            loss = outputs[0]
            tr_logits = outputs[1]
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            flattened_targets = labels.view(-1)
            active_logits = tr_logits.view(-1, self.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_labels.extend(labels.cpu().numpy())
            tr_preds.extend(predictions.cpu().numpy())

            tmp_tr_accuracy = accuracy_score(
                labels.cpu().numpy(), predictions.cpu().numpy()
            )
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.parameters(), max_norm=max_norm
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

        self.eval()

        val_loss = 0
        nb_val_steps = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_loader):
                val_ids = val_batch["input_ids"].to(device, dtype=torch.long)
                val_mask = val_batch["attention_mask"].to(device, dtype=torch.long)
                val_labels_batch = val_batch["label"].to(device, dtype=torch.long)

                outputs = self(input_ids=val_ids, attention_mask=val_mask, labels=val_labels_batch)
                loss = outputs[0]
                val_logits = outputs[1]
                val_loss += loss.item()

                flattened_val_targets = val_labels_batch.view(-1)
                active_val_logits = val_logits.view(-1, self.num_labels)
                flattened_val_predictions = torch.argmax(active_val_logits, axis=1)

                # only compute accuracy at active labels
                active_val_accuracy = val_labels_batch.view(-1) != -100

                val_labels_batch = torch.masked_select(flattened_val_targets, active_val_accuracy)
                val_predictions = torch.masked_select(flattened_val_predictions, active_val_accuracy)

                val_labels.extend(val_labels_batch.cpu().numpy())
                val_preds.extend(val_predictions.cpu().numpy())

                nb_val_steps += 1

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = accuracy_score(tr_labels, tr_preds)
        
        val_loss = val_loss / nb_val_steps if nb_val_steps > 0 else 0
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")
        print(f"Validation loss epoch: {val_loss}")
        print(f"Validation accuracy epoch: {val_accuracy}")
        print(f"Validation recall epoch: {val_recall}")
        print(f"Validation F1 epoch: {val_f1}")

        return epoch_loss, tr_accuracy, val_loss, val_accuracy, val_recall, val_f1


    def testLabeledData(self, test_loader, device, id_to_BOI):
        self.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):

                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)
                labels = batch["label"].to(device, dtype=torch.long)

                outputs = self(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs[0]
                eval_logits = outputs[1]

                eval_loss += loss.item()

                nb_eval_steps += 1
                nb_eval_examples += labels.size(0)

                if idx % 100 == 0:
                    loss_step = eval_loss / nb_eval_steps
                    print(f"Validation loss per 100 evaluation steps: {loss_step}")

                flattened_targets = labels.view(-1)  
                active_logits = eval_logits.view(
                    -1, self.num_labels
                )  
                flattened_predictions = torch.argmax(
                    active_logits, axis=1
                )  

                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100  

                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(
                    flattened_predictions, active_accuracy
                )

                eval_labels.extend(labels)
                eval_preds.extend(predictions)

                tmp_eval_accuracy = accuracy_score(
                    labels.cpu().numpy(), predictions.cpu().numpy()
                )
                eval_accuracy += tmp_eval_accuracy

        labels = [id_to_BOI[id.item()] for id in eval_labels]
        predictions = [id_to_BOI[id.item()] for id in eval_preds]

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print(f"Loss: {eval_loss}")
        print(f"Accuracy: {eval_accuracy}")

        return labels, predictions

    def predict(self, dataloader, device, id_to_BOI):
        self.eval()

        pred_array = []
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                ids = batch["input_ids"].to(device, dtype=torch.long)
                mask = batch["attention_mask"].to(device, dtype=torch.long)

                outputs = self(input_ids=ids, attention_mask=mask)
                eval_logits = outputs[0]

                pred_array.append(
                    torch.argmax(eval_logits, axis=2)
                )  # shape (batch_size * seq_len,)

        for i in range(len(pred_array)):
            for pred in pred_array[i].cpu().numpy():
                predictions.append([id_to_BOI[id.item()] for id in pred])

        return predictions

# 处理带有标签的 NER 数据集
class NewNerLabeledDataset(Dataset):
    def __init__(self, data, tokenizer, BOI_to_id, max_len=256):   # entities

        self.max_len = max_len
        self.BOI_to_id = BOI_to_id

        IOBs = toBertNer(data, tokenizer, padding=True, max_len=self.max_len)   # entities
        self.len = len(IOBs["sentence"])

        self.sentences = IOBs["sentence"]
        self.labels = IOBs["labels"]
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sentence = self.sentences[index].strip()
        # 过滤掉特殊标记如 "[CLS]" 和 "[SEP]"
        labels = [label for label in self.labels[index] if label != '[CLS]' and label != '[SEP]']
        encoded_labels = [self.BOI_to_id[iob] for iob in labels]
        encoding = self.tokenizer(
            sentence, padding="max_length", max_length=self.max_len
        )
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        return item

    # def __getitem__(self, index):
    #     sentence = self.sentences[index].strip()
    #     encoded_labels = [self.BOI_to_id[iob] for iob in self.labels[index]]
    #     encoding = self.tokenizer(
    #         sentence, padding="max_length", max_length=self.max_len
    #     )
    #     item = {key: torch.as_tensor(val) for key, val in encoding.items()}
    #     item["labels"] = torch.as_tensor(encoded_labels)
    #     return item

    def __len__(self):
        return self.len
# 处理带有标签的 NER 数据集
class NerLabeledDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, BOI_to_id):

        self.max_len = max_len
        self.BOI_to_id = BOI_to_id

        IOBs = data.get_token_entities(tokenizer, padding=True, max_len=self.max_len)
        self.len = len(IOBs["sentences"])

        self.sentences = IOBs["sentences"]
        self.labels = IOBs["labels"]
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        sentence = self.sentences[index].strip()
        encoded_labels = [self.BOI_to_id[iob] for iob in self.labels[index]]

        encoding = self.tokenizer(
            sentence, padding="max_length", max_length=self.max_len
        )

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len

# 处理不带标签的 NER 数据集
class NerUnlabeledDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):

        self.max_len = max_len
        self.len = len(data)
        self.sentences = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):

        sentence = self.sentences[index].strip()

        encoding = self.tokenizer(
            preprocess_text(sentence),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["plain"] = (
            sentence.replace("$", "")
            .replace("_", "")
            .replace("}", "")
            .replace("{", "")
            .replace("~", "")
            .replace("\\", "")
        )

        return item

    def __len__(self):
        return self.len

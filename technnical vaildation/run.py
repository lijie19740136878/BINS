import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from torchcrf import CRF
from seqeval.metrics import classification_report
class CommonConfig:
    bert_dir = "./model_hub/bert-based-cased/"
    output_dir = "./checkpoint/"
    data_dir = "./data/"

class NerConfig:
    def __init__(self, data_name):
        path = CommonConfig()
        self.bert_dir = path.bert_dir
        self.output_dir = path.output_dir
        self.output_dir = os.path.join(self.output_dir, data_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_dir = path.data_dir

        self.data_path = os.path.join(os.path.join(self.data_dir, data_name), "ner_data")
        with open(os.path.join(self.data_path, "labels.txt"), "r",encoding='utf-8', errors='ignore') as fp :
            self.labels = fp.read().strip().split("\n")
        self.bio_labels = ["O"]
        for label in self.labels:
            self.bio_labels.append("B-{}".format(label))
            self.bio_labels.append("I-{}".format(label))
        self.num_labels = len(self.bio_labels)
        self.label2id = {label: i for i, label in enumerate(self.bio_labels)}
        self.id2label = {i: label for i, label in enumerate(self.bio_labels)}

        self.max_seq_len =256
        self.epochs = 1
        self.train_batch_size = 8
        self.dev_batch_size =8
        self.bert_learning_rate = 0.00001
        self.crf_learning_rate = 0.0001
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.005
        self.save_step = 500

class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):    
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        try:  
            text = self.data[item]["text"]  
        except KeyError:  
            print(f"KeyError occurred for index {item}. Skipping this item.") 
        labels = self.data[item]["labels"]
        if len(text) > self.max_seq_len - 2:
            print(f"This sentence is{self.data[item]}")
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data
class ReDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ModelOutput:  
    def __init__(self, logits, labels, loss=None):  
        self.logits = logits  
        self.labels = labels  
        self.loss = loss  
class BertNer(nn.Module):
  def __init__(self, args):
    super(BertNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    input_size = self.bert_config.hidden_size
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.LSTM(input_size, self.lstm_hiden,  num_layers=2, bidirectional=True, batch_first=True,dropout=0.2)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    seq_out = bert_output[0] 
    batch_size = seq_out.size(0)
    seq_out, _ = self.bilstm(seq_out)
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output



class BertRe(nn.Module):
  def __init__(self, args):
    super(BertRe, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    self.hidden_size = self.bert_config.hidden_size
    self.linear = nn.Linear(self.hidden_size, args.num_labels)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self,
         input_ids,
         attention_mask,
         token_type_ids,
         labels=None):
    bert_output = self.bert(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids)
    pooled_output  = bert_output[1]  
    logits  = self.linear(pooled_output )
    loss = None

    if labels is not None:
      loss = self.criterion(logits, labels)
    model_output = ModelOutput(logits, labels, loss)
    return model_output


class ReCollate:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def collate(self, batch_data):
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []
        for d in batch_data:
            text = d["text"]
            labels = d["labels"]
            h = labels[0] 
            t = labels[1]
            label = labels[2]
            pre_length = 4 + len(h) + len(t)
            if len(text) > self.max_seq_len - pre_length:
                text = text[:self.max_seq_len - pre_length]
            if h not in text or t not in text:
                continue
            tmp_input_ids = self.tokenizer.tokenize("[CLS]" + h + "[SEP]" + t + "[SEP]" + text + "[SEP]")
            tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_input_ids)
            attention_mask = [1] * len(tmp_input_ids)
            input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
            attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
            token_type_ids = [0] * self.max_seq_len
            label = self.label2id[label]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(label)
        if len(batch_input_ids) == 0:
            print("Warning: Batch is empty after processing.")
            return None  
        input_ids = torch.tensor(np.array(batch_input_ids))
        attention_mask = torch.tensor(np.array(batch_attention_mask))
        token_type_ids = torch.tensor(np.array(batch_token_type_ids))
        labels = torch.tensor(np.array(batch_labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return data


class ReConfig:
    def __init__(self, data_name):
        cf = CommonConfig()
        self.bert_dir = cf.bert_dir
        self.output_dir = cf.output_dir
        self.output_dir = os.path.join(self.output_dir, data_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.data_dir = cf.data_dir

        self.data_path = os.path.join(os.path.join(self.data_dir, data_name), "re_data")
        with open(os.path.join(self.data_path, "labels.txt"), "r",encoding='utf-8', errors='ignore') as fp:
            self.labels = fp.read().strip().split("\n")

        self.num_labels = len(self.labels)
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        print(self.label2id)
        self.id2label = {i: label for i, label in enumerate(self.labels)}

        self.max_seq_len = 256
        self.epochs = 1
        self.train_batch_size = 8
        self.dev_batch_size = 8
        self.learning_rate = 3e-5
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.01
        self.save_step = 500

class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 save_step=500,
                 dev_loader=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="cpu",
                 id2label=None):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(self.train_loader) * self.epochs

    def train(self):
        global_step = 1
        loss_values = []
        glasb_step_values = []
        for epoch in range(1, self.epochs + 1):
            with tqdm(total=self.total_step, desc=f"Epoch {epoch}/{self.epochs}", unit="step") as pbar:
                for step, batch_data in enumerate(self.train_loader):
                    self.model.train()
                    for key, value in batch_data.items():
                        batch_data[key] = value.to(self.device)
                    input_ids = batch_data["input_ids"]
                    attention_mask = batch_data["attention_mask"]
                    labels = batch_data["labels"]
                    output = self.model(input_ids, attention_mask, labels.to(torch.int64))
                    loss = output.loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.schedule.step()
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    if global_step % 10 == 0 and global_step != 0:
                        print(f"\n【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{loss.item()} ")
                        if loss.item() < 20:
                            loss=f"{loss.item():.2f}"
                            loss_values.append(loss)
                            glasb_step_values.append(global_step)
                    if global_step % self.save_step == 0:
                        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))
                    global_step += 1
                    torch.cuda.empty_cache()
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "pytorch_model_ner.bin")))
        self.model.eval()
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            output = self.model(input_ids, attention_mask,labels.to(torch.int64))
            logits = output.logits
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]
                # print(f"logit:{logit}")
                logit = [self.id2label[i] for i in logit]
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                preds.append(logit)
                trues.append(label)
        report = classification_report(trues, preds)
        return report

def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )
    return optimizer, scheduler

def main(data_name):
    args = NerConfig(data_name)
    with open(os.path.join(args.output_dir, "ner_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_path, "train.txt"), "r" ,encoding='utf-8') as fp:
        train_data = fp.read().split("\n")
    train_data = [json.loads(d) for d in train_data]

    with open(os.path.join(args.data_path, "dev.txt"), "r", encoding='utf-8') as fp:
        dev_data = fp.read().split("\n")
    dev_data = [json.loads(d) for d in dev_data]

    train_dataset = NerDataset(train_data, args, tokenizer)
    dev_dataset = NerDataset(dev_data, args, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2)
    model = BertNer(args)
    model.to(device)
    t_total = len(train_loader) * args.epochs
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_total)

    trainer = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label
    )
    trainer.train()
    out_index = trainer.test()
    print(out_index)
if __name__ == "__main__":
    data_name = "data_name"
    main(data_name)

import json
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, classification_report

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def convert_to_spacy_format(data):
    spacy_data = []
    for item in data:
        text = item["text"]
        labels = item["labels"]
        entities = []
        start = 0
        for word, label in zip(text, labels):
            end = start + len(word)
            if label != "O":
                entity = (start, end, label)
                entities.append(entity)
            start = end + 1
        spacy_data.append((" ".join(text), {"entities": entities}))
    return spacy_data

def train_and_evaluate(dataset_name):
    # 自动生成文件路径
    train_file = f"data/{dataset_name}/ner_data/train.txt"
    dev_file = f"data/{dataset_name}/ner_data/dev.txt"
    model_save_path = f"checkpoint/{dataset_name}/debert-v3-ner_model"

    # 加载和转换数据
    trains_data = load_data(train_file)
    devs_data = load_data(dev_file)
    train_data = convert_to_spacy_format(trains_data)
    eval_data = convert_to_spacy_format(devs_data)

    # 加载预训练的语言模型
    nlp = spacy.load("en_core_web_sm")

    # 创建NER管道，如果已经存在则获取
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # 添加标签到NER
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # 使用CuPy（如果可用）来创建优化器
    if spacy.prefer_gpu():
        spacy.require_gpu()
        optimizer = nlp.create_optimizer(use_gpu=True)
    else:
        optimizer = nlp.create_optimizer()

    # 设置超参数
    optimizer.learn_rate = 0.0001  # 学习率

    # 定义训练参数
    num_epochs = 3  # 训练轮数
    batch_size = 16  # 批处理大小
    dropout = 0.4

    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()
        losses = {}
        # 使用minibatch进行训练
        batches = list(minibatch(train_data, size=compounding(4.0, batch_size, 1.001)))
        total_loss = 0.0
        num_batches = len(batches)
        with tqdm(total=num_batches, desc=f"Training Epoch {epoch + 1}") as pbar:
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in zip(texts, annotations)]
                nlp.update(examples, drop=dropout, losses=losses, sgd=optimizer)
                # 累加总损失
                total_loss += losses['ner']
                pbar.update(1)
                pbar.set_postfix(loss=losses['ner'])
        end_time = time.time()
        epoch_duration = end_time - start_time
        # 归一化损失值
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds. Average Training Loss: {avg_loss}")

        # 计算验证集损失
        eval_loss = 0.0
        eval_batches = list(minibatch(eval_data, size=batch_size))
        for batch in eval_batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in zip(texts, annotations)]
            ner.update(examples, drop=0.0, losses=losses)  # 评估时不使用dropout
            eval_loss += losses['ner']
        eval_loss /= len(eval_batches)
        eval_losses.append(eval_loss)
        print(f"Epoch {epoch + 1} completed. Average Evaluation Loss: {eval_loss}")

    # 保存模型
    nlp.to_disk(model_save_path)

    # 加载训练好的模型
    nlp = spacy.load(model_save_path)

    # 评估模型
    def evaluate_model(data):
        y_true = []
        y_pred = []
        labels = set()
        excluded_labels = {"ORG", "LAW", "NORP", "ORDINAL","GPE","PERCENT"}

        for text, annotations in data:
            doc = nlp(text)
            true_entities = [entity for entity in annotations["entities"] if entity[2] not in excluded_labels]
            pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents if ent.label_ not in excluded_labels]

            y_true.extend([entity[2].split('-')[-1] for entity in true_entities])
            y_pred.extend([entity[2].split('-')[-1] for entity in pred_entities])
            labels.update([entity[2].split('-')[-1] for entity in true_entities + pred_entities])

            # 对于未检测到的实体，预测标签为 'O'
            if len(true_entities) > len(pred_entities):
                y_pred.extend(['O'] * (len(true_entities) - len(pred_entities)))
            elif len(pred_entities) > len(true_entities):
                y_true.extend(['O'] * (len(pred_entities) - len(true_entities)))

        # 计算每个实体类型的评价值
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(labels))

        # 计算宏观平均和微观平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

        # 输出每个实体类型的评价值
        print("Entity Type Evaluation:")
        for label, p, r, f, s in zip(labels, precision, recall, f1, support):
            print(f"{label}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, Support: {s}")

        # 输出宏观平均和微观平均
        print("\nMacro Average: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(macro_precision, macro_recall, macro_f1))
        print("Micro Average: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(micro_precision, micro_recall, micro_f1))

        # 输出分类报告
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=list(labels)))

    # 使用验证集进行评估
    evaluate_model(eval_data)

# 调用函数进行训练和评估
dataset_name = "volu"  # 可以根据需要修改数据集名称
train_and_evaluate(dataset_name)

# import json
# import torch
# from torch.utils.data import DataLoader, Dataset
# from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification, AdamW, get_scheduler
# from datasets import load_metric
# from sklearn.metrics import precision_recall_fscore_support
# from tqdm import tqdm

# dataset_name = "normal" 
# train_file = f"data/{dataset_name}/ner_data/train.txt"
# dev_file = f"data/{dataset_name}/ner_data/dev.txt"
# model_save_path = f"checkpoint/{dataset_name}/deberta-v3-ner_model"
# local_model_path = r"model_hub\deberta-v3-large"

# def load_data(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         data = [json.loads(line) for line in file]
#     return data

# train_data = load_data(train_file)
# dev_data = load_data(dev_file)


# def convert_to_transformers_format(data):
#     tokens = []
#     labels = []
#     for item in data:
#         tokens.append(item["text"])
#         labels.append(item["labels"])
#     return tokens, labels

# train_tokens, train_labels = convert_to_transformers_format(train_data)
# dev_tokens, dev_labels = convert_to_transformers_format(dev_data)


# unique_labels = set(label for labels in train_labels for label in labels)
# label_to_id = {label: i for i, label in enumerate(sorted(unique_labels))}
# id_to_label = {i: label for label, i in label_to_id.items()}

# class NERDataset(Dataset):
#     def __init__(self, tokens, labels, tokenizer, label_to_id, max_length=128):
#         self.tokens = tokens
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.label_to_id = label_to_id
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.tokens)

#     def __getitem__(self, idx):
#         tokens = self.tokens[idx]
#         labels = self.labels[idx]
#         tokenized_inputs = self.tokenizer(tokens, truncation=True, is_split_into_words=True, padding="max_length", max_length=self.max_length)
#         word_ids = tokenized_inputs.word_ids()

#         label_ids = []
#         previous_word_idx = None
#         for word_idx in word_ids:
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:
#                 label_ids.append(self.label_to_id[labels[word_idx]])
#             else:
#                 label_ids.append(-100)
#             previous_word_idx = word_idx

#         tokenized_inputs["labels"] = label_ids
#         return {key: torch.tensor(val) for key, val in tokenized_inputs.items()}


# tokenizer = DebertaV2TokenizerFast.from_pretrained(local_model_path)
# model = DebertaV2ForTokenClassification.from_pretrained(local_model_path, num_labels=len(unique_labels))


# train_dataset = NERDataset(train_tokens, train_labels, tokenizer, label_to_id)
# dev_dataset = NERDataset(dev_tokens, dev_labels, tokenizer, label_to_id)

# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# dev_dataloader = DataLoader(dev_dataset, batch_size=8)


# optimizer = AdamW(model.parameters(), lr=2e-5)
# num_training_steps = len(train_dataloader) * 3  
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)


# model.train()
# progress_bar = tqdm(range(num_training_steps))

# for epoch in range(3):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)


# model.save_pretrained(model_save_path)
# tokenizer.save_pretrained(model_save_path)


# metric = load_metric("seqeval")

# model.eval()
# for batch in dev_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     predictions = outputs.logits.argmax(dim=-1)
#     labels = batch["labels"]

#     true_labels = [[id_to_label[l.item()] for l in label if l.item() != -100] for label in labels]
#     true_predictions = [[id_to_label[p.item()] for p, l in zip(prediction, label) if l.item() != -100] for prediction, label in zip(predictions, labels)]

#     metric.add_batch(predictions=true_predictions, references=true_labels)

# results = metric.compute()
# print(f"Precision: {results['overall_precision']:.4f}")
# print(f"Recall: {results['overall_recall']:.4f}")
# print(f"F1 Score: {results['overall_f1']:.4f}")
# print(f"Accuracy: {results['overall_accuracy']:.4f}")


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

    train_file = f"data/{dataset_name}/ner_data/train.txt"
    dev_file = f"data/{dataset_name}/ner_data/dev.txt"
    model_save_path = f"checkpoint/{dataset_name}/debert-v3-ner_model"

    trains_data = load_data(train_file)
    devs_data = load_data(dev_file)
    train_data = convert_to_spacy_format(trains_data)
    eval_data = convert_to_spacy_format(devs_data)
    
    nlp = spacy.load("file")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    if spacy.prefer_gpu():
        spacy.require_gpu()
        optimizer = nlp.create_optimizer(use_gpu=True)
    else:
        optimizer = nlp.create_optimizer()

    optimizer.learn_rate = 0.0001  #
    num_epochs = 3  
    batch_size = 16  
    dropout = 0.4
    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()
        losses = {}
        batches = list(minibatch(train_data, size=compounding(4.0, batch_size, 1.001)))
        total_loss = 0.0
        num_batches = len(batches)
        with tqdm(total=num_batches, desc=f"Training Epoch {epoch + 1}") as pbar:
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in zip(texts, annotations)]
                nlp.update(examples, drop=dropout, losses=losses, sgd=optimizer)
    
                total_loss += losses['ner']
                pbar.update(1)
                pbar.set_postfix(loss=losses['ner'])
        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds. Average Training Loss: {avg_loss}")

        eval_loss = 0.0
        eval_batches = list(minibatch(eval_data, size=batch_size))
        for batch in eval_batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in zip(texts, annotations)]
            ner.update(examples, drop=0.0, losses=losses) 
            eval_loss += losses['ner']
        eval_loss /= len(eval_batches)
        eval_losses.append(eval_loss)
        print(f"Epoch {epoch + 1} completed. Average Evaluation Loss: {eval_loss}")

    nlp.to_disk(model_save_path)
    nlp = spacy.load(model_save_path)
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

            if len(true_entities) > len(pred_entities):
                y_pred.extend(['O'] * (len(true_entities) - len(pred_entities)))
            elif len(pred_entities) > len(true_entities):
                y_true.extend(['O'] * (len(pred_entities) - len(true_entities)))

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(labels))

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

        print("Entity Type Evaluation:")
        for label, p, r, f, s in zip(labels, precision, recall, f1, support):
            print(f"{label}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, Support: {s}")


        print("\nMacro Average: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(macro_precision, macro_recall, macro_f1))
        print("Micro Average: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(micro_precision, micro_recall, micro_f1))

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=list(labels)))

    evaluate_model(eval_data)

dataset_name = "name"  
train_and_evaluate(dataset_name)

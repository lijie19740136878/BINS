# -*- coding: utf-8 -*-
import os
import re
import json
import codecs
import random
import codecs
from tqdm import tqdm
from collections import defaultdict

class ProcessnormalData:
    def __init__(self):
        self.data_path = "./data/normal/"
        self.train_file = self.data_path + "ori_data/train.json"
        self.dev_file = self.data_path + "ori_data/dev.json"
        self.test_file = self.data_path + "ori_data/test2.json"
        self.schema_file = self.data_path + "ori_data/schema.json"

    def get_predicate(self):
        rels = set()

        with open(self.schema_file, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                data = eval(line)
                rels.add(data['predicate'])
                

        with open(self.data_path + "re_data/labels.txt", 'w', encoding="utf-8") as fp:
            fp.write("\n".join(["No predicate"] + list(rels)))
    def get_ents(self):
        ents = set()
        rels = defaultdict(list)
        with open(self.schema_file, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                data = eval(line)
                subject_type = data['subject_type']['@value'] if '@value' in data['subject_type'] else data[
                    'subject_type']
                object_type = data['object_type']['@value'] if '@value' in data['object_type'] else data['object_type']
                ents.add(subject_type)
                ents.add(object_type)
                predicate = data["predicate"]
                rels[subject_type + "_" + object_type].append(predicate)

        with open(self.data_path + "ner_data/labels.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(list(ents)))

        with open(self.data_path + "re_data/rels.txt", "w", encoding="utf-8") as fp:
            json.dump(rels, fp, ensure_ascii=False, indent=2)

    def get_ner_data(self, input_file, output_file):
        res = []
        with codecs.open(input_file, 'r', encoding="utf-8", errors="replace") as fp:
            lines = fp.read().strip().split("\n")
            for i, line in enumerate(tqdm(lines)):
                try:
                    line = eval(line)
                except Exception as e:
                    continue
                tmp = {}
                text = line['text']
                words = text.split()
                tmp['text'] = words
                tmp["labels"] = ["O"] * len(words)
                tmp['id'] = i
                spo_list = line['spo_list']
                for j, spo in enumerate(spo_list):
                    if spo['subject'] == "" or spo['object']['@value'] == "":
                        continue
                    subject = spo['subject']
                    object_ = spo['object']['@value']
                    subject_type = spo["subject_type"]
                    object_type = spo['object_type']['@value']
                    
                    # Find subject in words
                    subject_words = subject.split()
                    subject_len = len(subject_words)
                    for idx in range(len(words) - subject_len + 1):
                        if words[idx:idx + subject_len] == subject_words:
                            tmp["labels"][idx] = f"B-{subject_type}"
                            for k in range(1, subject_len):
                                tmp["labels"][idx + k] = f"I-{subject_type}"
                    
                    object_words = object_.split()
                    object_len = len(object_words)
                    for idx in range(len(words) - object_len + 1):
                        if words[idx:idx + object_len] == object_words:
                            tmp["labels"][idx] = f"B-{object_type}"
                            for k in range(1, object_len):
                                tmp["labels"][idx + k] = f"I-{object_type}"
                res.append(tmp)

        with open(output_file, 'w', encoding="utf-8") as fp:
            fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))

    def get_re_data(self, input_file, output_file):
        res = []

        with codecs.open(input_file, 'r', encoding="utf-8", errors="replace") as fp:
            lines = fp.read().strip().split("\n")
            for i, line in enumerate(lines):
                try:
                    line = eval(line)
                except Exception as e:
                    continue
                tmp = {}
                text = line['text']
                tmp['text'] = text
                tmp['id'] = i
                spo_list = line['spo_list']

                ent_rel_dict = defaultdict(list)
                sub_obj = []  
                for j, spo in enumerate(spo_list):
                    if spo['subject'] == "" or spo['object']['@value'] == "":
                        continue
                    sbj = spo['subject']
                    obj = spo['object']['@value']
                    tmp["labels"] = [sbj, obj, spo["predicate"]]
                    sub_obj.append((sbj, obj))
                    ent_rel_dict[spo["predicate"]].append((sbj, obj))
                    res.append(tmp)
            
                for k, v in ent_rel_dict.items():
                    sbjs = list(set([p[0] for p in v]))
                    objs = list(set([p[1] for p in v]))
                    if len(sbjs) > 1 and len(objs) > 1:
                        neg_total = 3
                        neg_cur = 0
                        for sbj in sbjs:
                            random.shuffle(objs)
                            for obj in objs:
                                if (sbj, obj) not in sub_obj:
                                    tmp["id"] = str(i) + "_" + "norel"
                                    tmp["labels"] = [sbj, obj, "No predicate"]
                                    res.append(tmp)
                                    neg_total += 1
                                break
                            if neg_cur == neg_total:
                                break

        with open(output_file, 'w') as fp:
            fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))


class ProcesscamiData:
    def __init__(self):
        self.data_path = "./data/cami/"
        self.train_file = self.data_path + "ori_data/train.json"
        self.dev_file = self.data_path + "ori_data/dev.json"
        self.test_file = self.data_path + "ori_data/test2.json"
        self.schema_file = self.data_path + "ori_data/schema.json"

    def get_predicate(self):
        rels = set()

        with open(self.schema_file, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                data = eval(line)
                rels.add(data['predicate'])
                

        with open(self.data_path + "re_data/labels.txt", 'w', encoding="utf-8") as fp:
            fp.write("\n".join(["No predicate"] + list(rels)))

    def get_ents(self):
        ents = set()
        rels = defaultdict(list)
        with open(self.schema_file, 'r', encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                data = eval(line)
                subject_type = data['subject_type']['@value'] if '@value' in data['subject_type'] else data[
                    'subject_type']
                object_type = data['object_type']['@value'] if '@value' in data['object_type'] else data['object_type']
                ents.add(subject_type)
                ents.add(object_type)
                predicate = data["predicate"]
                rels[subject_type + "_" + object_type].append(predicate)

        with open(self.data_path + "ner_data/labels.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(list(ents)))

        with open(self.data_path + "re_data/rels.txt", "w", encoding="utf-8") as fp:
            json.dump(rels, fp, ensure_ascii=False, indent=2)

    def get_ner_data(self, input_file, output_file):
        res = []
        with codecs.open(input_file, 'r', encoding="utf-8", errors="replace") as fp:
            lines = fp.read().strip().split("\n")
            for i, line in enumerate(tqdm(lines)):
                try:
                    line = eval(line)
                except Exception as e:
                    continue
                tmp = {}
                text = line['text']
                words = text.split()
                tmp['text'] = words
                tmp["labels"] = ["O"] * len(words)
                tmp['id'] = i
                spo_list = line['spo_list']
                for j, spo in enumerate(spo_list):
                    if spo['subject'] == "" or spo['object']['@value'] == "":
                        continue
                    subject = spo['subject']
                    object_ = spo['object']['@value']
                    subject_type = spo["subject_type"]
                    object_type = spo['object_type']['@value']
                    
                    # Find subject in words
                    subject_words = subject.split()
                    subject_len = len(subject_words)
                    for idx in range(len(words) - subject_len + 1):
                        if words[idx:idx + subject_len] == subject_words:
                            tmp["labels"][idx] = f"B-{subject_type}"
                            for k in range(1, subject_len):
                                tmp["labels"][idx + k] = f"I-{subject_type}"
                    
                    # Find object in words
                    object_words = object_.split()
                    object_len = len(object_words)
                    for idx in range(len(words) - object_len + 1):
                        if words[idx:idx + object_len] == object_words:
                            tmp["labels"][idx] = f"B-{object_type}"
                            for k in range(1, object_len):
                                tmp["labels"][idx + k] = f"I-{object_type}"
                res.append(tmp)

        with open(output_file, 'w', encoding="utf-8") as fp:
            fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))

    def get_re_data(self, input_file, output_file):
        res = []

        with codecs.open(input_file, 'r', encoding="utf-8", errors="replace") as fp:
            lines = fp.read().strip().split("\n")
            for i, line in enumerate(lines):
                try:
                    line = eval(line)
                except Exception as e:
                    continue
                tmp = {}
                text = line['text']
                tmp['text'] = text
                tmp['id'] = i
                spo_list = line['spo_list']

                ent_rel_dict = defaultdict(list)
                sub_obj = []  # 
                for j, spo in enumerate(spo_list):
                    if spo['subject'] == "" or spo['object']['@value'] == "":
                        continue
                    sbj = spo['subject']
                    obj = spo['object']['@value']
                    tmp["labels"] = [sbj, obj, spo["predicate"]]
                    sub_obj.append((sbj, obj))
                    ent_rel_dict[spo["predicate"]].append((sbj, obj))
                    res.append(tmp)
                
                for k, v in ent_rel_dict.items():
                    sbjs = list(set([p[0] for p in v]))
                    objs = list(set([p[1] for p in v]))
                    if len(sbjs) > 1 and len(objs) > 1:
                        neg_total = 3
                        neg_cur = 0
                        for sbj in sbjs:
                            random.shuffle(objs)
                            for obj in objs:
                                if (sbj, obj) not in sub_obj:
                                    tmp["id"] = str(i) + "_" + "norel"
                                    tmp["labels"] = [sbj, obj, "No predicate"]
                                    res.append(tmp)
                                    neg_total += 1
                                break
                            if neg_cur == neg_total:
                                break

        with open(output_file, 'w') as fp:
            fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))

if __name__ == "__main__":

    ProcessnormalData = ProcessnormalData()
    ProcessnormalData.get_ents()
    ProcessnormalData.get_predicate()
    ProcessnormalData.get_ner_data(ProcessnormalData.train_file,
                                os.path.join(ProcessnormalData.data_path, "ner_data/train.txt"))
    ProcessnormalData.get_ner_data(ProcessnormalData.dev_file, os.path.join(ProcessnormalData.data_path, "ner_data/dev.txt"))
    ProcessnormalData.get_re_data(ProcessnormalData.train_file,
                                os.path.join(ProcessnormalData.data_path, "re_data/train.txt"))
    ProcessnormalData.get_re_data(ProcessnormalData.dev_file, os.path.join(ProcessnormalData.data_path, "re_data/dev.txt"))


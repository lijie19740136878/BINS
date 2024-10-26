import torch.nn as nn  
from torchcrf import CRF  
from transformers import BertForTokenClassification, BertTokenizerFast  
class ModelOutput:  
    def __init__(self, logits, labels, loss=None):  
        self.logits = logits  
        self.labels = labels  
        self.loss = loss  
class BertNer(nn.Module):  
    def __init__(self, args):  
        super(BertNer, self).__init__()  
        self.num_labels = args.num_labels  
        self.bert = BertForTokenClassification.from_pretrained(args.bert_dir,num_labels=19)  
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)  
        self.crf = CRF(self.num_labels, batch_first=True)  

    def forward(self, input_ids, attention_mask, labels=None): 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits  
        
        loss=None
        if labels is not None: 
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')  
        logits = self.crf.decode(logits, mask=attention_mask.bool())  
        model_output = ModelOutput(logits, labels, loss)  

        return model_output
from torch import nn
from transformers import BertModel

class BERT_Model(nn.Module):
    """ A Model for bert training """

    def __init__(self, bert_config='bert-base-uncased', n_class=None, dropout=0.2):
        super(BERT_Model, self).__init__()
        self.bert_config = bert_config
        self.bert = BertModel.from_pretrained(self.bert_config)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, n_class) # 1024
        self.relu = nn.ReLU()

    def forward(self, ids, mask, token_type_ids):

        output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids)

        dropout_output = self.dropout(output['pooler_output'])
        linear_output = self.linear(dropout_output)

        return self.relu(linear_output)


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastai import *
from fastai.text import *
from fastai.callbacks import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

from pytorch_transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
from pytorch_transformers import AdamW


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


# adapt this to your needs!
config_data = Config(
    root_folder='.',  # where is the root folder? Keep it that way if you want to load from Google Drive
    model_path='/models/',  # where is the folder for the model(s); relative to the root
    model_name='NoRBERT_Task1_NFR_e10_NoSampling.pkl',  # what is the model name?
)


# Load/Define NoRBERT classes {display-mode: "form"}
class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int = 512, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str):
        """Limits the maximum sequence length. Prepend with [CLS] and append [SEP]"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]


##

class BertTokenizeProcessor(TokenizeProcessor):
    """Special Tokenizer, where we remove sos/eos tokens since we add that ourselves in the tokenizer."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)


class BertNumericalizeProcessor(NumericalizeProcessor):
    """Use a custom vocabulary to match the original BERT model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)


def get_bert_processor(tokenizer: Tokenizer = None, vocab: Vocab = None):
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]


class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path: PathOrStr, train_df: DataFrame, valid_df: DataFrame, test_df: Optional[DataFrame] = None,
                tokenizer: Tokenizer = None, vocab: Vocab = None, classes: Collection[str] = None,
                text_cols: IntsOrStrs = 1,
                label_cols: IntsOrStrs = 0, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls == TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)


##

class BertTextClassifier(BertPreTrainedModel):
    def __init__(self, model_name, num_labels):
        config = BertConfig.from_pretrained(model_name)
        super(BertTextClassifier, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(model_name, config=config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # self.apply(self.init_weights)

    def forward(self, tokens, labels=None, position_ids=None, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(tokens, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        pooled_output = outputs[1]
        # According to documentation of pytorch-transformers, pooled output might not be the best
        # and you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence
        # hidden_states = outputs[0]
        # pooled_output = torch.mean(hidden_states, 1)

        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)

        activation = nn.Softmax(dim=1)
        probs = activation(logits)

        return logits


# classifier = load_learner(config_data.root_folder + config_data.model_path, config_data.model_name)

# predict
def predict(df):
    config_data = Config(
        root_folder='.',  # where is the root folder? Keep it that way if you want to load from Google Drive
        model_path='/models/',  # where is the folder for the model(s); relative to the root
        model_name='NoRBERT_Task1_NFR_e10_NoSampling.pkl',  # what is the model name?
    )
    classifier = load_learner(config_data.root_folder + config_data.model_path, config_data.model_name)
    prediction = classifier.predict(df['RequirementText'])
    prediction_class = prediction[1]
    label = classifier.data.classes[prediction_class]
    labels = ['F', 'NF']
    return labels[label]

dataset = pd.read_csv('./data/dataset.csv', sep=',', header=0)
#dataset['RequirementText'] = dataset['US_title'].str.replace('.*I want to', '')
dataset['predClass'] = dataset.apply(predict, axis = 1)
dataset.to_csv('./data/datasetClassified.csv', index=False)


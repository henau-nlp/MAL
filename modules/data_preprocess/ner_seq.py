""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True, ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    # leng = 10
    i=0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if isinstance(example.text_a, list):
            example.text_a = " ".join(example.text_a)
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        #print(len(label_ids))
        # if len(label_ids) > leng:
        #     leng = len(label_ids)
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
        #print('这个是tokens',tokens,'这个是tokens的长度',len(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #print("输出ids",input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #print(len(input_ids),len(label_ids))
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        # print(len(input_ids))
        # print(len(label_ids))
        # print(padding_length)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # if(len(input_ids) != len(label_ids)):
        #     print(example.text_a)
        #     print(example.guid)

        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            # logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            # logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            # logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            # logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))
        # print(leng)
    return features


class JybProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, datapool):
        """See base class."""
        return self._create_examples(datapool, "dlab")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples2(self._read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_eval_examples(self, datapool):
        """See base class."""
        return self._create_examples1(datapool, "dloop")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples2(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["[PAD]", 'O', 'B-Cause', 'B-Degree', 'B-Holder', 'B-Negation', 'B-Property', 'B-Trigger', 'B-compared_Entity',
                'B-sent_Entity', 'I-Cause', 'I-Degree', 'I-Holder', 'I-Negation', 'I-Property', 'I-Trigger',
                'I-compared_Entity', 'I-sent_Entity',  "[CLS]", "[SEP]"]

        # return ["[PAD]", 'O', 'B-LOC', 'B-PER', 'B-ORG', 'I-LOC', 'I-PER', 'I-ORG', "[CLS]", "[SEP]"]

    def _create_examples(self, datapool, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        words, label_list, = datapool.get_annotated_data()
        for i in range(len(words)):
            guid = "%s-%s" % (set_type, i)
            text_a = words[i]
            # BIOS
            labels = label_list[i]
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples
    def _create_examples1(self, datapool, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        words, label_list, = datapool.get_unannotated_data()
        for i in range(len(words)):
            guid = "%s-%s" % (set_type, i)
            text_a = words[i]
            # BIOS
            labels = label_list[i]
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


    def _create_examples2(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


ner_processors = {
    "jyb": JybProcessor
}

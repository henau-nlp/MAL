import os
import sys
import random
import time

from transformers import BertConfig, BertTokenizer

# sys.path.append(r"C:\Users\Zhang_guipei\Desktop\jyb-model-master\PT_AL_BERT_CRF")

from modules.model_train.bert import BertCrfForNer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.utils import vec_to_tags
from utils.tools import EarlyStopping

sys.path.append("..")

import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# AL-NER-DEMO Modules


from core.pipeline import Pipeline

from modules.data_preprocess.data_loader import Preprocessor
# from modules.data_preprocess.DataPool import DataPool
from modules.assessment.eval_index import EvaluationIndex
from modules.select_strategy.ALStrategy import *
from modules.assessment.sample_metrics import SampleMetrics
from modules.data_preprocess.ner_seq import convert_examples_to_features
from modules.data_preprocess.ner_seq import ner_processors as processors
from modules.data_preprocess.ner_seq import collate_fn

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, BertTokenizer),
}


def setup_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


class BERTCRFALPipeline(Pipeline):
    """
    BERT + CRF + AL
    ===============================
    """

    def __init__(self):
        self.tags = None
        self.labels = None
        self.eval_dl = None
        self.test_dl = None
        self.train_dl = None
        self.model = None
        self.preprocessor = Preprocessor(vocab=[], tags=[])
        self.datapool = None
        self.dlab_clses = None
        self.dlab_probs = None
        # TODO: Complete
        self.strategy = {
            "RANDOM": RandomStrategy,
            "LC": LeastConfidenceStrategy,
            "NLC": NormalizedLeastConfidenceStrategy,
            "LTP": LeastTokenProbabilityStrategy,
            "MTP": MinimumTokenProbabilityStrategy,
            "MTE": MaximumTokenEntropyStrategy,
            "LONG": LongStrategy,
            "TE": TokenEntropyStrategy,
            # "CC": ContrastiveConfidenceStrategy,
            "CAL": ContrastiveConfidenceStrategy111,
            "MAL": MAL,
            "MAL_AblationRandomPartners": MAL_AblationRandomPartners,
            "SS": SubSequence,  # 子序列不确定性
            "SSDS": SubSequenceDiversityStatic,  # 子序列不确定性 + 多样性 + 静态epsilon
            "SSDD": SubSequenceDiversityDynamic,  # 子序列不确定性 + 多样性 + 动态epsilon
            "UDD": UncertaintyDiversityDynamic,  # 不确定性 + 多样性 + 动态epsilon
        }
        super(BERTCRFALPipeline, self).__init__()

    # fix random seed

    def build_dataset(self):
        """
        Step 01
        Building dataset.
        """
        self.logger.info("Step01 Begin: word embedding.\n")

        batch_size = self.config.param("BERT", "batch_size", type="int")
        all_word_embedding_path = self.config.param("WORD2VEC", "all_word_embedding_path", type="filepath")
        courpus_file = self.config.param("BERT", "data_dir", type="string")
        # train_courpus_file = self.config.param("")
        courpus_name = self.config.param("WORD2VEC", "courpus_name", type="string")
        choose_fraction = self.config.param("WORD2VEC", "choose_fraction", type="float")
        embedding_dim = self.config.param("WORD2VEC", "embedding_dim", type="int")
        entity_type = self.config.param("WORD2VEC", "entity_type", type="int")
        max_seq_len = self.config.param("WORD2VEC", "max_seq_len", type="int")
        tags_file = self.config.param("WORD2VEC", "tags_file", type="filepath")
        seed = self.config.param("BERT", "seed", type="int")

        setup_seeds(seed)
        self.tags = [line.strip() for line in open(tags_file, 'r', encoding='utf8').readlines()]
        self.labels = [label for label in self.tags if label not in ['O', '[PAD]', '[CLS]', '[SEP]', 'X']]
        all_words_embeds = pickle.load(open(all_word_embedding_path, 'rb'))
        vocab = list(all_words_embeds.keys())
        self.preprocessor = Preprocessor(vocab=vocab, tags=self.tags)

        self.datapool, self.test_data = self.preprocessor.load_dataset_init(courpus_file,
                                                                            courpus_name,
                                                                            entity_type,
                                                                            choose_fraction,
                                                                            max_seq_len,
                                                                            os.path.join(courpus_file, "statistics.csv")
                                                                            )

        processor = processors[self.config.param("BERT", "task_name", type="string")]()
        model_type = self.config.param("BERT", "model_type", type="string")
        label_list = processor.get_labels()
        self.config.id2label = {i: label for i, label in enumerate(label_list)}
        self.config.label2id = {label: i for i, label in enumerate(label_list)}
        num_labels = len(label_list)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[
            self.config.param("BERT", "model_type", type="string")]
        config = config_class.from_pretrained(self.config.param("BERT", "model_name_or_path", type="string"),
                                              num_labels=num_labels)
        tokenizer = tokenizer_class.from_pretrained(self.config.param("BERT", "model_name_or_path", type="string"),
                                                    do_lower_case=self.config.param("BERT", "do_lower_case",
                                                                                    type="boolean"), )

        train_dataset = self.load_and_cache_examples(tokenizer, data_type='train')
        self.train_dl = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        dev_dataset = self.load_and_cache_examples(tokenizer, data_type='dev')
        self.eval_dl = DataLoader(dev_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataset = self.load_and_cache_examples(tokenizer, data_type='test')
        self.test_dl = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

        self.logger.info("Step01 Finish: Dataset building.\n")
        return

    def load_and_cache_examples(self, tokenizer, data_type='train'):
        local_rank = self.config.param("BERT", "local_rank", type="int")
        data_dir = self.config.param("BERT", "data_dir", type="string")
        model_type = self.config.param("BERT", "model_type", type="string")
        train_max_seq_length = self.config.param("BERT", "train_max_seq_length", type="int")
        eval_max_seq_length = self.config.param("BERT", "eval_max_seq_length", type="int")
        overwrite_cache = self.config.param("BERT", "overwrite_cache", type="boolean")
        model_name_or_path = self.config.param("BERT", "model_name_or_path", type="string")

        if local_rank not in [-1, 0] and not self.eval:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        processor = processors["jyb"]()
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(data_dir, 'cached_crf-{}_{}_{}_{}'.format(
            data_type,
            list(filter(None, model_name_or_path.split('/'))).pop(),
            str(train_max_seq_length if data_type == 'train' else eval_max_seq_length),
            str("jyb")))
        if os.path.exists(cached_features_file) and not overwrite_cache:
            self.logger.info("Loading features from cached file: " + cached_features_file)
            features = torch.load(cached_features_file)
        else:
            self.logger.info("Creating features from dataset file at: " + data_dir)
            label_list = processor.get_labels()
            if data_type == 'train':
                examples = processor.get_train_examples(self.datapool)
            elif data_type == 'dev':
                examples = processor.get_dev_examples(data_dir)
            elif data_type == 'eval':
                examples = processor.get_eval_examples(self.datapool)
            else:
                examples = processor.get_test_examples(data_dir)
            features = convert_examples_to_features(examples=examples,
                                                    tokenizer=tokenizer,
                                                    label_list=label_list,
                                                    max_seq_length=train_max_seq_length if data_type == 'train'
                                                        else eval_max_seq_length,
                                                    cls_token_at_end=bool(model_type in ["xlnet"]),
                                                    pad_on_left=bool(model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                                    )
            if local_rank in [-1, 0]:
                self.logger.info("Saving features into cached file: " + cached_features_file)
                torch.save(features, cached_features_file)
        if local_rank == 0 and not self.eval:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
        return dataset

    def build_bert_crf(self):
        """
        Step 02
        Build BERT+CRF Model.
        """
        self.logger.info("Step02 Begin: build bert crf.\n")

        batch_size = self.config.param("BERT", "batch_size", type="int")
        max_seq_len = self.config.param("WORD2VEC", "max_seq_len", type="int")
        device = self.config.param("BERT", "device", type="string")
        num_epoch = self.config.param("BERT", "num_epoch", type="int")
        learning_rate = self.config.param("BERT", "learning_rate", type="float")
        model_path_prefix = self.config.param("BERT", "model_path_prefix", type="string")
        processor = processors[self.config.param("BERT", "task_name", type="string")]()
        model_type = self.config.param("BERT", "model_type", type="string")
        strategy = self.config.param("ActiveStrategy", "strategy", type="string")
        choose_fraction = self.config.param("WORD2VEC", "choose_fraction", type="float")
        entity_digits = self.config.param("ENTITYLEVELF1", "digits", type="int")
        entity_return_report_boolean = self.config.param("ENTITYLEVELF1", "return_report", type="boolean")
        entity_average = self.config.param("ENTITYLEVELF1", "average", type="string")

        label_list = processor.get_labels()
        num_labels = len(label_list)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[
            self.config.param("BERT", "model_type", type="string")]
        config = config_class.from_pretrained(self.config.param("BERT", "model_name_or_path", type="string"),
                                              num_labels=num_labels, )
        self.model = model_class.from_pretrained(self.config.param("BERT", "model_name_or_path", type="string"),
                                                 config=config, tag_to_ix=self.preprocessor.tag_to_idx)
        self.model.to(device)
        # 改动begin
        tokenizer = tokenizer_class.from_pretrained(self.config.param("BERT", "model_name_or_path", type="string"),
                                                    do_lower_case=self.config.param("BERT", "do_lower_case",
                                                                                    type="boolean"), )
        train_dataset = self.load_and_cache_examples(tokenizer, data_type='train')
        train_dl = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        # 改动end

        # 统计训练集的token变化
        self.train_set_token(train_dl)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)
        loss_info = ""
        entity_f1_score_info = ""
        dlab_probs = []
        patience = 30
        early_stopping = EarlyStopping(patience, verbose=True)
        for epoch in range(1, num_epoch + 1):
            # if epoch % 1 == 0 and epoch != 0:
            #     torch.save(self.model.state_dict(),
            #                model_path_prefix + strategy + "_" + str(choose_fraction) + "_" + str(epoch) + '.pth')
            dlab_clses = []
            self.model.train()
            bar = tqdm(train_dl)
            # gradient accumulation 梯度累积
            # accum_step = 4
            for bi, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if model_type in ["bert", "xlnet"] else None)

                dlab_score, dlab_tag_seq, dlab_prob, dlab_cls = self.model(**inputs)
                # dlab_prob = torch.softmax(dlab_prob, dim=0)
                dlab_prob = dlab_prob.cpu().detach().numpy()
                dlab_probs.extend(dlab_prob.tolist())
                dlab_cls = dlab_cls.cpu().detach().numpy()
                dlab_clses.extend(dlab_cls.tolist())

                self.model.zero_grad()
                loss = self.model.loss(**inputs)
                # loss = loss / accum_step
                loss.backward()
                optimizer.step()
                # if (bi + 1) % accum_step == 0:
                #     optimizer.step()
                    # self.model.zero_grad()
                bar.set_description(f"{epoch:2d}/{num_epoch} loss: {loss:5.2f}")
            loss_info += f"{epoch:2d}/{num_epoch} loss: {loss:5.2f}\n"
            # dlab_clses -- (Dlab样本数, 768)
            self.dlab_clses = np.array(dlab_clses)
            self.dlab_probs = np.array(dlab_probs)

            # torch.save(self.model.state_dict(), model_path_prefix + '.pth')
            # self.model.load_state_dict(torch.load(model_path_prefix + '.pth'))

            self.model.eval()
            pred_tag_seq = []
            gold_tag_seq = []

            for bi, batch in enumerate(self.eval_dl):
                batch = tuple(t.to(device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if model_type in ["bert", "xlnet"] else None)
                with torch.no_grad():

                    scores, tag_seq, probs, cls = self.model(**inputs)
                # tag_seq = vec_to_tags(self.tags, tag_seq, max_seq_len)
                pred_tag_seq.extend(tag_seq)
                for item in batch[3].tolist():
                    item = [v for v in item if v > 0]
                    gold_tag_seq.append(item)

            pred_tag_seq = vec_to_tags(self.tags, pred_tag_seq, max_seq_len)
            gold_tag_seq = vec_to_tags(self.tags, gold_tag_seq, max_seq_len)
            eval = EvaluationIndex(self.logger)
            entity_f1_score = eval.entity_level_f1(gold_tag_seq, pred_tag_seq, entity_digits,
                                                   entity_return_report_boolean, entity_average)
            self.logger.info(f"Entity-level F1: {entity_f1_score}")
            entity_f1_score_info += f"{epoch:2d}/{num_epoch} entity_f1_score: {entity_f1_score:5.4f}\n"
            early_stopping(entity_f1_score, self.model, model_path_prefix, strategy, choose_fraction)
            if early_stopping.early_stop:
                print("*************")
                print("Early stopping")
                print("*************")
                break

        self.model.load_state_dict(torch.load(model_path_prefix + strategy + "_" + str(choose_fraction) + '.pth'))
        self.logger.info(f"Early Stopping  ————>>>  loss值：{loss_info}")
        self.logger.info(f"Early Stopping  ————>>>  在dev集上的分数：{entity_f1_score_info}")
        # torch.save(self.model.state_dict(),
        #            model_path_prefix + strategy + "_" + str(choose_fraction) + "_" + str(epoch) + '.pth')
        self.logger.info("Step02 Finish: bert crf.\n")

        return

    def predict_eval(self):
        """
        Step 03
        Use training model to predict and evaluate
        entity-level-F1
        sentence-level-accuracy
        """
        self.logger.info("Step03 Begin: Predicting and evaluation.\n")
        device = self.config.param("BERT", "device", type="string")
        max_seq_len = self.config.param("BERT", "max_seq_len", type="int")
        entity_digits = self.config.param("ENTITYLEVELF1", "digits", type="int")
        entity_return_report_boolean = self.config.param("ENTITYLEVELF1", "return_report", type="boolean")
        entity_average = self.config.param("ENTITYLEVELF1", "average", type="string")
        model_path_prefix = self.config.param("BERT", "model_path_prefix", type="string")
        model_type = self.config.param("BERT", "model_type", type="string")
        strategy = self.config.param("ActiveStrategy", "strategy", type="string")
        num_epoch = self.config.param("BERT", "num_epoch", type="int")
        choose_fraction = self.config.param("WORD2VEC", "choose_fraction", type="float")

        self.model.load_state_dict(torch.load(model_path_prefix + strategy + "_" + str(choose_fraction) + '.pth'))
        self.model.eval()

        pred_tag_seq = []
        gold_tag_seq = []

        for bi, batch in enumerate(self.test_dl):
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if model_type in ["bert", "xlnet"] else None)
            with torch.no_grad():

                scores, tag_seq, probs, cls = self.model(**inputs)
            # tag_seq = vec_to_tags(self.tags, tag_seq, max_seq_len)
            pred_tag_seq.extend(tag_seq)
            for item in batch[3].tolist():
                item = [v for v in item if v > 0]
                gold_tag_seq.append(item)

        pred_tag_seq = vec_to_tags(self.tags, pred_tag_seq, max_seq_len)
        gold_tag_seq = vec_to_tags(self.tags, gold_tag_seq, max_seq_len)

        eval = EvaluationIndex(self.logger)

        if entity_return_report_boolean:
            entity_f1_score, entity_return_report = eval.entity_level_f1(gold_tag_seq, pred_tag_seq,
                                                                         entity_digits, entity_return_report_boolean,
                                                                         entity_average)
            print(f"在test集上的  ————>>>  Classification report(Entity level):\n{entity_return_report}")
        else:
            entity_f1_score = eval.entity_level_f1(gold_tag_seq, pred_tag_seq, entity_digits,
                                                   entity_return_report_boolean, entity_average)

        self.logger.info(f"在test集上的  ————>>>  Entity-level F1: {entity_f1_score}")

        sentence_ac_score = eval.sentence_level_accuracy(gold_tag_seq, pred_tag_seq)

        self.logger.info("Step03 Finish: Predicting and evaluation.\n")

        return entity_f1_score, sentence_ac_score

    def train_set_token(self, train_dl):
        """
        统计训练集的token变化
        """
        self.logger.info("begin：统计训练集的token变化.\n")
        device = self.config.param("BERT", "device", type="string")
        max_seq_len = self.config.param("BERT", "max_seq_len", type="int")
        entity_digits = self.config.param("ENTITYLEVELF1", "digits", type="int")
        entity_return_report_boolean = self.config.param("ENTITYLEVELF1", "return_report", type="boolean")
        entity_average = self.config.param("ENTITYLEVELF1", "average", type="string")
        model_path_prefix = self.config.param("BERT", "model_path_prefix", type="string")
        model_type = self.config.param("BERT", "model_type", type="string")
        strategy = self.config.param("ActiveStrategy", "strategy", type="string")
        choose_fraction = self.config.param("WORD2VEC", "choose_fraction", type="float")


        self.model.load_state_dict(torch.load(model_path_prefix + strategy + "_" + str(choose_fraction) + '.pth'))
        self.model.eval()


        pred_tag_seq = []
        gold_tag_seq = []

        for bi, batch in enumerate(train_dl):
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if model_type in ["bert", "xlnet"] else None)
            with torch.no_grad():

                scores, tag_seq, probs, cls = self.model(**inputs)
            # tag_seq = vec_to_tags(self.tags, tag_seq, max_seq_len)
            pred_tag_seq.extend(tag_seq)
            for item in batch[3].tolist():
                item = [v for v in item if v > 0]
                gold_tag_seq.append(item)

        pred_tag_seq = vec_to_tags(self.tags, pred_tag_seq, max_seq_len)
        gold_tag_seq = vec_to_tags(self.tags, gold_tag_seq, max_seq_len)

        eval = EvaluationIndex(self.logger)

        if entity_return_report_boolean:
            entity_f1_score, entity_return_report = eval.entity_level_f1(gold_tag_seq, pred_tag_seq,
                                                                         entity_digits, entity_return_report_boolean,
                                                                         entity_average)
            print(f"在train集上的  ————>>>  Classification report(Entity level):\n{entity_return_report}")
        else:
            entity_f1_score = eval.entity_level_f1(gold_tag_seq, pred_tag_seq, entity_digits,
                                                   entity_return_report_boolean, entity_average)

        self.logger.info(f"在train集上的  ————>>>  Entity-level F1: {entity_f1_score}")

        sentence_ac_score = eval.sentence_level_accuracy(gold_tag_seq, pred_tag_seq)

        self.logger.info(" Finish: 统计训练集的token变化.\n")

        return entity_f1_score, sentence_ac_score

    def eval(self):
        """
        Step 04
        Use training model to predict and evaluate
        entity-level-F1
        sentence-level-accuracy
        """
        self.logger.info("Step04 Begin: evaluation.\n")
        device = self.config.param("BERT", "device", type="string")
        max_seq_len = self.config.param("BERT", "max_seq_len", type="int")
        model_path_prefix = self.config.param("BERT", "model_path_prefix", type="string")
        strategy = self.config.param("ActiveStrategy", "strategy", type="string")
        choose_fraction = self.config.param("WORD2VEC", "choose_fraction", type="float")
        model_type = self.config.param("BERT", "model_type", type="string")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[
            self.config.param("BERT", "model_type", type="string")]
        tokenizer = tokenizer_class.from_pretrained(self.config.param("BERT", "model_name_or_path", type="string"),
                                                    do_lower_case=self.config.param("BERT", "do_lower_case",
                                                                                    type="boolean"), )
        batch_size = self.config.param("BERT", "batch_size", type="int")
        ev_dataset = self.load_and_cache_examples(tokenizer, data_type='eval')

        # self.train_xs, self.train_ys = self.datapool.get_annotated_data()
        ev_dl = DataLoader(ev_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
        self.model.load_state_dict(torch.load(model_path_prefix + strategy + "_" + str(choose_fraction) + '.pth'))
        self.model.eval()
        scores, tag_seq_l, probs, tag_seq_str = [], [], [], []
        dloop_clses = []
        all_lengths = []  # 存储每个句子的长度，用于子序列查询c
        for bi, batch in enumerate(ev_dl):
            batch = tuple(t.to(device) for t in batch)
            _, _, _, _, batch_len = batch
            all_lengths.extend(batch_len.tolist())
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if model_type in ["bert", "xlnet"] else None)
            with torch.no_grad():

                score, tag_seq, prob, dloop_cls = self.model(**inputs)
                dloop_cls = dloop_cls.cpu().detach().numpy()
                dloop_clses.extend(dloop_cls.tolist())
                prob = torch.softmax(prob, dim=0)
                score, prob = score.cpu().detach().numpy(), prob.cpu().detach().numpy()

                scores.extend(score.tolist())
                probs.extend(prob.tolist())
                tag_seq_l.extend(tag_seq)
                tag_seq_str.extend(vec_to_tags(self.tags, tag_seq, max_seq_len))
        # dloop_clses -- (Dloop样本数，768）
        dloop_clses = np.array(dloop_clses)
        scores = np.array(scores)
        probs = np.array(probs)

        return scores, tag_seq_l, probs, tag_seq_str, dloop_clses, all_lengths

    def active_learning(self):

        self.logger.info("Begin active_learning.")
        strategy = self.config.param("ActiveStrategy", "strategy", type="string")
        strategy_name = strategy.lower()
        stop_echo = self.config.param("ActiveStrategy", "stop_echo", type="int")
        query_batch_fraction = self.config.param("ActiveStrategy", "query_batch_fraction", type="float")
        num_neighborhood = self.config.param("ActiveStrategy", "num_neighborhood", type="int")
        max_seq_len = self.config.param("BERT", "max_seq_len", type="int")
        seed = self.config.param("BERT", "seed", type="int")
        epsilon = self.config.param("ActiveStrategy", "epsilon", type="float")
        subseq_minlen = self.config.param("ActiveStrategy", "subseq_minlen", type="int")

        setup_seeds(seed)
        np.random.seed(seed)

        choice_number = int(self.datapool.get_total_number() * query_batch_fraction)
        choice_number = 1 if choice_number < 1 else choice_number
        strategy = self.strategy[strategy]
        test_Entity_level_f1_info = ''
        for i in range(0, stop_echo):
            now = int(round(time.time() * 1000))
            now02 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
            self.logger.info(f"使用weibo数据。{now02}")

            self.logger.info(
                f"[No. {i + 1}/{stop_echo}] ActiveStrategy--{strategy_name}\n"
                f"BatchFraction--{query_batch_fraction}\n"
                f"邻居数--{num_neighborhood}\n"
                f"seed--{seed}\n"
                f"子序列最短长度--{subseq_minlen}\n"
                f"epsilon--{epsilon}\n")
            self.build_bert_crf()
            # dlab_scores, dlab_tag_seq, dlab_probs, dlab_tag_seq_str, dlab_clses = self.eval(data_type="train")

            entity_f1_score, sentence_ac_score = self.predict_eval()
            test_Entity_level_f1_info += f"{i + 1:2d}/{stop_echo} test_Entity_level_f1_info: {entity_f1_score:5.4f}\n"
            if i == stop_echo-1:
                self.logger.info(f"AL在test集上的分数：{test_Entity_level_f1_info}")
            scores, tag_seq, probs, tag_seq_str, dloop_clses, all_lengths = self.eval()
            _, unannotated_labels = self.datapool.get_unannotated_data()

            idx = strategy.select_idx(choices_number=choice_number,
                                      epsilon=epsilon,
                                      iteration=i,
                                      subseq_minlen=subseq_minlen,
                                      probs=probs,
                                      scores=scores,
                                      all_lengths=all_lengths,
                                      best_path=tag_seq,
                                      num_neighborhood=num_neighborhood,
                                      dlab_clses=self.dlab_clses,
                                      dloop_clses=dloop_clses,
                                      dlab_probs=self.dlab_probs,
                                      )
            # selected_samples = unannotated_labels[idx]
            selected_samples = vec_to_tags(self.tags, tag_seq, max_seq_len)
            # tag_seq_str = [tag_seq_str[id] for id in idx]

            # update datapool
            self.datapool.update(mode="internal_exchange_u2a", selected_idx=idx)
            _reading_cost = SampleMetrics._reading_cost(selected_samples)
            self.logger.info(f"Reading Cost is {_reading_cost}")
            _annotation_cost = SampleMetrics._annotation_cost(selected_samples, tag_seq_str)
            self.logger.info(f"Annotation Cost is {_annotation_cost}")
            _wrong_select = SampleMetrics._percentage_wrong_selection(selected_samples, tag_seq_str)
            self.logger.info(f"Wrong Selected percentage: {_wrong_select}")
            self.logger.info(
                f"{strategy_name},{i},{entity_f1_score},{sentence_ac_score},{_reading_cost},{_annotation_cost},{_wrong_select}")
            del self.model
            torch.cuda.empty_cache()

    @property
    def tasks(self):
        return [
            self.build_dataset,
            self.active_learning,
        ]


if __name__ == '__main__':
    BERTCRFALPipeline()

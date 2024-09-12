from typing import List
import torch
import numpy as np
from torch import nn
from abc import ABCMeta, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from tqdm import tqdm
from scipy import stats
from decimal import *
from sklearn.metrics.pairwise import cosine_similarity


class ActiveLearningStrategy(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def select_idx(cls, choices_number: int,
                   epsilon: float,
                   iteration: int,
                   subseq_minlen: int,
                   dloop_probs: np.ndarray = None,
                   dloop_scores: np.ndarray = None,
                   all_lengths=None,
                   best_path: List[List[int]] = None,
                   num_neighborhood: int = None,
                   dlab_clses: np.ndarray = None,
                   dloop_clses: np.ndarray = None,
                   dlab_probs: np.ndarray = None,

                   **kwargs) -> np.ndarray:
        """
        probs: [B, L, C]
        scores: [B]
        best_path: [B, L]
        """
        pass


# 子序列不确定性+多样性+动态epsilon
class SubSequenceDiversityDynamic(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int,
                   epsilon: float,
                   iteration: int,
                   subseq_minlen: int,
                   probs: np.ndarray = None,
                   scores: np.ndarray = None,
                   all_lengths=None,
                   best_path: List[List[int]] = None,
                   dlab_clses: np.ndarray = None,
                   dloop_clses: np.ndarray = None,
                   **kwargs) -> np.ndarray:
        """
        Least Confidence Strategy
        all_lengths: [list]，存储每个句子的长度
        epsilon： 权重因子，控制不确定性与多样性所占比重，与不确定性成正比
        subseq_minlen: 子序列最短长度
        iteration：主动学习迭代轮次，用于线性改变epsilon
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)

        raw_prediction = []
        for p in probs:
            p = np.array(p)
            raw_prediction.append(p)

        word_prob = np.array([np.max(arr, axis=1) for arr in raw_prediction])

        ori_uncertainty = []  # 存储每个字的不确定分数
        for i, sentence in enumerate(word_prob):
            ori_uncertainty.append(-np.log(sentence[1:all_lengths[i] + 1]))

        sentence_uncertainty = []  # 存储句子整体的不确定分数
        for j, sentence in enumerate(word_prob):
            sentence_uncertainty.append(1 / all_lengths[j] * np.sum(-np.log(sentence[1:all_lengths[j] + 1])))

        subsequence_score_point = []  # 存储每个句子的子序列的不确定分数，以及开始索引和结束索引
        subsequence_score = []  # 存储每个句子的子序列的不确定分数

        for idx, word_score in enumerate(ori_uncertainty):
            start = 0
            end = len(word_score)
            initial_score = np.mean(word_score)
            start_score = initial_score
            end_score = initial_score

            while 1:
                start += 1
                current_score = np.mean(word_score[start:])  # current_score表示当前子序列的不确定分数
                if current_score < start_score:  # 如果 当前子序列不确定分数 < 变动之前的不确定分数
                    new_start = start - 1  # 那就撤销指针的变动
                    break
                else:
                    start_score = current_score  # 否则更新基准分数
                if start == len(word_score):
                    new_start = start
                    break

            while 1:
                end -= 1
                current_score = np.mean(word_score[:end])
                if current_score < end_score:
                    new_end = end + 1
                    break
                else:
                    end_score = current_score
                if end == 0:
                    new_end = end
                    break

            if new_start >= new_end:
                new_score = 0
                new_start = 0
                new_end = 0
            else:
                if new_start > len(word_score) or new_end > len(word_score):
                    import pdb
                    pdb.set_trace()

                key_temp = 0
                start_in = 0
                end_in = 0
                if new_start == 0 and new_end == 0:
                    pass
                else:
                    # 如果子序列长度不满足最小长度要求，则轮流在首尾添加，直至满足要求
                    while (new_end - new_start) < subseq_minlen:
                        if key_temp % 2 == 0:
                            if new_start - 1 < 0:
                                start_in = 1
                                pass
                            else:
                                new_start -= 1
                        if key_temp % 2 == 1:
                            if new_end + 1 > all_lengths[idx]:
                                end_in = 1
                                pass
                            else:
                                new_end += 1
                        key_temp += 1
                        if start_in == 1 and end_in == 1:
                            break
                new_score = np.mean(word_score[new_start:new_end])
            subsequence_score_point.append((new_score, new_start, new_end))
            subsequence_score.append(new_score)  # subsequence_score是子序列不确定分数

        # 对子序列不确定分数归一化
        subseq_score = np.array(subsequence_score)
        subseq_score_nor = (subseq_score - subseq_score.min()) / (subseq_score.max() - subseq_score.min())

        # 相似度分数similarity_score
        similarity_matrix = cosine_similarity(dloop_clses, dlab_clses)
        simi_score = similarity_matrix.mean(axis=1)

        # 相似度分数归一化 simi_score_nor
        simi_score_nor = (-1) * (simi_score - simi_score.min()) / (simi_score.max() - simi_score.min())

        # alpha为了控制epsilon线性增长
        alpha = round(iteration * epsilon, 2)
        print("alpha:", alpha)
        # 采样分数 acquire_score
        acquire_score = alpha * subseq_score_nor + (1 - alpha) * simi_score_nor

        # 返回前choices_number个采样分数最高的索引
        sort_index = np.argsort(-acquire_score)
        idx = sort_index[:choices_number]

        return idx


# 子序列不确定性+多样性+静态epsilon
class SubSequenceDiversityStatic(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int,
                   epsilon: float,
                   iteration: int,
                   subseq_minlen: int,
                   probs: np.ndarray = None,
                   scores: np.ndarray = None,
                   all_lengths=None,
                   best_path: List[List[int]] = None,
                   dlab_clses: np.ndarray = None,
                   dloop_clses: np.ndarray = None,
                   **kwargs) -> np.ndarray:
        """
        Least Confidence Strategy
        all_lengths: [list]，存储每个句子的长度
        epsilon： 权重因子，控制不确定性与多样性所占比重，与不确定性成正比
        subseq_minlen: 子序列最短长度
        iteration：主动学习迭代轮次，用于线性改变epsilon
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)

        raw_prediction = []
        for p in probs:
            p = np.array(p)
            raw_prediction.append(p)

        word_prob = np.array([np.max(arr, axis=1) for arr in raw_prediction])

        ori_uncertainty = []  # 存储每个字的不确定分数
        for i, sentence in enumerate(word_prob):
            ori_uncertainty.append(-np.log(sentence[1:all_lengths[i] + 1]))

        sentence_uncertainty = []  # 存储句子整体的不确定分数
        for j, sentence in enumerate(word_prob):
            sentence_uncertainty.append(1 / all_lengths[j] * np.sum(-np.log(sentence[1:all_lengths[j] + 1])))

        subsequence_score_point = []  # 存储每个句子的子序列的不确定分数，以及开始索引和结束索引
        subsequence_score = []  # 存储每个句子的子序列的不确定分数

        for idx, word_score in enumerate(ori_uncertainty):
            start = 0
            end = len(word_score)
            initial_score = np.mean(word_score)
            start_score = initial_score
            end_score = initial_score

            while 1:
                start += 1
                current_score = np.mean(word_score[start:])  # current_score表示当前子序列的不确定分数
                if current_score < start_score:  # 如果 当前子序列不确定分数 < 变动之前的不确定分数
                    new_start = start - 1  # 那就撤销指针的变动
                    break
                else:
                    start_score = current_score  # 否则更新基准分数
                if start == len(word_score):
                    new_start = start
                    break

            while 1:
                end -= 1
                current_score = np.mean(word_score[:end])
                if current_score < end_score:
                    new_end = end + 1
                    break
                else:
                    end_score = current_score
                if end == 0:
                    new_end = end
                    break

            if new_start >= new_end:
                new_score = 0
                new_start = 0
                new_end = 0
            else:
                if new_start > len(word_score) or new_end > len(word_score):
                    import pdb
                    pdb.set_trace()

                key_temp = 0
                start_in = 0
                end_in = 0
                if new_start == 0 and new_end == 0:
                    pass
                else:
                    # 如果子序列长度不满足最小长度要求，则轮流在首尾添加，直至满足要求
                    while (new_end - new_start) < subseq_minlen:
                        if key_temp % 2 == 0:
                            if new_start - 1 < 0:
                                start_in = 1
                                pass
                            else:
                                new_start -= 1
                        if key_temp % 2 == 1:
                            if new_end + 1 > all_lengths[idx]:
                                end_in = 1
                                pass
                            else:
                                new_end += 1
                        key_temp += 1
                        if start_in == 1 and end_in == 1:
                            break
                new_score = np.mean(word_score[new_start:new_end])
            subsequence_score_point.append((new_score, new_start, new_end))
            subsequence_score.append(new_score)  # subsequence_score是子序列不确定分数

        # 对子序列不确定分数归一化
        subseq_score = np.array(subsequence_score)
        subseq_score_nor = (subseq_score - subseq_score.min()) / (subseq_score.max() - subseq_score.min())

        # 相似度分数similarity_score
        similarity_matrix = cosine_similarity(dloop_clses, dlab_clses)
        simi_score = similarity_matrix.mean(axis=1)

        # 相似度分数归一化 simi_score_nor，值越大，多样性越高
        simi_score_nor = (-1) * (simi_score - simi_score.min()) / (simi_score.max() - simi_score.min())

        # 采样分数 acquire_score
        acquire_score = epsilon * subseq_score_nor + (1 - epsilon) * simi_score_nor

        # 返回前choices_number个采样分数最高的索引
        sort_index = np.argsort(-acquire_score)
        idx = sort_index[:choices_number]

        return idx


# 不确定性+多样性+动态epsilon
class UncertaintyDiversityDynamic(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int,
                   epsilon: float,
                   iteration: int,
                   subseq_minlen: int,
                   probs: np.ndarray = None,
                   scores: np.ndarray = None,
                   all_lengths=None,
                   best_path: List[List[int]] = None,
                   dlab_clses: np.ndarray = None,
                   dloop_clses: np.ndarray = None,
                   **kwargs) -> np.ndarray:
        """
        Least Confidence Strategy
        all_lengths: [list]，存储每个句子的长度
        epsilon： 权重因子，控制不确定性与多样性所占比重，与不确定性成正比
        subseq_minlen: 子序列最短长度
        iteration：主动学习迭代轮次，用于线性改变epsilon
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)

        # 对不确定分数归一化
        ucertainty_nor = (scores - scores.min()) / (scores.max() - scores.min())

        # 相似度分数similarity_score
        similarity_matrix = cosine_similarity(dloop_clses, dlab_clses)
        simi_score = similarity_matrix.mean(axis=1)

        # 相似度分数归一化 simi_score_nor，值越大，多样性越高
        simi_score_nor = (-1) * (simi_score - simi_score.min()) / (simi_score.max() - simi_score.min())

        # alpha为了控制epsilon线性增长
        alpha = round(iteration * epsilon, 2)
        # 采样分数 acquire_score
        acquire_score = alpha * ucertainty_nor + (1 - alpha) * simi_score_nor

        # 返回前choices_number个采样分数最高的索引
        sort_index = np.argsort(-acquire_score)
        idx = sort_index[:choices_number]

        return idx


# 只有子序列不确定性
class SubSequence(ActiveLearningStrategy):
        @classmethod
        def select_idx(cls, choices_number: int,
                       iteration: int,
                       subseq_minlen: int,
                       probs: np.ndarray = None,
                       scores: np.ndarray = None,
                       all_lengths=None,
                       best_path: List[List[int]] = None,
                       dlab_clses: np.ndarray = None,
                       dloop_clses: np.ndarray = None,
                       **kwargs) -> np.ndarray:
            """
            Least Confidence Strategy
            all_lengths: [list]，存储每个句子的长度
            subseq_minlen: 子序列最短长度
            iteration：主动学习迭代轮次，用于线性改变epsilon
            """
            assert probs.shape[0] == scores.shape[0] == len(best_path)

            raw_prediction = []
            for p in probs:
                p = np.array(p)
                raw_prediction.append(p)

            word_prob = np.array([np.max(arr, axis=1) for arr in raw_prediction])

            ori_uncertainty = []  # 存储每个字的不确定分数
            for i, sentence in enumerate(word_prob):
                ori_uncertainty.append(-np.log(sentence[1:all_lengths[i] + 1]))

            sentence_uncertainty = []  # 存储句子整体的不确定分数
            for j, sentence in enumerate(word_prob):
                sentence_uncertainty.append(1 / all_lengths[j] * np.sum(-np.log(sentence[1:all_lengths[j] + 1])))

            subsequence_score_point = []  # 存储每个句子的子序列的不确定分数，以及开始索引和结束索引
            subsequence_score = []  # 存储每个句子的子序列的不确定分数

            for idx, word_score in enumerate(ori_uncertainty):
                start = 0
                end = len(word_score)
                initial_score = np.mean(word_score)
                start_score = initial_score
                end_score = initial_score

                while 1:
                    start += 1
                    current_score = np.mean(word_score[start:])  # current_score表示当前子序列的不确定分数
                    if current_score < start_score:  # 如果 当前子序列不确定分数 < 变动之前的不确定分数
                        new_start = start - 1  # 那就撤销指针的变动
                        break
                    else:
                        start_score = current_score  # 否则更新基准分数
                    if start == len(word_score):
                        new_start = start
                        break

                while 1:
                    end -= 1
                    current_score = np.mean(word_score[:end])
                    if current_score < end_score:
                        new_end = end + 1
                        break
                    else:
                        end_score = current_score
                    if end == 0:
                        new_end = end
                        break

                if new_start >= new_end:
                    new_score = 0
                    new_start = 0
                    new_end = 0
                else:
                    if new_start > len(word_score) or new_end > len(word_score):
                        import pdb
                        pdb.set_trace()

                    key_temp = 0
                    start_in = 0
                    end_in = 0
                    if new_start == 0 and new_end == 0:
                        pass
                    else:
                        # 如果子序列长度不满足最小长度要求，则轮流在首尾添加，直至满足要求
                        while (new_end - new_start) < subseq_minlen:
                            if key_temp % 2 == 0:
                                if new_start - 1 < 0:
                                    start_in = 1
                                    pass
                                else:
                                    new_start -= 1
                            if key_temp % 2 == 1:
                                if new_end + 1 > all_lengths[idx]:
                                    end_in = 1
                                    pass
                                else:
                                    new_end += 1
                            key_temp += 1
                            if start_in == 1 and end_in == 1:
                                break
                    new_score = np.mean(word_score[new_start:new_end])
                subsequence_score_point.append((new_score, new_start, new_end))
                subsequence_score.append(new_score)  # subsequence_score是子序列不确定分数

            # 返回前choices_number个采样分数最高的索引
            sort_index = np.argsort(-np.array(subsequence_score))

            idx = sort_index[:choices_number]

            return idx

class LeastConfidenceStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Least Confidence Strategy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)

        idx = np.argpartition(-scores, choices_number - 1)[:choices_number]
        return idx


#
class RandomStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Random Select Strategy
        This method you can directly pass candidate_number: int

        .. Note:: Random Select does not require to predict on the unannotated samples!!
        """
        if "candidate_number" in kwargs:
            candidate_number = kwargs["candidate_number"]
        else:
            candidate_number = scores.shape[0]

        return np.random.choice(np.arange(candidate_number), size=choices_number, replace=False)


class LongStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        length = np.array([-len(path) for path in best_path])
        return np.argpartition(length, choices_number - 1)[:choices_number]


class NormalizedLeastConfidenceStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Normalized Least Confidence Strategy
        """

        assert probs.shape[0] == scores.shape[0] == len(best_path)
        normalized_frac = np.array([len(path) for path in best_path])
        scores = scores / normalized_frac
        idx = np.argpartition(-scores, choices_number - 1)[:choices_number]
        return idx


#
class LeastTokenProbabilityStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Least Token Probability Strategy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        ltp_scores = []
        for prob, path in zip(probs, best_path):
            prob = np.take(prob, path)
            ltp_scores.append(np.min(prob))
        idx = np.argpartition(ltp_scores, choices_number - 1)[:choices_number]
        return idx


class MinimumTokenProbabilityStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Minimum Token Probability Strategy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        mtp_socres = []
        for prob, path in zip(probs, best_path):
            prob = prob[:len(path)]
            prob -= np.max(prob)
            prob = np.exp(prob) / np.sum(np.exp(prob))
            mtp_socres.append(np.min(np.max(prob[:len(path)], axis=1)))
        idx = np.argpartition(mtp_socres, choices_number - 1)[:choices_number]
        return idx


class MaximumTokenEntropyStrategy(ActiveLearningStrategy):
    """
    TTE
    """

    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Maximum Token Entropy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        mte_socres = []
        for prob, path in zip(probs, best_path):
            prob = prob[:len(path)]
            prob -= np.max(prob)
            prob_softmax = np.exp(prob) / np.sum(np.exp(prob))
            mte_socres.append(np.sum(prob_softmax * np.log(prob_softmax)))
        idx = np.argpartition(mte_socres, choices_number - 1)[:choices_number]
        return idx


class TokenEntropyStrategy(ActiveLearningStrategy):
    """
    TTE
    """

    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Maximum Token Entropy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        mte_socres = []
        for prob, path in zip(probs, best_path):
            prob = prob[:len(path)]
            prob -= np.max(prob)
            prob_softmax = np.exp(prob) / np.sum(np.exp(prob))
            mte_socres.append(np.mean(prob_softmax * np.log(prob_softmax)))
        idx = np.argpartition(mte_socres, choices_number - 1)[:choices_number]
        return idx


class ContrastiveConfidenceStrategy(ActiveLearningStrategy):

    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, num_neighborhood: int = None,
                   dlab_clses: np.ndarray = None, dloop_clses: np.ndarray = None,
                   dlab_probs: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Our strategy, The Least Confidence & Contrastive
        """

        assert probs.shape[0] == scores.shape[0] == len(best_path)

        print(dlab_clses.shape)
        print(dloop_clses.shape)
        print(type(dlab_clses))
        KL_set = []
        for (xp, xp_prob) in tqdm(zip(dloop_clses, probs)):
            # xp_prob,待选样本的概率分布，[seq_len, 20]
            xp_prob = np.array(xp_prob)
            xp = np.expand_dims(xp, axis=0)
            euclidean_dist = cdist(xp, dlab_clses, metric='euclidean')
            idx = np.argpartition(euclidean_dist[-1, :], num_neighborhood)[:num_neighborhood]
            KL = 0.0
            KL2 = 0.0
            for neighborhood_idx in idx:
                # dlab_neigh_probs, 每个邻居的概率分布，[seq_len, 20]
                dlab_neigh_probs = np.array(dlab_probs[neighborhood_idx])
                for i in range(dlab_neigh_probs.shape[0]):
                    xl_word_prob = dlab_neigh_probs[i, :]
                    for j in range(xp_prob.shape[0]):
                        xp_word_prob = xp_prob[j, :]
                        for k in range(20):
                            xl_word_prob[k] = 1e-10 if xl_word_prob[k] < 0 else xl_word_prob[k]
                            xp_word_prob[k] = 1e-10 if xp_word_prob[k] < 0 else xp_word_prob[k]
                            KL += xl_word_prob[k] * np.log(xl_word_prob[k] / xp_word_prob[k])
                            KL1 = float(KL / (xp_prob.shape[0] * dlab_neigh_probs.shape[0]))
                            KL2 += KL1
            KL_set.append(KL2)
        print(KL_set)
        idx = np.argpartition(KL_set, (dloop_clses.shape[0] - choices_number))[-choices_number:]
        print(idx)
        return idx


class ContrastiveConfidenceStrategy111(ActiveLearningStrategy):

    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, num_neighborhood: int = None,
                   dlab_clses: np.ndarray = None, dloop_clses: np.ndarray = None,
                   dlab_probs: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        先根据cls选出邻居，再根据KL散度选择KL散度大的样本，CAL
        """

        assert probs.shape[0] == scores.shape[0] == len(best_path)

        print(dlab_clses.shape)
        print(dloop_clses.shape)
        print(type(dlab_clses))
        KL_set = []

        for (xp, xp_prob) in tqdm(zip(dloop_clses, probs)):
            # xp_prob,待选样本的概率分布，[seq_len, 20]
            xp_prob = np.array(xp_prob)
            # ndarry -> tensor
            xp_prob_tensor = torch.from_numpy(xp_prob)
            xp_prob_tensor.cuda()
            # 将概率值为负的替换成1e-10
            xp_prob_tensor_positive = torch.where(xp_prob_tensor < 0.0, 1e-10, xp_prob_tensor)
            chunk_xp_prob_tensor_positive = torch.chunk(xp_prob_tensor_positive,
                                                        xp_prob_tensor_positive.shape[0],
                                                        dim=0)
            xp = np.expand_dims(xp, axis=0)
            # print("计算欧几里得距离")
            euclidean_dist = cdist(xp, dlab_clses, metric='euclidean')
            # print("获取邻居")
            idx = np.argpartition(euclidean_dist[-1, :], num_neighborhood)[:num_neighborhood]
            kl = 0.0
            kl2 = 0.0
            # print("求xp与各个邻居之间的散度")
            for neighborhood_idx in idx:
                # dlab_neigh_probs, 每个邻居的概率分布，[seq_len, 20]
                dlab_neigh_probs = np.array(dlab_probs[neighborhood_idx])
                # ndarry -> tensor
                dlab_neigh_probs_tensor = torch.from_numpy(dlab_neigh_probs)
                dlab_neigh_probs_tensor.cuda()
                # 将概率值为负的替换成1e-10
                dlab_neigh_probs_tensor_positive = torch.where(dlab_neigh_probs_tensor < 0.0, 1e-10,
                                                               dlab_neigh_probs_tensor)
                chunk_dlab_neigh_probs_tensor_positive = torch.chunk(dlab_neigh_probs_tensor_positive,
                                                                     dlab_neigh_probs_tensor_positive.shape[0],
                                                                     dim=0)
                for chunk_dlab in chunk_dlab_neigh_probs_tensor_positive:
                    for chunk_xp in chunk_xp_prob_tensor_positive:
                        kl = stats.entropy(chunk_dlab, chunk_xp, axis=1)

                        kl1 = float(kl / (dlab_neigh_probs_tensor_positive.shape[0] * xp_prob_tensor_positive.shape[0]))
                        kl2 += kl1
            KL_set.append(kl2)
        # print(KL_set)
        print(len(KL_set))
        idx = np.argpartition(KL_set, (dloop_clses.shape[0] - choices_number))[-choices_number:]

        return idx


class MAL(ActiveLearningStrategy):

    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, num_neighborhood: int = None,
                   dlab_clses: np.ndarray = None, dloop_clses: np.ndarray = None,
                   dlab_probs: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        与CC111不同，CC222先根据cls选出邻居，再根据KL散度选择KL散度小的样本
        """

        assert probs.shape[0] == scores.shape[0] == len(best_path)

        print(dlab_clses.shape)
        print(dloop_clses.shape)
        print(type(dlab_clses))
        KL_set = []

        for (xp, xp_prob) in tqdm(zip(dloop_clses, probs)):
            # xp_prob,待选样本的概率分布，[seq_len, 20]
            xp_prob = np.array(xp_prob)
            # ndarry -> tensor
            xp_prob_tensor = torch.from_numpy(xp_prob)
            xp_prob_tensor.cuda()
            # 将概率值为负的替换成1e-10
            xp_prob_tensor_positive = torch.where(xp_prob_tensor < 0.0, 1e-10, xp_prob_tensor)
            chunk_xp_prob_tensor_positive = torch.chunk(xp_prob_tensor_positive,
                                                        xp_prob_tensor_positive.shape[0],
                                                        dim=0)
            xp = np.expand_dims(xp, axis=0)
            # print("计算欧几里得距离")
            euclidean_dist = cdist(xp, dlab_clses, metric='euclidean')
            # print("获取邻居")
            idx = np.argpartition(euclidean_dist[-1, :], num_neighborhood)[:num_neighborhood]
            kl = 0.0
            kl2 = 0.0
            # print("求xp与各个邻居之间的散度")
            for partner_idx in idx:
                # dlab_neigh_probs, 每个邻居的概率分布，[seq_len, 20]
                dlab_neigh_probs = np.array(dlab_probs[partner_idx])
                # ndarry -> tensor
                dlab_neigh_probs_tensor = torch.from_numpy(dlab_neigh_probs)
                dlab_neigh_probs_tensor.cuda()
                # 将概率值为负的替换成1e-10
                dlab_neigh_probs_tensor_positive = torch.where(dlab_neigh_probs_tensor < 0.0, 1e-10,
                                                               dlab_neigh_probs_tensor)
                chunk_dlab_neigh_probs_tensor_positive = torch.chunk(dlab_neigh_probs_tensor_positive,
                                                                     dlab_neigh_probs_tensor_positive.shape[0],
                                                                     dim=0)
                for chunk_dlab in chunk_dlab_neigh_probs_tensor_positive:
                    for chunk_xp in chunk_xp_prob_tensor_positive:
                        kl = stats.entropy(chunk_dlab, chunk_xp, axis=1)

                        kl1 = float(kl / (dlab_neigh_probs_tensor_positive.shape[0] * xp_prob_tensor_positive.shape[0]))
                        kl2 += kl1
            KL_set.append(kl2)
        # print(KL_set)
        print(len(KL_set))
        idx = np.argpartition(KL_set, choices_number - 1)[:choices_number]

        return idx


class MAL_AblationRandomPartners(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, num_neighborhood: int = None,
                   dlab_clses: np.ndarray = None, dloop_clses: np.ndarray = None,
                   dlab_probs: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Ablation Study:
            不再根据欧几里得距离选择partners，而是随机选择。
        """

        assert probs.shape[0] == scores.shape[0] == len(best_path)

        print(dlab_clses.shape)
        print(dloop_clses.shape)
        print(type(dlab_clses))
        KL_set = []

        for (xp, xp_prob) in tqdm(zip(dloop_clses, probs)):
            # xp_prob,待选样本的概率分布，[seq_len, 20]
            xp_prob = np.array(xp_prob)
            # ndarry -> tensor
            xp_prob_tensor = torch.from_numpy(xp_prob)
            xp_prob_tensor.cuda()
            # 将概率值为负的替换成1e-10
            xp_prob_tensor_positive = torch.where(xp_prob_tensor < 0.0, 1e-10, xp_prob_tensor)
            chunk_xp_prob_tensor_positive = torch.chunk(xp_prob_tensor_positive,
                                                        xp_prob_tensor_positive.shape[0],
                                                        dim=0)
            # 随机获取partners，而不靠欧几里得距离
            idx = np.random.randint(dlab_clses.shape[0], size=num_neighborhood)
            kl2 = 0.0
            # print("求xp与各个邻居之间的散度")
            for partner_idx in idx:
                # dlab_neigh_probs, 每个邻居的概率分布，[seq_len, 20]
                dlab_neigh_probs = np.array(dlab_probs[partner_idx])
                # ndarry -> tensor
                dlab_neigh_probs_tensor = torch.from_numpy(dlab_neigh_probs)
                dlab_neigh_probs_tensor.cuda()
                # 将概率值为负的替换成1e-10
                dlab_neigh_probs_tensor_positive = torch.where(dlab_neigh_probs_tensor < 0.0, 1e-10,
                                                               dlab_neigh_probs_tensor)
                chunk_dlab_neigh_probs_tensor_positive = torch.chunk(dlab_neigh_probs_tensor_positive,
                                                                     dlab_neigh_probs_tensor_positive.shape[0],
                                                                     dim=0)
                for chunk_dlab in chunk_dlab_neigh_probs_tensor_positive:
                    for chunk_xp in chunk_xp_prob_tensor_positive:
                        kl = stats.entropy(chunk_dlab, chunk_xp, axis=1)

                        kl1 = float(kl / (dlab_neigh_probs_tensor_positive.shape[0] * xp_prob_tensor_positive.shape[0]))
                        kl2 += kl1
            KL_set.append(kl2)
        # print(KL_set)
        print(len(KL_set))
        idx = np.argpartition(KL_set, choices_number - 1)[:choices_number]

        return idx

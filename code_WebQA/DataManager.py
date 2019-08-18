# -*- encoding: utf-8 -*-
"""
    @author: hongzhi, jiuniu
"""
import sys
sys.path.append('../')
import json
import os
import numpy as np
import pandas as pd
import jieba

from word_vocab import Vocab, CharVocab
import data_path as dt_p
from reformat_dat import RawWebQaReader

class DataManager:
    # 标起始位置的
    def __init__(self, params, log, no_need_dat=False, only_test_data=False):
        self.params = params
        self.log = log
        self.char_vocab = CharVocab(params.char_vocab_file, params.char_emb_dim)
        self.debug = params.debug
        self.data_rate = params.data_rate
        self.use_ir_evd = params.use_ir_evd

        if not no_need_dat:  # 快速加载
            self.test_dat = self.read_data(data_type='test')
            if not only_test_data:
                self.valid_dat = self.read_data(data_type='valid')
                train_dat = self.read_data(data_type='train')
                self.q_c_idxs, self.q_masks, self.p_c_idxs, self.p_masks, self.qe_comm_ft, \
                    self.b_pos, self.e_pos, self.p_tokens, self.q_tokens, _, self.p_evd_places = train_dat
                self.p_evd_places = np.array(self.p_evd_places)
                self.train_smp_num = len(self.qe_comm_ft)

    def load_both_dat(self, data_type):
        """
        载入原始数据 和 远监督数据（可选）
        :param which: train test or valid
        :return:
        """
        if data_type == 'train':
            dt_file = dt_p.web_qa_train_dat
        elif data_type == 'valid':
            dt_file = [dt_p.web_qa_valid_dat, dt_p.web_qa_ir_valid_dat] if self.use_ir_evd else dt_p.web_qa_valid_dat
        elif data_type == 'test':
            dt_file = [dt_p.web_qa_test_dat, dt_p.web_qa_ir_test_dat] if self.use_ir_evd else dt_p.web_qa_test_dat
        elif data_type == 'update_eval':
            dt_file = dt_p.web_qa_update_eval_dat
        else:
            raise RuntimeError('Error!')

        smps = self.load_webqa_lns(dt_file, self.use_ir_evd)
        self.log.info('[{}] data sample num [{}]'.format(data_type, len(smps)))

        if data_type == 'train':  # 控制训练数据量
            smps = smps[:int(self.params.kept_train_rate * len(smps))]
        # q, st, s_pos_, e_pos_, ans 每个sample都用这种格式
        return smps

    def load_webqa_lns(self, file_path, use_ir_evd=False):
        all_infos = RawWebQaReader.read_file(file_path, use_refined_flag=self.params.use_refined_flag, debug=self.debug,
                                             data_rate=self.data_rate, use_ir_evd=self.use_ir_evd)
        smps = []
        for q_infos in all_infos:
            q = q_infos['question_tokens']
            if use_ir_evd:
                st = []  # 'evidence_tokens'
                s_pos, e_pos, ans, evd_place = [],[],[],[]  # evd_place记录evd的结束位置
                evd_len = 0
                for i, evd in enumerate(q_infos['evidences']):
                    if evd['type'] != 'positive':
                        if ('training'in file_path or 'update' in file_path):
                            # 保证训练集中均为positive
                            continue
                        if i != 0:
                            # 测试和验证时，对于ann中的内容，直接放入passage中，等待模型处理
                            continue
                    if len(st) >= self.params.p_max_len:
                        continue
                    if 'end' in evd and evd_len + evd['end'] >= self.params.p_max_len:
                        continue

                    # if evd['type'] != 'positive' and ('training'in file_path or 'update' in file_path):
                    #     # 第一个条件是保证训练集中均为positive，第二个条件是限定valid test全部数据参与预测
                    #     continue
                    st += evd['evidence_tokens']
                    if 'train' in file_path:
                        s_pos.append(evd_len + evd['begin'])
                        e_pos.append(evd_len + evd['end'])
                        evd_len += len(evd['evidence_tokens'])
                        evd_place.append(evd_len)
                    else:
                        s_pos.append(-1)
                        e_pos.append(-1)
                    ans = evd['golden_answers']
                if len(ans) == 0:
                    continue
                # if 'train' in file_path:
                #     print("example:", (q, st, s_pos, e_pos, ans))
                smps.append((q, st, s_pos, e_pos, ans, evd_place))
            else:
                evd_place = []
                for evd in q_infos['evidences']:
                    if 'end' in evd and evd['end'] >= self.params.p_max_len:
                        continue
                    if evd['type'] != 'positive' and ('training'in file_path or 'update' in file_path):
                        continue
                    st = evd['evidence_tokens']
                    if 'train' in file_path:
                        s_pos = evd['begin']
                        e_pos = evd['end']
                    else:
                        s_pos = -1
                        e_pos = -1
                    ans = evd['golden_answers']
                    smps.append((q, st, s_pos, e_pos, ans, evd_place))
        # self.log.info("length of smps: {}".format(len(smps)))
        return smps

    def read_data(self, data_type):
        q_tokens, p_tokens, ans_tokens, b_pos, e_pos, flags = [], [], [], [], [], []
        q_c_idxs, p_c_idxs = [], []
        p_evd_places = []
        qe_comm_ft = []
        len_q_idxs, len_p_idxs = [], []
        smps = self.load_both_dat(data_type=data_type)
        for smp in smps:
            q, st, s_pos_, e_pos_, ans, evd_place = smp
            # for ln_info in ln_infos:
            try:
                e_max = max(e_pos_) if type(e_pos_) == list else e_pos_
            except:
                print("smp:", smp)
                print("e_pos_:", e_pos_)
                print(max(e_pos_))
            if e_max >= self.params.p_max_len:  # 删除训练集中超过长度限制的样本
                continue
            q_c_idxs.append(self.get_char_idxs(q, max_seq_len=self.params.q_max_len))
            p_c_idxs.append(self.get_char_idxs(st, max_seq_len=self.params.p_max_len))
            b_pos.append(s_pos_)
            e_pos.append(e_pos_)
            p_evd_places.append(evd_place)


            q_ws_set = set(q)
            qe_comm_ft.append([1 if p_w in q_ws_set else 0 for p_w in st])
            len_q_idxs.append([0] * len(q))
            len_p_idxs.append([0] * len(st))
            p_tokens.append(st)
            if data_type == 'test' or data_type == 'valid':  # 只有测试集保留 question/ans tokens
                q_tokens.append(q)
                ans_tokens.append(ans)
        _, q_masks = self.padding_sequence(len_q_idxs, max_len=self.params.q_max_len)
        _, p_masks = self.padding_sequence(len_p_idxs, max_len=self.params.p_max_len)
        qe_comm_ft, _ = self.padding_sequence(qe_comm_ft, max_len=self.params.p_max_len)
        p_c_idxs = np.array(p_c_idxs)
        q_c_idxs = np.array(q_c_idxs)
        b_pos, e_pos = [np.array(i) for i in (b_pos, e_pos)]
        return q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, b_pos, e_pos, p_tokens, q_tokens,\
               ans_tokens, p_evd_places

    def single_data(self, Questions, Passages):
        q_tokens, p_tokens = [], []
        q_c_idxs, p_c_idxs = [], []
        qe_comm_ft = []
        len_q_idxs, len_p_idxs = [], []

        for q, st in zip(Questions, Passages):
            q, st = list(jieba.cut(q, cut_all=False)), list(jieba.cut(st, cut_all=False))
            q_tokens.append(q)
            p_tokens.append(st)
            q_c_idxs.append(self.get_char_idxs(q, max_seq_len=self.params.q_max_len))
            p_c_idxs.append(self.get_char_idxs(st, max_seq_len=self.params.p_max_len))
            len_q_idxs.append([0] * len(q))
            len_p_idxs.append([0] * len(st))
            q_ws_set = set(q)
            qe_comm_ft.append([1 if p_w in q_ws_set else 0 for p_w in st])
        _, q_masks = self.padding_sequence(len_q_idxs, max_len=self.params.q_max_len)
        _, p_masks = self.padding_sequence(len_p_idxs, max_len=self.params.p_max_len)
        qe_comm_ft, _ = self.padding_sequence(qe_comm_ft, max_len=self.params.p_max_len)
        p_c_idxs = np.array(p_c_idxs)
        q_c_idxs = np.array(q_c_idxs)

        return q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, q_tokens, p_tokens

    def get_char_idxs(self, seq_ws, max_seq_len):
        char_idxs = []
        # print("seq_ws:",seq_ws)
        pad_char_idx = self.char_vocab.word2id(self.char_vocab.PAD_TOKEN)
        if len(seq_ws) > max_seq_len:
            seq_ws = seq_ws[:max_seq_len]
        else:
            seq_ws = seq_ws + [''] * (max_seq_len - len(seq_ws))
        for w in seq_ws:
            w_char_idxs = self.char_vocab.seqword2id(w)
            if len(w_char_idxs) < self.params.char_max_len:
                w_char_idxs = [pad_char_idx] * (self.params.char_max_len - len(w_char_idxs)) + w_char_idxs
            elif len(w_char_idxs) > self.params.char_max_len:
                w_char_idxs = w_char_idxs[:self.params.char_max_len]
            char_idxs.append(w_char_idxs)
        return char_idxs

    def padding_sequence(self, idxs, max_len):
        new_idxs = []
        masks = []
        for idx_smp in idxs:
            to_add = max_len - len(idx_smp)
            if to_add >= 0:
                new_idxs.append(idx_smp + [0] * to_add)
                masks.append([0] * len(idx_smp) + [1] * to_add)
            else:
                new_idxs.append(idx_smp[:max_len])
                masks.append([0] * max_len)
        return np.array(new_idxs), np.array(masks)

    def get_train_batch(self, batch_size):
        indexes = np.array(np.random.randint(self.train_smp_num, size=batch_size))
        q_masks_batch = self.q_masks[indexes]
        p_masks_batch = self.p_masks[indexes]
        qe_comm_fts_batch = self.qe_comm_ft[indexes]
        q_c_idxs_batch = self.q_c_idxs[indexes]
        b_batch = self.b_pos[indexes]
        e_batch = self.e_pos[indexes]
        evd_place_batch = self.p_evd_places[indexes]

        p_c_idxs_batch = self.p_c_idxs[indexes]
        return p_c_idxs_batch, qe_comm_fts_batch, p_masks_batch, q_c_idxs_batch, \
               q_masks_batch, b_batch, e_batch, evd_place_batch


if __name__ == '__main__':
    pass

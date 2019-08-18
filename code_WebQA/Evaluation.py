# -*- encoding: utf-8 -*-
# class Evaluation:
#     def __init__(self):
import gzip
import json
import sys

import data_path as dat
from evaluation.evaluate_tagging_result import get_val_result
from evaluation.tagging_util import get_label
from evaluation.tagging_evaluation_util import get_tagging_results
from evaluation.evaluate_voting_result import get_voting_val_result, get_voting_val_result_in_ans
from evaluation.evaluation_stats_util import F1Stats


class Options:
    def __init__(self, test_or_valid, fuzzy, use_ir_evd=False):
        if 'valid' in test_or_valid:
            self.test_file = dat.web_qa_ir_valid_dat if use_ir_evd else dat.web_qa_valid_dat
        elif 'test' in test_or_valid:
            self.test_file = dat.web_qa_ir_test_dat if use_ir_evd else dat.web_qa_test_dat
        else:
            raise NotImplementedError('No such kind of data!')
        self.schema = 'BIO2'
        self.fuzzy = fuzzy
        self.need_every_score = False


class VotingOptions:
    def __init__(self, test_or_valid, fuzzy):
        self.test_file = dat.web_qa_ir_valid_dat if test_or_valid == 'valid' else dat.web_qa_ir_test_dat
        self.schema = 'BIO2'
        self.fuzzy = fuzzy

def load_golden_answers(test_file):
    f = gzip.open(test_file, 'rb')
    lns = f.readlines()
    golden_answers = []
    for ln in lns:
        raw_q_infos = json.loads(ln)
        g_answer = None
        for raw_evd in raw_q_infos['evidences']:
            # if raw_evd['type'] == 'positive':
                g_answer = [''.join(raw_evd['golden_answers'][0])]
                break   # 一个问题、一个正例
        if g_answer is None:
            print('')
        golden_answers.append(g_answer)
    return golden_answers


def f1_on_hand(answers, test_or_valid):
    options = VotingOptions(test_or_valid, False)
    golden_answers = load_golden_answers(options.test_file)
    stats = F1Stats(is_fuzzy=False)
    stats_fuzzy = F1Stats(is_fuzzy=True)
    for ans, g_ans in zip(answers, golden_answers):
        stats.update(g_ans, ans)
        stats_fuzzy.update(g_ans, ans)
    return stats, stats_fuzzy




def eval_it(p_lbs, valid_or_test='test', use_ir_evd=False):
    options = Options(valid_or_test, fuzzy=False, use_ir_evd=use_ir_evd)
    res = get_val_result(options=options, labels=p_lbs)
    options = Options(valid_or_test, fuzzy=True, use_ir_evd=use_ir_evd)
    res_fussy = get_val_result(options=options, labels=p_lbs)
    return res, res_fussy


def eval_it_and_get_every_score(p_lbs, valid_or_test='test'):
    options = Options(valid_or_test, fuzzy=True)
    options.need_every_score = True
    res_fussy = get_val_result(options=options, labels=p_lbs)
    return res_fussy


def eval_it_vote(p_lbs, valid_or_test='test'):
    options = VotingOptions(valid_or_test, fuzzy=False)
    res = get_voting_val_result(options=options, labels=p_lbs)
    options = VotingOptions(valid_or_test, fuzzy=True)
    res_fussy = get_voting_val_result(options=options, labels=p_lbs)
    return res, res_fussy


def eval_it_vote_decode_mode(p_lbs, pred_ans, valid_or_test='test'):
    options = VotingOptions(valid_or_test, fuzzy=False)
    res = get_voting_val_result_in_ans(options=options, labels=p_lbs, pred_answers=pred_ans)
    options = VotingOptions(valid_or_test, fuzzy=True)
    res_fussy = get_voting_val_result_in_ans(options=options, labels=p_lbs, pred_answers=pred_ans)
    return res, res_fussy


def get_pred_chunks(p_tokens, tag_idxs, schema='BIO2'):
    """
    :param p_tokens: 一个列表
    :param tag_idxs:
    :param schema:
    :return:
    """
    tags = get_label(tag_idxs, schema=schema)
    chunks = get_tagging_results(p_tokens, tags)
    return chunks


if __name__ == '__main__':
    f1_on_hand([], 'test')
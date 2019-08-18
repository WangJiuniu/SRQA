import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path += [curdir]
import argparse
from evaluation_stats_util import F1Stats
from voter import iter_voting_results
from ioutil import open_file


def get_voting_val_result(options, labels):
    stats = F1Stats(options.fuzzy)

    for q_tokens, golden_answers, pred_answers, freqs in \
            iter_voting_results(labels, options.test_file, options.schema):
        stats.update(golden_answers, pred_answers)   # 都是列表
    return stats


def get_voting_val_result_in_ans(options, pred_answers, labels):
    stats = F1Stats(options.fuzzy)
    # pred_answers
    idx = 0
    for q_tokens, golden_answers, no_use_pred_answers, freqs in \
            iter_voting_results(labels, options.test_file, options.schema):
        # pred_answers Ԥ��Ľ��
        pred_answers_i = pred_answers[idx]
        stats.update(golden_answers, pred_answers_i)
        idx += 1
    return stats

if __name__ == '__main__':
    main()

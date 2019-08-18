import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path += [curdir]
import argparse

from evaluation.tagging_evaluation_util import get_tagging_results
from evaluation.evaluation_stats_util import F1Stats
from evaluation.raw_result_parser import iter_results
from evaluation.ioutil import open_file

class Options:
    def __init__(self):
        self.raw_prediction = 'example/raw_prediction_example.txt'
        self.test_file = 'example/test_file_example.json.gz'
        self.schema = 'BIO2'
        self.fuzzy = True


def get_val_result(options, labels):
    stats = F1Stats(options.fuzzy, need_every_score=options.need_every_score)
    # print("len(labels)[evaluate_tagging_result]:",len(labels))
    for q_tokens, e_tokens, tags, golden_answers in \
            iter_results(labels, options.test_file, options.schema):
        if q_tokens is None: continue # one question has been processed
        # print('len tags:', len(tags), 'len e_tokens:', len(e_tokens))
        pred_answers = get_tagging_results(e_tokens, tags)
        # print("pred_answers:", pred_answers)
        # print("golden_answers:", golden_answers)
        stats.update(golden_answers, pred_answers)
    # print((stats.get_metrics_str()))
    return stats


if __name__ == '__main__':
    options = Options()
    get_val_result(options, labels=options.raw_prediction)
# chunk_f1=0.413521 chunk_precision=0.436303 chunk_recall=0.393000 true_chunks=4000 result_chunks=3603 correct_chunks=1572
# chunk_f1=0.460345 chunk_precision=0.485706 chunk_recall=0.437500 true_chunks=4000 result_chunks=3603 correct_chunks=1750

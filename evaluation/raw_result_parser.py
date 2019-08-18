import json
from tagging_util import get_label
from ioutil import open_file
from datapoint import DataPoint, Evidence

__all__ = ['iter_results']


def parse_line(line):
    def concat_answers(answers):
        return [' '.join(answer) for answer in answers]

    data = json.loads(line.decode('utf-8'))
    q_tokens = data[DataPoint.Q_TOKENS]
    evis = data[DataPoint.EVIDENCES]
    evi_tokens_list = [evi[Evidence.E_TOKENS] for evi in evis]
    golden_answers_list = [concat_answers(evi[Evidence.GOLDEN_ANSWERS]) \
            for evi in evis]
    return q_tokens, evi_tokens_list, golden_answers_list


def iter_results(lbs_or_file, test_file, schema):
    predictions = []
    if isinstance(lbs_or_file, str):
        with open_file(lbs_or_file) as predict:
            for line in predict:
                label = get_label(int(line.split(';')[0]), schema)
                predictions.append(label)
    else:
        for lb in lbs_or_file:
            label = get_label(lb, schema)
            predictions.append(label)

    idx = 0
    with open_file(test_file) as test_file:
        line_num = 0
        evi_num = 0
        for line in test_file:
            line_num += 1
            q_tokens, evi_tokens_list, golden_answers_list = parse_line(line)
            for e_tokens, golden_answers in \
                    zip(evi_tokens_list, golden_answers_list):
                tags = predictions[idx:idx+len(e_tokens)]
                evi_num += 1
                yield q_tokens, e_tokens, tags, golden_answers
                idx += len(e_tokens)
            # one question has been parsed, needed by voters
            yield None, None, None, None
    assert idx == len(predictions)

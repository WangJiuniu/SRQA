#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from flask import Flask, request
import json
app = Flask(__name__)
import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)  # 本目录
sys.path.append(os.path.split(curdir)[0])  # 上级目录
from parameters_WebQA_test import ModelParameters
from main import logger, Train

params = ModelParameters()

log = logger()
log.info('params.train_idx: {}'.format(params.train_idx))
log.info('params.debug: {}'.format(params.debug))
log.info('params.use_self_attention: {}'.format(params.use_self_attention))
log.info('params.loss_type: {}'.format(params.loss_type))
params = ModelParameters()
qa_model = Train(params, no_need_dat=True)


def get_qa_pairs(data):
    Questions, Passages = [], []
    for each_question in data:
        question = each_question['question']
        for each_passage in each_question['passages']:
            passage = each_passage['text']
            Questions.append(question)
            Passages.append(passage)
    return Questions, Passages


def get_result(Ans, data):
    result = []
    count = 0
    for each_question in data:
        each_result = {"question_id": each_question['question_id'], }
        answer = []
        for each_passage in each_question['passages']:
            each_answer = {
                "passage_id": each_passage['passage_id'],
                "answer": Ans[count]['answer'],
                "prob": Ans[count]['prob'],
                "start_pos": Ans[count]['start_pos'],
                "end_pos": Ans[count]['end_pos']
            }
            answer.append(each_answer)
            count += 1
        each_result["answer"] = answer
        result.append(each_result)
    return result


@app.route('/mrc', methods=['POST'])
def mrc():
    try:
        input_json = request.data
        input_json = json.loads(input_json)
        Questions, Passages = get_qa_pairs(input_json['data'])
    except:
        output_json = {"success": False, "message": "Error when reading input_json.",
                       "result": []}
        return json.dumps(output_json)

    try:
        Ans = qa_model.test_sample(Questions, Passages)
        result = get_result(Ans, input_json['data'])
        output_json = {
            "success": True,
            "message": "Get answers",
            "result": result,
        }
    except:
        output_json = {"success": False, "message": "Error when calculating answers.",
                       "result": []}
        return json.dumps(output_json)
    return json.dumps(output_json)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
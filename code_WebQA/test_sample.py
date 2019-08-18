# -*- encoding: utf-8 -*-
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


if __name__ == '__main__':
    params = ModelParameters()
    qa_model = Train(params, no_need_dat=True)
    
    while(True):
        Question = input('\n请输入问题（输入\'n\'表示退出系统）：\n')
        if Question.strip() == 'n':
            break
        Passage = input('请输入依据的文档（输入\'n\'表示退出系统）：\n')
        if Passage.strip() == 'n':
            break
        Questions, Passages = [], []  # the list for questions and their corresponding passages
        Questions.append(Question)
        Passages.append(Passage)
        Ans = qa_model.test_sample(Questions, Passages)
        print('\nQuestion: {}'.format(Question))
        print('Passage: {}'.format(Passage))
        print('Ans: {}'.format(Ans[0]['answer']))
        print()


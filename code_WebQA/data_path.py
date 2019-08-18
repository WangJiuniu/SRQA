import os
import pickle

dp = os.path.abspath(os.path.dirname(__file__))

web_qa_dp = os.path.join(dp, '..', 'data')

# 原来数据集给的文件
web_qa_train_dat = os.path.join(web_qa_dp, 'training.json.gz')
web_qa_update_eval_dat = os.path.join(web_qa_dp, 'update_eval.json.gz')
web_qa_test_dat = os.path.join(web_qa_dp, 'test.ann.json.gz')
web_qa_ir_test_dat = os.path.join(web_qa_dp, 'test.ir.json.gz')
web_qa_valid_dat = os.path.join(web_qa_dp, 'validation.ann.json.gz')
web_qa_ir_valid_dat = os.path.join(web_qa_dp, 'validation.ir.json.gz')
refined_train_lbs = os.path.join(web_qa_dp, 'refined_lbs.pkl')   # 重新整理了训练的label


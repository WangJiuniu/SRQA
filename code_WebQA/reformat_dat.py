# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""
import sys
sys.path.append("../")
import gzip
import json
import pickle
import logging
from collections import Counter
import data_path as dt

logger = logging.getLogger('mrc')
class RawWebQaReader:
    def __init__(self):
        # 在初始化的时候就读了所有人工标注的数据
        self.train_dat = self.read_file(dt.web_qa_train_dat)
        # self.test_dat = self.read_file(dt.web_qa_ir_test_dat)
        self.test_dat = self.read_file(dt.web_qa_test_dat)
        self.valid_dat = self.read_file(dt.web_qa_valid_dat)

    @staticmethod
    def read_file(file_path, use_refined_flag=False, debug=False, data_rate=1.0, use_ir_evd=False):
        """
        从原始文件中读取数据
        :param file_path: 需要读取的文件路径（.json.gz结尾）,若为list形式，则表明希望将ann与ir的evidence统一
        :param use_refined_flag: 是否使用重新规划过的flag（默认不使用）
        :param debug: 是否只读入很少的训练数据
        :param data_rate: 读入数据的比例
        :return: all_infos
        """
        if type(file_path) != list:
            f = gzip.open(file_path, 'rb')  #
            if 'training' in file_path and 'training_eval' not in file_path and debug:
                lns = [f.readline() for _ in range(500)]  # 调试时只读500行
            else:
                if 'training' in file_path:
                    # 总条数事先写好，按比例读取数据
                    train_lns_length = 36181
                    logger.info("load {} in data rate {}".format(file_path, data_rate))
                    lns = [f.readline() for _ in range(int(train_lns_length * data_rate))]
                else:
                    lns = f.readlines()
            oo, tt = 0, 0  # oo:答案多于1被滤除， tt:type positive
            all_infos = []
            all_refined_lbs = []
            if use_refined_flag and 'training' in file_path:
                all_refined_lbs = pickle.load(open(dt.refined_train_lbs, 'rb'))
            refine_idx = 0
            for ln in lns:
                q_infos = {'evidences': []}
                raw_q_infos = json.loads(ln.decode('utf-8'))
                q_infos['question_tokens'] = raw_q_infos['question_tokens']
                if use_ir_evd:  # 暂时没有做特异化处理
                    # TODO：此处对于refine的方式和读取内容的方式都有值得改进之处
                    raw_q_infos['evidences'].sort(key=lambda x: len(x['evidence_tokens']))
                    for raw_evd in raw_q_infos['evidences']:
                        evd = {k: raw_evd[k] for k in
                               ['evidence_tokens', 'type', 'golden_answers', 'q-e.comm_features', 'source']}
                        if evd['evidence_tokens'][-1] not in [".。？！?!"]:
                            evd['evidence_tokens'].append("。")
                            evd['q-e.comm_features'].append(0)
                        if 'training' in file_path:  # 只有在训练的时候才考虑label的问题
                            if raw_evd['type'] == 'positive':
                                tt += 1
                                lbs = raw_evd['golden_labels']
                                cnt = Counter(lbs)
                                if cnt.get('b', 0) > 1:  # 这里还多虑掉了一部分训练数据 将来加进来，很烦的哈  训练数据而已的   如果模型结果好，应该也不在意这一点点的
                                    oo += 1
                                    if not use_refined_flag:
                                        continue
                                    else:  # 将多个‘b’的也取第一个
                                        lbs = all_refined_lbs[refine_idx]  # 因为在梳理最佳结果的时候是赋予同样条件的，所以有对应关系，但目前只有train的部分
                                        refine_idx += 1
                                evd['begin'] = lbs.index('b')
                                end = evd['begin'] + 1
                                while end < len(lbs) and lbs[end] == 'i':
                                    end += 1
                                evd['end'] = end
                        q_infos['evidences'].append(evd)
                else:
                    for raw_evd in raw_q_infos['evidences']:
                        evd = {k: raw_evd[k] for k in
                               ['evidence_tokens', 'type', 'golden_answers', 'q-e.comm_features', 'source']}
                        if 'training' in file_path:  # 只有在训练的时候才考虑label的问题
                            if raw_evd['type'] == 'positive':
                                tt += 1
                                lbs = raw_evd['golden_labels']
                                cnt = Counter(lbs)
                                if cnt.get('b', 0) > 1:  # 这里还多虑掉了一部分训练数据 将来加进来，很烦的哈  训练数据而已的   如果模型结果好，应该也不在意这一点点的
                                    oo += 1
                                    if not use_refined_flag:
                                        continue
                                    else:  # 将多个‘b’的也取第一个
                                        lbs = all_refined_lbs[refine_idx]
                                        refine_idx += 1
                                evd['begin'] = lbs.index('b')
                                end = evd['begin'] + 1
                                while end < len(lbs) and lbs[end] == 'i':
                                    end += 1
                                evd['end'] = end
                        q_infos['evidences'].append(evd)
                all_infos.append(q_infos)

            if 'train' in file_path and 'train_eval' not in file_path:
                logger.info('Training data loaded.')
                logger.info(
                    'question num:{}, total_pos_num:{}, more than one b:{}, refined: {}'.format(len(all_infos), tt, oo,
                                                                                                refine_idx))
            else:
                set_type = 'Test' if 'test' in file_path else 'Dev'
                logger.info('{} data loaded.'.format(set_type))
                logger.info('question num:{}'.format(len(all_infos)))
            return all_infos
        else:  # file_path是 ann和ir组成的列表
            file_path_ann, file_path_ir = file_path[0], file_path[1]  # 目前只有在使用ir测评的时候才会出现这种情况
            f_ann = gzip.open(file_path_ann, 'rb')
            lns_ann = f_ann.readlines()
            f_ir = gzip.open(file_path_ir, 'rb')
            lns_ir = f_ir.readlines()
            all_infos = []
            for (ln_ann, ln_ir) in zip(lns_ann, lns_ir):
                q_infos = {'evidences': []}
                raw_q_infos_ann = json.loads(ln_ann.decode('utf-8'))
                raw_q_infos_ir = json.loads(ln_ir.decode('utf-8'))
                q_infos['question_tokens'] = raw_q_infos_ann['question_tokens']
                # TODO：此处对于refine的方式和读取内容的方式都有值得改进之处
                for raw_evd in raw_q_infos_ann['evidences']:
                    evd = {k: raw_evd[k] for k in
                           ['evidence_tokens', 'type', 'golden_answers', 'q-e.comm_features', 'source']}
                    q_infos['evidences'].append(evd)
                # ir中tokens短的排在前面
                raw_q_infos_ir['evidences'].sort(key=lambda x:len(x['evidence_tokens']))
                # print("raw_q_infos_ir['evidences']\n", raw_q_infos_ir['evidences'])
                for raw_evd in raw_q_infos_ir['evidences']:
                    evd = {k: raw_evd[k] for k in
                           ['evidence_tokens', 'type', 'golden_answers', 'q-e.comm_features', 'source']}
                    q_infos['evidences'].append(evd)
                all_infos.append(q_infos)

            set_type = 'Test' if 'test' in file_path else 'Dev'
            logger.info('{} data loaded.'.format(set_type))
            logger.info('question num:{}'.format(len(all_infos)))
            return all_infos



    @staticmethod
    def read_evidences_need_refine(file_path=dt.web_qa_train_dat):
        f = gzip.open(file_path, 'rb')
        lns = f.readlines()
        oo, tt = 0, 0
        all_infos = []
        for ln in lns: #[:1000]
            q_infos = {'evidences': []}
            raw_q_infos = json.loads(ln.decode('utf-8'))
            q_infos['question_tokens'] = raw_q_infos['question_tokens']
            for raw_evd in raw_q_infos['evidences']:
                evd = {k: raw_evd[k] for k in ['evidence_tokens', 'type', 'golden_answers', 'q-e.comm_features', 'golden_labels']}
                if raw_evd['type'] == 'positive':
                    tt += 1
                    lbs = raw_evd['golden_labels']
                    cnt = Counter(lbs)
                    if cnt.get('b', 0) > 1:
                        oo += 1
                        q_infos['evidences'].append(evd)
                    else:
                        continue
            if len(q_infos):
                all_infos.append(q_infos)
        return all_infos

# build a vocab map them to idx then i can train it.   point net loss function 怎么写？

def write_all_questions():
    all_infos = RawWebQaReader.read_file(dt.web_qa_train_dat)
    questions = []
    for info in all_infos:
        q = info['question_tokens']
        q = u''.join(q) + u'\n'
        questions.append(q)
    of = open('web_qa_questions', 'w')
    of.writelines(questions)
    of.close()


if __name__ == '__main__':
    # RawWebQaReader()
    # test_dat = RawWebQaReader.read_file(dt.web_qa_test_dat)
    # fo = open('test.txt', 'w', encoding='utf-8')
    # lns = []
    # for q_info in test_dat:
    #     lns.append(u' '.join(q_info['question_tokens'])+'\n')
    #     for evi in q_info['evidences']:
    #         # lns.append('----'*10+'\n')
    #         lns.append(u' '.join(evi['golden_answers'][0])+'\n')
    #         lns.append(u' '.join(evi['evidence_tokens'])+'\n')
    #     lns.append('==='*10+'\n')
    # fo.writelines(lns)
    write_all_questions()
    pass
# question num:36181, total_pos_num:139199, more than one b:42971  还有这么多没用的数据


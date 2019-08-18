# -*- encoding: utf-8 -*-
import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)  # current direction
sys.path.append(os.path.split(curdir)[0])  # former direction
from parameters import ModelParameters
import logging
import torch
from DataManager import DataManager
from SingleDocReaderPNet import DocReaderModel
from get_F1 import F1Counter
import pickle

params = ModelParameters()
# define the log
def logger():
    log = logging.getLogger('mrc')
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(params.log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
    return log

log = logger()
log.info('params.train_idx: {}'.format(params.train_idx))
log.info('params.debug: {}'.format(params.debug))
log.info('params.use_self_attention: {}'.format(params.use_self_attention))
log.info('params.loss_type: {}'.format(params.loss_type))
if params.loss_type == 'adv':
    log.info('params.p_mult: {}'.format(params.p_mult))
    log.info('params.add_random_noise: {}'.format(params.add_random_noise))


class Train:
    def __init__(self, params, no_need_dat=False, dt=None, only_test_data=False):
        """
        :param no_need_dat: 默认False，表示需要载入数据
        :param dt: 传入数据管理器(默认为None，在程序中读取)
        :param only_test_data: 默认False，表示不仅载入test dev data，还需要载入train data
        """
        self.params = params
        self.log = log
        self.best_test_f1 = 0
        self.best_dev_f1 = 0
        self.real_test_f1 = 0
        self.best_test_strict_f1 = 0
        if dt is not None:
            self.dt = dt
        else:
            self.dt = DataManager(self.params, log=self.log, no_need_dat=no_need_dat,
                                  only_test_data=only_test_data)
        if self.params.resume_flag or only_test_data:
            # load the previous model
            self.model = self.resume()
        else:
            model_dir = self.params.model_dir
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.model = DocReaderModel(opt=self.params,
                                        char_embedding=self.dt.char_vocab.vs, state_dict=None)
        if torch.cuda.is_available():
            self.model.cuda()

    def resume(self):
        # 载入模型文件
        print('Load model here!')
        self.log.info('[loading previous model...]')
        checkpoint = torch.load(self.params.pretrained_model)
        opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt=opt,
                               char_embedding=self.dt.char_vocab.vs, state_dict=state_dict)
        return model

    def valid_it(self):
        # 验证dev 和 test的正确率
        strict_f1 = {}
        fuzzy_f1 = {}
        eval_list = ['valid_noise', 'test_noise'] if self.params.show_noise_eval else []
        eval_list += ['valid', 'test']
        for valid_or_test in eval_list:
            self.log.info('===evaluating {}==='.format(valid_or_test))
            lbs, all_s, all_e, b_pos, e_pos, p_tokens, q_tokens, ans_tokens\
                = self.model.predict_on_valid_or_test_constrained(self.dt, valid_or_test, return_p_res=True)
            strict_stats = F1Counter(is_fuzzy=False)
            fuzzy_stats = F1Counter(is_fuzzy=True)
            for i in range(len(q_tokens)):
                pre_real = ''.join(p_tokens[i][all_s[i]:all_e[i]])
                glod_real = ''.join(ans_tokens[i][0])
                fuzzy_stats.update(i, glod_real, pre_real)
                strict_stats.update(i, glod_real, pre_real)

            strict_tatal, strict_correct, strict_f1[valid_or_test] = strict_stats.get_f1()
            fuzzy_tatal, fuzzy_correct, fuzzy_f1[valid_or_test] = fuzzy_stats.get_f1()
            self.log.info("{} strict correct in {} total samples, F1:{:.5f}".
                          format(strict_correct, strict_tatal, strict_f1[valid_or_test]))
            self.log.info("{} fuzzy correct in {} total samples, F1:{:.5f}".
                          format(fuzzy_correct, fuzzy_tatal, fuzzy_f1[valid_or_test]))
        return fuzzy_f1['valid'], fuzzy_f1['test'], strict_f1['valid'], strict_f1['test']


    def model_train(self):
        # 训练网络
        bt_loss_log = 500
        bt_valid_log = 1000
        if self.params.debug:
            bt_loss_log = 100
            bt_valid_log = 500
        self.log.info('sample number of supervision learning: {}'.format(len(self.dt.q_c_idxs)))
        for bt in range(self.params.batch_num):
            # 随机生成batch
            self.model.update(self.dt.get_train_batch(batch_size=self.params.batch_size))
            if bt % bt_loss_log == 0:
                # 打印loss的情况。adv的loss设置为永久打印（即使可能不参与优化）
                add_str = '(adv loss is not optimised)' if self.params.loss_type == 'none' else ''
                self.log.info('updates[{0:6}] train loss total[{1:.5f}], none[{2:.5f}], adv[{3:.5f}]'
                              .format(self.model.updates,
                                      self.model.train_loss_total.avg,
                                      self.model.train_loss_none.avg,
                                      self.model.train_loss_adv.avg
                                      ) + add_str)
            if bt % bt_valid_log == 0:
                # 打印evaluate的情况
                dev_fuzzy_f1, test_fuzzy_f1, dev_strict_f1, test_strict_f1 = self.valid_it()
                self.evaluate_log_and_save(dev_fuzzy_f1, test_fuzzy_f1, dev_strict_f1, test_strict_f1, bt)

    def evaluate_log_and_save(self, dev_fuzzy_f1, test_fuzzy_f1, dev_strict_f1, test_strict_f1, bt):
        self.log.info('current fuzzy_f1: {:.5f} (best fuzzy_f1: {:.5f})\tcurrent strict_f1: {:.5f} (best strict_f1: {:.5f})'
                      .format(test_fuzzy_f1, self.best_test_f1, test_strict_f1, self.best_test_strict_f1))
        if self.best_test_strict_f1 < test_strict_f1:
            self.best_test_strict_f1 = test_strict_f1

        if self.best_dev_f1 < dev_fuzzy_f1:
            self.best_dev_f1 = dev_fuzzy_f1
            self.real_test_f1 = test_fuzzy_f1
        self.log.info('best dev f1: {:.5f} (test f1 at that time: {:.5f})'
                      .format(self.best_dev_f1, self.real_test_f1))
        # 保存best_model 和 final_model
        if self.best_test_f1 < test_fuzzy_f1:
            self.best_test_f1 = test_fuzzy_f1
            model_file = os.path.join(self.params.model_dir, 'best_model.pt')
            self.log.info('saved best model to {} in setp {}...'.format(model_file, bt))
            self.model.save(model_file, bt)
        model_file = os.path.join(self.params.model_dir, 'final_model.pt')
        self.model.save(model_file, bt)

    def model_test(self):
        lbs, all_s, all_e, b_pos, e_pos, p_tokens, q_tokens, ans_tokens = \
            self.model.predict_on_valid_or_test_constrained(self.dt, valid_or_test='test', return_p_res=True)
        strict_stats = F1Counter(is_fuzzy=False)
        fuzzy_stats = F1Counter(is_fuzzy=True)
        valid_or_test = 'test'
        strict_f1 = {}
        fuzzy_f1 = {}
        for i in range(len(q_tokens)):
            pre_ans = ' '.join(p_tokens[i][all_s[i]:all_e[i]])
            glod_ans = ' '.join(ans_tokens[i][0])
            pre_real = ''.join(p_tokens[i][all_s[i]:all_e[i]])
            glod_real = ''.join(ans_tokens[i][0])

            fuzzy_correct = fuzzy_stats.update(i, glod_real, pre_real)
            strict_correct = strict_stats.update(i, glod_real, pre_real)
            self.log.info("index:{}\nquestion:{}\npassage:{}\nans:{}\npre:{}\nfuzzy_correct:{}\nstrict_correct:{}\n".
                          format(i, ' '.join(q_tokens[i]), ' '.join(p_tokens[i]), glod_ans,
                                 pre_ans, fuzzy_correct, strict_correct))

        strict_tatal, strict_correct, strict_f1[valid_or_test] = strict_stats.get_f1()
        fuzzy_tatal, fuzzy_correct, fuzzy_f1[valid_or_test] = fuzzy_stats.get_f1()
        strict_result_dict, fuzzy_result_dict = strict_stats.result_dict, fuzzy_stats.result_dict
        result_file = self.params.pretrained_model[:-3] + '_result_dic.pkl'
        self.log.info('Writing pkl file to {}'.format(result_file))
        f = open(result_file, 'wb')
        pickle.dump(strict_result_dict, f)  # 保存序号 以及是否正确，格式： {1:True, 2:False, ...}
        pickle.dump(fuzzy_result_dict, f)

        self.log.info("{} strict correct in {} total samples, F1:{:.5f}".
                      format(strict_correct, strict_tatal, strict_f1[valid_or_test]))
        self.log.info("{} fuzzy correct in {} total samples, F1:{:.5f}".
                      format(fuzzy_correct, fuzzy_tatal, fuzzy_f1[valid_or_test]))

    def test_sample(self, Questions, Passages):
        Ans = []
        q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, q_tokens, p_tokens\
            = self.dt.single_data(Questions, Passages)
        self.model.network.eval()
        all_s, all_e = self.model.predict_all_in_batch(p_c_idxs, qe_comm_ft,
                                                       p_masks, q_c_idxs, q_masks, need_prob_ditri=True)

        all_s_prob, all_s_pos = torch.max(all_s, dim=1)
        all_e_prob, all_e_pos = torch.max(all_e, dim=1)

        if not isinstance(all_s_pos, list):
            all_s_pos = all_s_pos.data.cpu().numpy()
            all_e_pos = all_e_pos.data.cpu().numpy()
            all_s_prob = all_s_prob.data.cpu().numpy()
            all_e_prob = all_e_prob.data.cpu().numpy()

        for p_s, p_e, prob_s, prob_e, p_token in zip(all_s_pos, all_e_pos, all_s_prob, all_e_prob, p_tokens):
            if p_s > p_e:
                p_s, p_e = p_e, p_s
            ans = ''.join(p_token[p_s: p_e])
            start_char_pos = len(''.join(p_token[:p_s]))
            each_ans = {
                "answer": ans,
                "prob": (prob_s + prob_e).item(),
                "start_pos": start_char_pos,
                "end_pos": start_char_pos + len(ans),
            }
            Ans.append(each_ans)
        return Ans


if __name__ == '__main__':
    train = Train(params)
    train.model_train()


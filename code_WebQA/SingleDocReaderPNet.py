import sys
sys.path.append('../')
import logging
import math

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable, grad
from utils import AverageMeter
from AANet import RnnDocReader
np.random.seed(0)

logger = logging.getLogger('mrc')


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, char_embedding, state_dict=None):
        # Book-keeping.
        self.opt = opt
        if opt.loss_type not in ['none', 'adv']:
            raise NotImplementedError('No such loss type: {}'.format(opt.loss_type))
        self.loss_type = opt.loss_type

        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss_total = AverageMeter()
        self.train_loss_none = AverageMeter()
        self.train_loss_adv = AverageMeter()

        # Building network.
        self.network = RnnDocReader(opt, char_embedding=char_embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, opt.learning_rate,
                                       momentum=opt.momentum,
                                       weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt.learning_rate,
                                          weight_decay=opt.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        num_params = sum(p.data.numel()
                         for p in parameters)  #  if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr()
        print(("{} parameters".format(num_params)))

    def update(self, ex):
        """
        :param ex: 样本参数,由DataManager.py的get_train_batch()决定（p_c_idxs_batch, qe_comm_fts_batch, p_masks_batch, q_c_idxs_batch, \
               q_masks_batch, b_batch, e_batch, evd_place_batch）
        """
        # Train mode
        self.network.train()  # 设置为训练模式
        self.num_ex = len(ex[0])  # batch size; ex[0]表示p_c_idxs_batch
        char_keep = self.opt.char_keep  # 此处随机隐藏/置为UNK一些char，也会提升模型效果

        if self.opt.random_char == "PAD":
            x1_c = ex[0] * np.random.choice(2, size=ex[0].shape, p=[1 - char_keep, char_keep])
        elif self.opt.random_char == "UNK":
            rd = np.random.choice(2, size=ex[0].shape, p=[1 - char_keep, char_keep])
            x1_c = ex[0] * rd + (1 - rd)
        elif self.opt.random_char == "none":
            x1_c = ex[0]
        else:
            raise NotImplementedError("random_char error!")

        inputs = [Variable(torch.from_numpy(e).long()) for e in
                  [np.asarray(x1_c)] + list(ex[1:5])]
        g_s, g_e = ex[5], ex[6]
        evd_place = ex[7]

        if self.opt.cuda:
            inputs = [i.cuda(async=True) for i in inputs]


        x1_c, x1_f, x1_mask, x2_c, x2_mask = inputs
        x1_f = x1_f.float()
        x1_mask = x1_mask.byte()
        x2_mask = x2_mask.byte()
        inputs = x1_c, x1_f, x1_mask, x2_c, x2_mask

        s, e, adv_dir = self.network(*inputs)
        loss = self._caculate_loss(inputs, (g_s, g_e), (s, e), adv_dir, evd_place)
        # Compute loss and accuracies
        self.train_loss_total.update(loss.data, self.num_ex)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.opt.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def _caculate_loss(self, inputs, gold_s_e, pred_s_e, adv_dir, evd_place):
        def get_position_variable(g_s, g_e):
            g_s = Variable(torch.from_numpy(np.array(g_s)).long())
            g_e = Variable(torch.from_numpy(np.array(g_e)).long())
            if self.opt.cuda:
                g_s, g_e = g_s.cuda(), g_e.cuda()
            return g_s, g_e

        g_s, g_e = gold_s_e
        s, e = pred_s_e
        loss = 0
        # 采用不同的路径计算loss
        if self.opt.use_ir_evd:
            assert len(g_s) == len(g_e), "Get wrong data!"
            gold_max_len = max([len(i) for i in g_s])
            for i in range(len(g_s)):
                while(len(g_s[i])<gold_max_len):
                    j = random.randint(0, len(g_s[i])-1)
                    g_s[i].append(g_s[i][j])
                    g_e[i].append(g_e[i][j])
            for i in range(gold_max_len):
                g_s_v, g_e_v = get_position_variable([each_s[i] for each_s in g_s], [each_e[i] for each_e in g_e])
                loss += F.cross_entropy(s.squeeze(), g_s_v) + F.cross_entropy(e.squeeze(), g_e_v)
            loss /= gold_max_len

        else:
            g_s_v, g_e_v = get_position_variable(g_s, g_e)
            loss += F.cross_entropy(s.squeeze(), g_s_v) + F.cross_entropy(e.squeeze(), g_e_v)

        self.train_loss_none.update(loss.data, self.num_ex)
        if (self.loss_type == 'adv') or self.opt.always_print_adv_loss:
            p_adv = {}
            p_mult = self.opt.p_mult
            # 综合部分
            grad_list = []  # 需要求导的变量列表
            grad_strs = []  # 需要求导的变量名称列表
            for point_str in p_mult.keys():
                grad_strs.append(point_str)
                grad_list.append(adv_dir[point_str])
            if self.opt.add_random_noise:  # 随机高斯噪声
                mean = [0]
                cov = [[1]]
                for i, grad_str in enumerate(grad_strs):
                    var_in = adv_dir[grad_str]
                    noise = np.random.multivariate_normal(mean, cov, size=var_in.size())
                    noise = np.asarray(noise).reshape(var_in.size())
                    p_adv[grad_str] = torch.FloatTensor(p_mult[grad_str] * _l2_normalize_2(noise, var_in))
                    if self.opt.cuda:
                        p_adv[grad_str] = p_adv[grad_str].cuda(async=True)
                    p_adv[grad_str] = Variable(p_adv[grad_str])
            else:  # 根据导数计算adv噪声
                emb_grad = grad(loss, tuple(grad_list), retain_graph=True)
                # assert self.opt.loss_calculate in ['norm', 'relative', 'part_relative'], "Error in loss_calculate."
                if not self.opt.use_ir_evd:  # single evidence
                    for i, grad_str in enumerate(grad_strs):
                        p_adv[grad_str] = torch.FloatTensor(
                            p_mult[grad_str] * _l2_normalize_2(emb_grad[i].data, grad_list[i]))
                        if self.opt.cuda:
                            p_adv[grad_str] = p_adv[grad_str].cuda(async=True)
                        p_adv[grad_str] = Variable(p_adv[grad_str])
                else:
                    for i, grad_str in enumerate(grad_strs):
                        if "_c_emb" in grad_str:
                            grad_data = get_np(emb_grad[i].data)
                            grad_shape = grad_data.shape
                            word_len = int(grad_shape[0] / self.opt.batch_size)
                            grad_data = grad_data.reshape(self.opt.batch_size, word_len, -1)
                            var_data = get_np(grad_list[i]).reshape(self.opt.batch_size, word_len, -1)
                            perturbation = _l2_normalize_part_relative(grad_data, var_data, evd_place)
                            p_adv[grad_str] = torch.FloatTensor(
                                p_mult[grad_str] * perturbation.view(grad_shape))
                        else:
                            p_adv[grad_str] = torch.FloatTensor(
                                p_mult[grad_str] * _l2_normalize_part_relative(emb_grad[i].data, grad_list[i], evd_place))
                        if self.opt.cuda:
                            p_adv[grad_str] = p_adv[grad_str].cuda(async=True)
                        p_adv[grad_str] = Variable(p_adv[grad_str])
            s_adv, e_adv, _ = self.network(*inputs, p_adv)
            adv_loss = 0
            if type(g_s[0]) == list:  # 说明引入多段，有多个golden truth
                for i in range(gold_max_len):
                    g_s_v, g_e_v = get_position_variable([each_s[i] for each_s in g_s], [each_e[i] for each_e in g_e])
                    adv_loss += F.cross_entropy(s_adv.squeeze(), g_s_v) + F.cross_entropy(e_adv.squeeze(), g_e_v)
                adv_loss /= gold_max_len
            else:
                g_s, g_e = get_position_variable(g_s, g_e)
                adv_loss = F.cross_entropy(s_adv.squeeze(), g_s) + F.cross_entropy(e_adv.squeeze(), g_e)
            self.train_loss_adv.update(adv_loss.data, self.num_ex)
            if self.loss_type == 'adv':  # 只有在'adv'时才真正优化adv_loss
                loss += self.opt.adv_loss_weight * adv_loss
        return loss


    def predict_on_valid_or_test(self, dt, valid_or_test ='valid'):
        # do valid on the whole valid set
        # 把valid data 分batch
        q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, b_pos, e_pos, p_tokens = \
                dt.valid_dat if valid_or_test == 'valid' else dt.test_dat
        all_s, all_e = self.predict_all_in_batch(p_c_idxs, qe_comm_ft, p_masks, q_c_idxs, q_masks)
        lbs = self.old_eval(all_s, all_e, p_tokens=p_tokens)
        return lbs

    def get_eval_acc(self, b_pos, e_pos, all_s, all_e):
        pos_num = 0
        for g_b, g_e, p_b, p_e in zip(b_pos, e_pos, all_s, all_e):
            if g_b == p_b and g_e == p_e:
                pos_num += 1
        return pos_num * 1.0 / len(b_pos)

    def old_eval(self, all_s, all_e, p_tokens):
        all_lbs = self.point_rse2lbs(all_e=all_e, all_s=all_s, p_tokens=p_tokens)
        return all_lbs

    def predict_on_valid_or_test_constrained(self, dt, valid_or_test='valid', return_p_res=False):
        noise = True if 'noise' in valid_or_test else False
        if 'valid' in valid_or_test:
            dataset = dt.valid_dat
        elif 'test' in valid_or_test:
            dataset = dt.test_dat
        else:
            raise NotImplementedError('No such kind of dataset!')
        q_c_idxs, q_masks, p_c_idxs, p_masks, qe_comm_ft, \
        b_pos, e_pos, p_tokens, q_tokens, ans_tokens, _ = dataset
        all_s, all_e = self.predict_all_in_batch(p_c_idxs, qe_comm_ft,  p_masks, q_c_idxs,
                                                 q_masks, need_prob_ditri=True, noise=noise)

        all_s, all_e = self.constrained_infer(all_s, all_e)
        # print("len(all_s), len(all_e):\n", len(all_s), len(all_e))
        lbs = self.old_eval(all_s, all_e, p_tokens=p_tokens)
        if return_p_res:
            return lbs, all_s, all_e, b_pos, e_pos, p_tokens, q_tokens, ans_tokens
        return lbs

    def constrained_infer(self, all_s, all_e, need_score=False):
        s_poses, e_poses = [], []

        all_s_pos_vs, all_s_pos = torch.max(all_s, dim=1)
        all_e_pos_vs, all_e_pos = torch.max(all_e, dim=1)
        all_s_pos, all_e_pos = all_s_pos.data.cpu().numpy(), all_e_pos.data.cpu().numpy()
        all_s_pos_vs, all_e_pos_vs = all_s_pos_vs.data.cpu().numpy(), all_e_pos_vs.data.cpu().numpy()
        infer_num = 0
        scores = []
        for idx, (p_s, p_e, glb_s, glb_e) in enumerate(zip(all_s, all_e, all_s_pos, all_e_pos)):
            if glb_s <= glb_e and glb_s+10 >= glb_e:
                s_poses.append(glb_s)
                e_poses.append(glb_e)
                scores.append(float(all_s_pos_vs[idx] + all_e_pos_vs[idx]))
                continue
            infer_num += 1
            pair_score = {}
            s_vs, cdt_s_poses = torch.topk(p_s, k=5)
            s_vs = s_vs.data.cpu().numpy()
            cdt_s_poses = cdt_s_poses.data.cpu().numpy()
            p_e = p_e.data.cpu().numpy()
            for s_v, cdt_s in zip(s_vs, cdt_s_poses):
                part_p_e = list(p_e[cdt_s: cdt_s+10])
                e_v = max(part_p_e)
                e = cdt_s + part_p_e.index(e_v)
                pair_score[(cdt_s, e)] = float(s_v) + float(e_v)
            max_score = max(pair_score.values())
            s, e = [pair for pair, score in pair_score.items() if score == max_score][0]
            s_poses.append(s)
            e_poses.append(e)
            scores.append(max_score)
        if need_score:
            return s_poses, e_poses, scores
        return s_poses, e_poses


    def point_rse2lbs(self, all_s, all_e, p_tokens):
        if not isinstance(all_s, list):
            all_s = all_s.data.cpu().numpy()
            all_e = all_e.data.cpu().numpy()
        all_lbs = []
        for s_idx, e_idx, st_token in zip(all_s, all_e, p_tokens):
            for idx, token in enumerate(st_token):
                if idx == s_idx and e_idx >= s_idx:
                    all_lbs.append(0)
                elif s_idx < idx < e_idx:
                    all_lbs.append(1)
                else:
                    all_lbs.append(2)
        return all_lbs

    def predict_batch(self, ex, gpu=True, noise=False):
        self.network.eval()
        inputs = [Variable(torch.from_numpy(e).long(), volatile=True) for e in ex[:5]]
        if gpu:
            inputs = [i.cuda(async=True) for i in inputs]
        x1_c, x1_f, x1_mask, x2_c, x2_mask = inputs
        x1_f = x1_f.float()
        x1_mask = x1_mask.byte()
        x2_mask = x2_mask.byte()
        inputs = x1_c, x1_f, x1_mask, x2_c, x2_mask
        self.network.eval()
        p_adv = None
        with torch.no_grad():
            s, e, adv_dir = self.network(*inputs, p_adv)
        if noise:
            p_mult = self.opt.p_mult_eval
            p_adv = {}
            var_str = self.opt.adv_part_var_str
            mean = [0]
            cov = [[1]]
            var_in = adv_dir[var_str]
            noise = np.random.multivariate_normal(mean, cov, size=var_in.size())
            noise = np.asarray(noise).reshape(var_in.size())
            p_adv[var_str] = torch.FloatTensor(p_mult * _l2_normalize_2(noise, var_in))
            if self.opt.cuda:
                p_adv[var_str] = p_adv[var_str].cuda(async=True)
            p_adv[var_str] = Variable(p_adv[var_str])
            s, e, _ = self.network(*inputs, p_adv)
        s = F.softmax(s.view(x1_c.size(0), x1_c.size(1)))
        e = F.softmax(e.view(x1_c.size(0), x1_c.size(1)))
        return s, e


    def predict_all_in_batch(self, p_c_idxs, qe_comm_ft, p_masks, q_c_idxs, q_masks,
                             need_prob_ditri=False, gpu=torch.cuda.is_available(), noise=False):
        batch_num = math.ceil(len(qe_comm_ft)/self.opt.valid_batch_size)
        all_s, all_e = [], []
        for batch_idx in range(int(batch_num)+1):
            batch_end = min((batch_idx + 1) * self.opt.valid_batch_size, len(qe_comm_ft))
            indexes = np.arange(batch_idx*self.opt.valid_batch_size, batch_end)
            if len(indexes) == 0:
                break
            inputs = [i[indexes] for i in (p_c_idxs, qe_comm_ft, p_masks, q_c_idxs, q_masks)]
            s, e = self.predict_batch(inputs, gpu=gpu, noise=noise)
            all_s.append(s)
            all_e.append(e)

        all_s = torch.cat(all_s, dim=0)
        all_e = torch.cat(all_e, dim=0)
        bs, p_len = all_s.size()
        all_s = all_s.view(bs, p_len)
        all_e = all_e.view(bs, p_len)
        if need_prob_ditri:
            return all_s, all_e
        _, all_s_pos = torch.max(all_s, dim=1)
        _, all_e_pos = torch.max(all_e, dim=1)
        return all_s_pos, all_e_pos

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt.tune_partial > 0:
            offset = self.opt.tune_partial + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()

def get_np(x):
    if isinstance(x, Variable):
        x = x.data.cpu().numpy()
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.cuda.FloatTensor):
        x = x.cpu().numpy()
    return x


def _l2_normalize_part_relative(grad, original_var, evd_place):
    def part_normal(grad):
        grad /= (np.sqrt(np.sum(grad ** 2, axis=1)).reshape((grad.shape[0], 1)) + 1e-16)
        return grad
    batch_size, passage_len, hidden_dim = grad.shape
    part_sample_grad = []
    grad = get_np(grad)
    original_var = get_np(original_var)
    # print("grad.shape:", grad.shape)
    # print("original_var.shape:", original_var.shape)
    for sample_id in range(batch_size):
        part_grad = []
        # print("sample_id:", sample_id)
        for i, place in enumerate(evd_place[sample_id]):
            start_len = 0 if i == 0 else evd_place[sample_id][i-1]
            end_len = evd_place[sample_id][i]
            end_len = min(end_len, passage_len)
            part_grad.append(part_normal(grad[sample_id, start_len:end_len, :]))
        part_sample_grad.append(np.concatenate(part_grad, axis=0)[np.newaxis, :, :])
    part_sample_grad_full = []
    for sample_grad in part_sample_grad:
        real_len = sample_grad.shape[1]
        part_sample_grad_full.append(np.concatenate([sample_grad, np.zeros([1, passage_len - real_len, hidden_dim])],
                                                    axis=1))
    grad = np.concatenate(part_sample_grad_full, axis=0)
    grad *= (
    np.sqrt(np.sum(original_var ** 2, axis=2)).reshape((original_var.shape[0], original_var.shape[1], 1)) + 1e-16)
    return torch.from_numpy(grad).float()


def _l2_normalize_2(grad, original_var):
    grad = get_np(grad)
    original_var = get_np(original_var)
    grad /= (np.sqrt(np.sum(grad ** 2, axis=2)).reshape((grad.shape[0], grad.shape[1], 1)) + 1e-16)
    grad *= (np.sqrt(np.sum(original_var ** 2, axis=2)).
             reshape((original_var.shape[0], original_var.shape[1], 1)) + 1e-16)
    return torch.from_numpy(grad).float()
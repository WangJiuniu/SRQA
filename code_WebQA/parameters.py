# -*- encoding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

class ModelParameters:
    def __init__(self):
        train_idx = 1
        self.p_max_len = 100
        self.debug = False  # 是否只录入部分训练数据(500条)
        self.IS_TEST = False
        self.use_ir_evd = False  # 是否使用ir得来的判据
        self.use_refined_flag = True  # 对训练集的flag进行了优化
        self.data_rate = 1
        self.Prefer_SRU = True
        self.hidden_size = 64  # hidden size of each RNN
        self.char_emb_dim = 64
        self.loss_type = 'none'  # ['none', 'adv']
        self.adv_loss_weight = 1  # 小于1时，表示降低adv_loss在训练过程中的权重
        self.char_keep = 0.93
        self.random_char = "none"  # ["UNK", "PAD", "none"]
        self.add_random_noise = False  # 作为adv的替代。default: False
        self.p_mult = {"x1_emb": 2e-4,
                       "p0": 0.5e-4
                       }

        self.show_noise_eval = False  # 是否测试噪声冲击下的test
        self.p_mult_eval = 1e-1

        # 继续训练模型的参数
        self.resume_flag = False  # 是否从文件载入模型
        self.pretrained_model = '../model/model_WebQA.pt'
        # 'checkpoint_step_{}.pt' 'final_model.pt'  'best_model.pt'

        self.use_qemb = True  # emb_att
        self.use_bi_direction_att = True
        self.use_qhiden_match_att = True
        self.use_self_attention = True

        self.always_print_adv_loss = False  # 注意：此时平白多了一些计算量  default: False (做结构ablation时需要false)

        # 正则
        self.dropout_rnn = 0.1
        self.dropout_rnn_output = True  # RNN 的输出上加的drop
        self.dropout_emb = 0

        self.batch_size = 8
        self.train_idx = train_idx
        self.kept_train_rate = 1.0  # 保持训练集的数据量（可选择载入小量数据:如0.5）
        self.q_max_len = 20

        self.char_max_len = 3

        # 文件保存
        for dir in ['../model', '../logs']:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.model_dir = '../model/model_{:0>3}/'.format(train_idx)
        self.log_file = '../logs/models_{:0>3}.log'.format(train_idx)

        if self.IS_TEST:
            self.log_file = self.log_file[:-4] + '_test.log'

        self.q_pre_dim = 200
        self.p_pre_dim = self.q_pre_dim

        self.match_lstm_dim = 150   # fts dim
        self.ft_cnn_size = 100

        # about training
        self.batch_num = 10000000  # 设的很大
        self.valid_batch_size = 128
        self.kernel_num = 100
        self.kernel_sizes = [1, 3, 5]

        self.pretrained_words = True
        self.fix_embeddings = True
        self.tune_partial = 0
        self.emb_dim = 64
        self.embedding_dim = self.emb_dim

        self.char_vocab_file = '../data/chars.txt'

        # model structure relevant
        self.doc_layers = 4
        self.question_layers = 4
        self.fusion_layers = 2

        self.concat_rnn_layers = False
        self.num_features = 1  #
        self.concat_rnn_layers = False
        self.use_interaction = True
        self.res_net = False

        # 优化器参数
        self.optimizer = 'adamax'
        self.learning_rate = 0.001
        self.weight_decay = 0
        self.momentum = 0
        self.grad_clipping = 20
        self.reduce_lr = 0.0

        self.cuda = torch.cuda.is_available()





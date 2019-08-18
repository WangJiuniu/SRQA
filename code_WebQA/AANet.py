import sys
sys.path.append('../')
import torch
import torch.nn as nn
import normal_layers as layers
import torch.nn.functional as F
import bidaf_layers as L


def normalize_emb_(data):
    norms = data.norm(2, 1) + 1e-8
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    data.div_(norms.expand_as(data))
    return data

class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt, char_embedding, padding_idx=0, normalize_emb=False):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt
        char_embedding = torch.FloatTensor(char_embedding)
        if normalize_emb:  # 可以选择是否归一化
            char_embedding = normalize_emb_(char_embedding)
        self.char_embedding = nn.Embedding(char_embedding.size(0), char_embedding.size(1), padding_idx=padding_idx)
        self.char_embedding.weight.data = char_embedding
        # Projection for attention weighted question
        if opt.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(opt.char_emb_dim * 2)
        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt.char_emb_dim * 2 + opt.num_features
        if opt.use_qemb:
            doc_input_size += opt.char_emb_dim * 2
        if opt.use_qhiden_match_att:
            self.qhiden_match_qa = layers.SeqAttnMatch(opt.hidden_size*2)  # 此处用于encoder后的交互

        self.char_rnn = layers.StackedBRNN(
            input_size=opt.char_emb_dim,
            hidden_size=opt.char_emb_dim, # 还保留同样的size吧
            num_layers=1,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
        )

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt.hidden_size,
            num_layers=opt.doc_layers,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net
        )
        question_input_size = opt.char_emb_dim * 2  # 因为char_rnn取双向rnn，故 ×2
        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=question_input_size,
            hidden_size=opt.hidden_size,
            num_layers=opt.question_layers,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net
        )

        if self.opt.use_bi_direction_att:
            self.bi_att = L.BiAttentionLayerMine(opt.hidden_size * 2, opt.hidden_size * 2)
        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt.hidden_size
        question_hidden_size = 2 * opt.hidden_size
        if opt.concat_rnn_layers:
            doc_hidden_size *= opt.doc_layers
            question_hidden_size *= opt.question_layers

        att_out_dim = 0
        if self.opt.use_qhiden_match_att:
            att_out_dim += 2 * opt.hidden_size
        if self.opt.use_bi_direction_att:
            att_out_dim += 2 * opt.hidden_size * 3
        self.fusion_rnn = layers.StackedBRNN(
            input_size=att_out_dim,
            hidden_size=opt.hidden_size,
            num_layers=opt.fusion_layers,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net
        )
        doc_fusion_size = 2 * opt.hidden_size
        self.self_attention = layers.SeqAttnMatch(doc_fusion_size, identity=True)  # 不使用线性层,直接点乘就很好
        doc_self_size = 2 * opt.hidden_size
        match_in_dim = doc_self_size
        if self.opt.use_qhiden_match_att:
            match_in_dim += question_hidden_size

        self.s_linear = nn.Linear(match_in_dim, 1)
        self.e_linear = nn.Linear(match_in_dim + 1, 1)

    def forward(self, x1_c, x1_f, x1_mask, x2_c, x2_mask, p_adv=None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]   # 加一个吧
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        adv_dir = {}
        use_adv = True if self.opt.loss_type == 'adv' else False
        batch_size, x1_seq_len, char_num = x1_c.size()
        x2_seq_len = x2_c.size(1)

        x1_c = x1_c.view(batch_size*x1_seq_len, char_num)
        x2_c = x2_c.view(batch_size*x2_seq_len, char_num)   # bs * seq_len, char_num

        x1_c_emb = self.char_embedding(x1_c)
        x2_c_emb = self.char_embedding(x2_c)
        if use_adv:
            adv_dir['x1_c_emb'] = x1_c_emb
            if p_adv is not None and 'x1_c_emb' in p_adv:
                x1_c_emb += p_adv['x1_c_emb'].contiguous()

            adv_dir['x2_c_emb'] = x2_c_emb
            if p_adv is not None and 'x2_c_emb' in p_adv:
                x2_c_emb += p_adv['x2_c_emb']

        x1_c_emb_processed = self.char_rnn(x1_c_emb)  # bs * seq_len, char_num, emb_dim
        x2_c_emb_processed = self.char_rnn(x2_c_emb)  # bs * seq_len, char_num, emb_dim
        c_emb_dim_processed = x1_c_emb_processed.size(2)
        x1_emb = F.max_pool1d(x1_c_emb_processed.transpose(1, 2), kernel_size=char_num).contiguous().view(batch_size, x1_seq_len, c_emb_dim_processed)
        x2_emb = F.max_pool1d(x2_c_emb_processed.transpose(1, 2), kernel_size=char_num).contiguous().view(batch_size, x2_seq_len, c_emb_dim_processed)



        x1_f = x1_f.unsqueeze(2)
        if self.opt.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt.dropout_emb, training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt.dropout_emb, training=self.training)
        if use_adv:
            adv_dir['x1_emb'] = x1_emb
            if p_adv is not None and 'x1_emb' in p_adv:
                x1_emb += p_adv['x1_emb']
            adv_dir['x2_emb'] = x2_emb
            if p_adv is not None and 'x2_emb' in p_adv:
                x2_emb += p_adv['x2_emb']

        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        if self.opt.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            if use_adv:
                adv_dir['x2_weighted_emb'] = x2_weighted_emb
                if p_adv is not None and 'x2_weighted_emb' in p_adv:
                    x2_weighted_emb += p_adv['x2_weighted_emb']

            drnn_input_list.append(x2_weighted_emb)


        drnn_input = torch.cat(drnn_input_list, 2)
        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input)
        if use_adv:
            adv_dir['doc_hiddens'] = doc_hiddens
            if p_adv is not None and 'doc_hiddens' in p_adv:
                doc_hiddens += p_adv['doc_hiddens']

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb)
        p0_list = []
        if self.opt.use_qhiden_match_att:
            question_hidden_expanded = self.qhiden_match_qa(doc_hiddens, question_hiddens, x2_mask)
            adv_dir['question_hidden_expanded'] = question_hidden_expanded
            if p_adv is not None and 'question_hidden_expanded' in p_adv:
                question_hidden_expanded += p_adv['question_hidden_expanded']
            p0_list.append(question_hidden_expanded)
        # 下面是representation的attention
        if self.opt.use_bi_direction_att:
            att_doc_hiddens, att_question_hiddens = self.bi_att(doc_hiddens, question_hiddens)
            p0_list.append(doc_hiddens)
            p0_list.append(att_question_hiddens)
            p0_list.append(torch.mul(doc_hiddens, att_doc_hiddens))
        p0 = torch.cat(p0_list, 2)
        if use_adv:
            adv_dir['p0'] = p0
            if p_adv is not None and 'p0' in p_adv:
                p0 += p_adv['p0']

        doc_fusion = self.fusion_rnn(p0)
        if self.opt.use_qhiden_match_att:
            if self.opt.use_self_attention:
                doc_self = self.self_attention(doc_fusion, doc_fusion, x1_mask)
                match_in = torch.cat([doc_self, question_hidden_expanded], 2)
            else:
                match_in = torch.cat([doc_fusion, question_hidden_expanded], 2)
        else:
            if self.opt.use_self_attention:
                doc_self = self.self_attention(doc_fusion, doc_fusion, x1_mask)
                match_in = doc_self
            else:
                match_in = doc_fusion
        if use_adv:
            adv_dir['match_in'] = match_in
            if p_adv is not None and 'match_in' in p_adv:
                match_in += p_adv['match_in']

        s = self.s_linear(match_in)
        match_in_e = torch.cat([match_in, s], dim=2)
        e = self.e_linear(match_in_e)
        return s, e, adv_dir
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from parameters import ModelParameters
from sru import SRU, SRUCell
opt = ModelParameters()
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.GRU,  # rnn_type could be nn.LSTM or nn.GRU
                 concat_layers=False, res_net=False, get_all_layers=False):
        super(StackedBRNN, self).__init__()
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.res_net = res_net
        self.concat_layers = concat_layers
        self.get_all_layers = get_all_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            if opt.Prefer_SRU:
                self.rnns.append(SRUCell(input_size, hidden_size,
                                            dropout=dropout_rate,
                                            rnn_dropout=dropout_rate,
                                            use_tanh=True,
                                            bidirectional=True))
            else:
                self.rnns.append(rnn_type(input_size, hidden_size,
                                          num_layers=1,
                                          bidirectional=True))

    def forward(self, x):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        # if x_mask.data.sum() == 0:
        #     return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        #if self.padding or not self.training:
        #    return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x)

    def _forward_unpadded(self, x):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
#            if self.dropout_rate > 0:
#                rnn_input = F.dropout(rnn_input,
#                                      p=self.dropout_rate,
#                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        elif self.get_all_layers:
            output = outputs[1:]
        elif self.res_net:
            output = outputs[1]
            for o in outputs[2:]:
                output = output + o
        else:
            output = outputs[-1]
        if self.get_all_layers:
            # Transpose back
            output = [o.transpose(0, 1) for o in output]
            if self.dropout_output and self.dropout_rate > 0:
                output = [F.dropout(o, p=self.dropout_rate, training=self.training).contiguous() for o in output]
        else:
            output = output.transpose(0, 1)
            # Dropout on output layer
            if self.dropout_output and self.dropout_rate > 0:
                output = F.dropout(output, p=self.dropout_rate, training=self.training)
                output = output.contiguous()
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask, need_attention=False):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.contiguous().view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.contiguous().view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        # 都不加啥transform吗？  dot product; bilinear form; additive projection  后面这两种要不要加一下？

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float(2*10**38))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        if not need_attention:
            return matched_seq
        return matched_seq, alpha


class SeqAttnWeights(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, x_input_size, y_input_size, mapped_dim):
        super(SeqAttnWeights, self).__init__()
        self.mapped_dim = mapped_dim
        self.linear1 = nn.Linear(x_input_size, mapped_dim)
        self.linear2 = nn.Linear(y_input_size, mapped_dim)
        self.o_liner = nn.Linear(self.mapped_dim, 1)

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        bs, len_x, dim_x = x.size()
        len_y = y.size(1)
        x_proj = self.linear1(x.view(-1, x.size(2))).view(bs, len_x, self.mapped_dim)
        x_proj = F.relu(x_proj)   # 这个激活函数合适吗？
        y_proj = self.linear2(y.view(-1, y.size(2))).view(bs, len_y, self.mapped_dim)
        y_proj = F.relu(y_proj)

        x_exp = x_proj.unsqueeze(2).expand(bs, len_x, len_y, self.mapped_dim)
        y_exp = y_proj.unsqueeze(1).expand(bs, len_x, len_y, self.mapped_dim)
        raw_att_vs = self.o_liner(x_exp + y_exp).view(bs, len_x, len_y)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(raw_att_vs.size())
        raw_att_vs.data.masked_fill_(y_mask.data, -float(2*10**38))
        # 加一个fill nan with zero

        # Normalize with softmax
        alpha_flat = F.softmax(raw_att_vs.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, len_x, len_y)

        # i need bs * len1 * len2
        return alpha


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float(2*10**38))
        alpha = F.sigmoid(xWy)
        return alpha
        # if self.training:
        #     # In training we output log-softmax for NLL
        #     alpha = F.log_softmax(xWy)
        # else:
        #     # ...Otherwise 0-1 probabilities
        #     alpha = F.softmax(xWy)
        # return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float(2*10**38))
        alpha = F.softmax(scores)
        return alpha


class LinearSeqAttnNoMask(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttnNoMask, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        alpha = F.softmax(scores)
        return alpha


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1, keepdim=True).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)

import math
from telnetlib import NEW_ENVIRON
from numpy.lib.twodim_base import mask_indices
import torch
from torch._C import device
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import utils
from model_PREVALENT import BertLayerNorm
import pdb



class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftLabelCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        # with torch.no_grad():
        #     true_dist = torch.zeros_like(pred)
        #     true_dist.fill_(self.smoothing / (self.cls - 1))
        #     true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return -target * pred




class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class CrossModalPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(CrossModalPositionalEmbedding, self).__init__()
        self.modality_embeddings = nn.Embedding(2, hidden_size) # [IMG], [TXT]
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vision, language):
        """
        Args:
            input_feat: (N, L, D)
        """
        b, lv, d = vision.size()
        visual_ids = torch.zeros((b, lv), dtype=torch.long, device=vision.device)
        visual_encoding = self.modality_embeddings(visual_ids)

        if language is None:
            return vision+visual_encoding, language        
        else:
            b, lt, d = language.size()
            text_ids = torch.ones((b, lt), dtype=torch.long, device=language.device)
            text_encoding = self.modality_embeddings(text_ids)
            return vision+visual_encoding, language+text_encoding


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class BEncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1,bert=None,update=False):
        super(BEncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in BEncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        #self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.bert = bert
        self.update = update
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths,att_mask=None,img_feats=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        #embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        assert att_mask is not None
        seq_max_len = att_mask.size(1)
        outputs = self.bert(inputs[:,:seq_max_len], attention_mask=att_mask, img_feats=img_feats)  # (batch, seq_len, embedding_size)
        embeds = outputs[0]
        if not self.update:
            embeds = embeds.detach()
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class CEncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1,bert=None,update=False,bert_hidden_size=768):
        super(CEncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in CEncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.linear_in = nn.Linear(bert_hidden_size, embedding_size)
        self.bert = bert
        self.update = update
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths,att_mask=None,img_feats=None):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        #embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        assert att_mask is not None
        seq_max_len = att_mask.size(1)
        outputs = self.bert(inputs[:,:seq_max_len], attention_mask=att_mask, img_feats=img_feats)  # (batch, seq_len, embedding_size)
        bertembeds = outputs[0]
        if not self.update:
            bertembeds = bertembeds.detach()
        embeds = self.linear_in(bertembeds)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True, stop=False):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if stop:
            pdb.set_trace()
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        if stop:
            pdb.set_trace()
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde

class BAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(BAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size * 2)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.dim = args.critic_dim
        self.state2value = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        
    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        ctx_mask = utils.length2mask(lengths)

        return x, ctx_mask


class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search
        
        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class BertSpeakerEncoder(SpeakerEncoder):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional, vocab_size, embedding_size, padding_idx):
        super().__init__(feature_size, hidden_size, dropout_ratio, bidirectional)
        # TODO: parameters
        self.visual_feat_dim = feature_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.nhead = args.nhead
        self.dim_feedforward = args.dim_feedforward
        self.num_layers = args.num_layers 

        self.p_drop = args.encoder_drop
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size # TODO:
        self.padding_idx = padding_idx

        # layers by order
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.positional_encoding = PositionEncoding(self.hidden_size)
        self.modality_embedding = CrossModalPositionalEmbedding(self.hidden_size, dropout=self.p_drop)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.visual_proj = nn.Sequential(
            LayerNorm(self.visual_feat_dim),
            nn.Dropout(self.p_drop),
            nn.Linear(self.visual_feat_dim, self.hidden_size),
            nn.ReLU(True),
            LayerNorm(self.hidden_size),
            nn.Dropout(self.p_drop),
        )

        self.word_proj = nn.Sequential(
            LayerNorm(self.embedding_size),
            nn.Dropout(self.p_drop),
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(True),
            LayerNorm(self.hidden_size),
            nn.Dropout(self.p_drop)
        )

        # construct transformer layers
        layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.p_drop) # TODO:
        self.layers = nn.TransformerEncoder(encoder_layer=layer, num_layers=self.num_layers) # TODO:
        
    def forward(self, action_embeds, feature, lengths, already_dropfeat=False, instr_encoding=None, multimodal_input=False):
        assert (multimodal_input and instr_encoding is not None) or (not multimodal_input and instr_encoding is None)
        # visual part
        # adopted from SpeakerEncoder
        x = action_embeds

        ctx = self.visual_proj(x) # contain drop

        batch_size, max_length, _ = ctx.size()

        feature = F.dropout(feature, p=self.p_drop)
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = F.dropout(x, p=self.p_drop)

        # get mask
        ctx_mask = utils.length2mask(lengths, size=max_length)

        # instr part
        if multimodal_input:
            embeds = self.embedding(instr_encoding)
            y = self.word_proj(embeds)
            y_mask = self.words_mask(instr_encoding)
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
            cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=y_mask.device)
            
            # TODO: add cross modal embedding
            x, y = self.modality_embedding(x, y)
            x = torch.cat([x, cls_tokens, y], dim=1) # (B, L, D)
            ctx_mask = torch.cat([ctx_mask, cls_mask, y_mask], dim=1) # (B, L)
        else:
            x, _ = self.modality_embedding(x, None)

        x = self.positional_encoding(x)

        out = self.layers(x.transpose(1, 0), src_key_padding_mask=ctx_mask)
        out = out.transpose(1, 0)
        out = F.dropout(out, p=self.p_drop)

        return out, ctx_mask
    
    def words_mask(self, words):
        return torch.tensor(words==self.embedding.padding_idx, dtype=torch.bool, device=words.device)


class BertSpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        # parameters
        self.dropout_ratio = dropout_ratio
        self.embed_dim = embedding_size
        self.hidden_size = hidden_size
        self.nhead = args.nhead
        
        # TODO: from args
        self.dim_feedforward = args.dim_feedforward
        self.num_layers = args.num_layers
        self.eps = args.layer_norm_eps
        self.p_drop = args.decoder_drop

        # layers by order
        self.positional_encoding = PositionEncoding(self.hidden_size)

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)

        self.word_proj = nn.Sequential(
            LayerNorm(self.embed_dim),
            nn.Dropout(self.p_drop),
            nn.Linear(self.embed_dim, self.hidden_size),
            nn.ReLU(True),
            LayerNorm(self.hidden_size),
            nn.Dropout(self.p_drop),
        )

        layer = nn.TransformerDecoderLayer(self.hidden_size, self.nhead, self.dim_feedforward, dropout=self.p_drop)
        self.layers = nn.TransformerDecoder(layer, num_layers=self.num_layers)

        self.projection = nn.Linear(hidden_size, vocab_size)

        self.drop = nn.Dropout(dropout_ratio) # TODO: redundant

    def forward(self, words, ctx, ctx_mask, *argv):
        # get embedding, proj to hidden
        embeds = self.embedding(words)
        
        x = self.word_proj(embeds)

        # get padding mask
        mask = self.words_mask(words)
        x = self.positional_encoding(x)        

        # transformer decoder
        x = self.layers(x.transpose(1, 0), ctx.transpose(1, 0), tgt_key_padding_mask=mask, memory_key_padding_mask=ctx_mask)
        x = x.transpose(0, 1)
        x = F.dropout(x, p=self.p_drop)
        
        # final proj
        logit = self.projection(x)
        return logit, None, None

    def words_mask(self, words):
        return torch.tensor(words==self.embedding.padding_idx, dtype=torch.bool, device=words.device)


class InstructionClassifier(nn.Module):
    def __init__(self, embedding_size, n_layers, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(embedding_size, n_classes)
    
    def forward(self, x):
        # x is a sequence
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        out = self.fc(x)
        return out


class E2E(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(E2E, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.future_pano_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.future_action_att_layer = SoftDotAttention(hidden_size, args.angle_feat_size)
        self.future_cand_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

        self.state_ctx = SoftDotAttention(hidden_size, hidden_size)
        self.state_cand = SoftDotAttention(hidden_size, feature_size)
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_size, 1 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(1 * hidden_size, 2)
        )

        self.encoder2decoder = nn.Linear(hidden_size, hidden_size)

        self.merge_cand_feats = nn.Sequential(
            nn.Linear(2 * feature_size, feature_size),
            nn.Sigmoid(),
        )

        self.vis_state_norm = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )

        self.lang_state_norm = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )


    def init_h1(self, ctx, ctx_mask):
        ctx_max, _ = (ctx * ctx_mask.unsqueeze(2)).max(1)
        decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        return decoder_init

    def forward(self, vis_state_scores, lang_state_scores, attn_vis, attn_lang):
        batch_size = len(vis_state_scores)

        # vis_state_scores = vis_state_scores.view(batch_size, -1)
        # n_repeat = self.hidden_size // vis_state_scores.size(1)
        # offset = self.hidden_size - n_repeat * vis_state_scores.size(1) 
        # vis_state_scores = torch.cat([vis_state_scores.repeat(1, n_repeat), vis_state_scores[:, :offset]], dim=-1)
        # vis_state_scores = self.vis_state_norm(vis_state_scores)

        # lang_state_scores = lang_state_scores.view(batch_size, -1)
        # n_repeat = self.hidden_size // lang_state_scores.size(1) 
        # offset = self.hidden_size - n_repeat * lang_state_scores.size(1) 
        # lang_state_scores = torch.cat([lang_state_scores.repeat(1, n_repeat), lang_state_scores[:, :offset]], dim=-1)
        # lang_state_scores = self.lang_state_norm(lang_state_scores)

        uncert = self.critic(torch.cat([attn_vis, attn_lang], dim=1))
        
        return uncert

from base64 import decode
from collections import defaultdict
# from ctypes.wintypes import tagRECT
# from lib2to3.pgen2.tokenize import tokenize
# from msilib import sequence
from re import I, X
import torch
import numpy as np
from torch._C import dtype
from param import args
import os
import utils
import model
import torch.nn.functional as F
import pdb


class Speaker():
    env_actions = {
        'left': (0,-1, 0), # left
        'right': (0, 1, 0), # right
        'up': (0, 0, 1), # up
        'down': (0, 0,-1), # down
        'forward': (1, 0, 0), # forward
        '<end>': (0, 0, 0), # <end>
        '<start>': (0, 0, 0), # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, listener, tok):
        self.env = env
        self.feature_size = self.env.feature_size
        self.tok = tok
        # self.tok.finalize()
        self.listener = listener

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size())
                  
        self.encoder, self.decoder, self.goal_decoder, self.lazy_decoder = self.create_model(self.feature_size+args.angle_feat_size, 
                                                                                             args.rnn_dim, 
                                                                                             args.encoder_drop,  # args.dropout / encoder_drop
                                                                                             args.bidir, 
                                                                                             self.tok.vocab_size(), 
                                                                                             args.wemb, 
                                                                                             self.tok.word_to_index['<PAD>']
                                                                                             )
        self.decoder_names = ['speaker', 'goal', 'lazy']

        self.speaker_type = args.speaker_type

        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_params = list(self.decoder.parameters()) + list(self.goal_decoder.parameters()) + list(self.lazy_decoder.parameters())
        self.decoder_optimizer = args.optimizer(self.decoder_params , lr=args.lr)

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])

        # Will be used in beam search
        self.nonreduced_softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tok.word_to_index['<PAD>'],
            size_average=False,
            reduce=False
        )

        self.tasks = args.tasks

    def create_model(self, feature_size, hidden_size, dropout_ratio, bidirectional, vocab_size, embedding_size, padding_idx):
        if args.speaker_type == 'lstm':
            encoder = model.SpeakerEncoder(feature_size, hidden_size, dropout_ratio, bidirectional).cuda()
        else:
            encoder = model.BertSpeakerEncoder(feature_size, hidden_size, dropout_ratio, bidirectional, vocab_size, embedding_size, padding_idx).cuda()

        decoder = model.SpeakerDecoder(vocab_size, embedding_size, padding_idx, hidden_size, args.speaker_drop).cuda()
        goal_decoder = model.SpeakerDecoder(vocab_size, embedding_size, padding_idx,hidden_size, args.goal_drop).cuda()
        lazy_decoder = model.SpeakerDecoder(vocab_size, embedding_size, padding_idx,hidden_size, args.lazy_drop).cuda()
        return encoder, decoder, goal_decoder, lazy_decoder

    def train(self, iters):
        loss_history = {}
        for i in range(iters):
            self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss, loss_dict = self.teacher_forcing(train=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder_params, 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            for key, value in loss_dict.items():
                if key not in loss_history:
                    loss_history[key] = [value]
                else:
                    loss_history[key].append(value)
        return loss_history

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = defaultdict(lambda: {})
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch()  # Get the insts of the result
            instr_ids = [ob['instr_id'] for ob in obs]  # Gather the path ids
            for name in insts.keys():
                for instr_id, inst in zip(instr_ids, insts[name]):
                    instr_id_elem = instr_id.split("_")
                    path_id = int(instr_id_elem[0]) if len(instr_id_elem) == 2 else "_".join(instr_id_elem[:-1])
                    if name == 'speaker':
                        if f"{path_id}_0" not in path2inst[name].keys():
                            path2inst[name][f"{path_id}_0"] = self.tok.shrink(inst)  # Shrink the words
                    else:
                        path2inst[name][instr_id] = self.tok.shrink(inst)
        return path2inst

    def valid(self, *aargs, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(*aargs, **kwargs)

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 1 if args.fast_train else 3     # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        loss_history = {}
        for i in range(N):
            self.env.reset()
            loss_dict, word_acc, sent_acc = self.teacher_forcing(train=False)
            metrics += np.array([loss_dict['speaker'], word_acc['speaker'], sent_acc['speaker']])
            for key, value in loss_dict.items():
                if key in loss_history:
                    loss_history[key].append(value)
                else:
                    loss_history[key] = [value]
        metrics /= N
        return (path2inst, *metrics, loss_history)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        first_feat = np.zeros((len(obs), self.feature_size+args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self.listener._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action
            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats), length

    def gt_words(self, obs, key='speaker_instr_encoding'):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        # NOTE:
        # seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_tensor = np.array([ob[key] for ob in obs])
        return torch.from_numpy(seq_tensor).cuda()

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False, mode='speaker'):
        if train:
            self.encoder.train()
            self.decoder.train()
            self.goal_decoder.train()
            self.lazy_decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.goal_decoder.eval()
            self.lazy_decoder.eval()

        # Get Image Input & Encode
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            assert insts is not None
            (img_feats, can_feats), lengths = features
            ctx, ctx_mask = self.encoder(can_feats, img_feats, lengths)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            (img_feats, can_feats), lengths = self.from_shortest_path()      # Image Feature (from the shortest path)
            ctx, ctx_mask = self.encoder(can_feats, img_feats, lengths) #TODO:
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        # ctx_mask = utils.length2mask(lengths) # mask returned by the encoder

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs, 'speaker_instr_encoding')                                       # Language Feature

        # Decode
        logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)

        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )

        if for_listener:
            return self.nonreduced_softmax_loss(
                input  = logits[:, :, :-1],         # -1 for aligning
                target = insts[:, 1:]               # "1:" to ignore the word <BOS>
            )

        loss_dict = {'speaker': loss.item()}

        if args.tasks == 'all':
            # decode goal and noise-free instr
            goal_loss, recon_loss = self.auxiliary_decode(obs, can_feats, img_feats, lengths, batch_size, insts)
            loss_dict['goal'] = goal_loss.item()
            loss_dict['recon'] = recon_loss.item()
        else:
            goal_loss, recon_loss = 0, 0
            loss_dict['goal'] = goal_loss
            loss_dict['recon'] = recon_loss

        total_loss = loss + args.loss_weight * (goal_loss + recon_loss)
        
        if train:
            return total_loss, loss_dict
        else:
            word_accu, sent_accu = self.calculate_pred_acc(batch_size, [logits], [insts], ["speaker"])
            return loss_dict, word_accu, sent_accu

    def calculate_pred_acc(self, batch_size, logits_list: list, refs_list: list, keys: list):
        word_accs = {}
        sent_accs = {}
        for key, logits, refs in zip(keys, logits_list, refs_list):
            _, predict = logits.max(dim=1)
            gt_mask = (refs != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == refs[:, 1:]) * gt_mask[:, 1:]
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_acc = correct.sum().item() / gt_mask[:, 1:].sum().item()
            sent_acc = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size
            word_accs[key] = word_acc
            sent_accs[key] = sent_acc
        return word_accs, sent_accs

    def auxiliary_decode(self, obs, can_feats, img_feats, lengths, batch_size, insts):
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        # encode multimodal inputs
        noisy_instr = self.insert_anchors(obs)
        multimodal_ctx, multimodal_ctx_mask = self.encoder(can_feats, img_feats, lengths, instr_encoding=noisy_instr, multimodal_input=True)

        # generate goal
        goals = self.gt_words(obs, 'goal_encoding')
        goal_logits, _, _ = self.goal_decoder(goals, multimodal_ctx, multimodal_ctx_mask, h_t, c_t)
        goal_logits = goal_logits.permute(0, 2, 1).contiguous()
        goal_loss = self.softmax_loss(
            input = goal_logits[:, :, :-1],
            target = goals[:, 1:]
        )

        # reconstruct noise-free instructions
        recon_logits, _, _ = self.lazy_decoder(insts, multimodal_ctx, multimodal_ctx_mask, h_t, c_t)
        recon_logits = recon_logits.permute(0, 2, 1).contiguous()
        recon_loss = self.softmax_loss(
            input = recon_logits[:, :, :-1],
            target = insts[:, 1:]
        )
        return goal_loss, recon_loss

    def replace_with_anchors(self, obs):
        def calc_offset(ob, k):
            if 'anchor_offset' in ob:
                k_offset = 0
                if ob['anchor_offset'] == None:
                    pdb.set_trace()
                for offset in ob['anchor_offset']:
                    k_offset += 1 if k >= offset else 0
                return k_offset, len(ob['anchor_offset'])
            else:
                return 0, 0
        
        noisy_instrs = []
        for i, ob in enumerate(obs):
            anchors = ob['anchor_index']
            if len(anchors) <= 1:
                noisy_instrs.append(ob['speaker_instr_encoding'])
                continue

            instr = ob['speaker_instr_encoding']
            path_length = len(ob['exploration_labels'])
            k = np.random.randint(0, high=len(anchors)-1)
            # ks = np.random.choice(len(anchors), size=min(1, len(anchors)), replace=False) # TODO: recurrent replace
            start = anchors[k-1] + 1 if k > 0 else 1 
            end = anchors[k] + 1
            anchored_seg = ob['anchored_encoding'][start:end]
            instr[start:end] = anchored_seg
            noisy_instrs.append(instr)

            k_offset, total_offset = calc_offset(ob, k)
            unit = path_length / (len(anchors) + total_offset)
            exploration_loc = max(0, round((k+1+k_offset) * unit) - 1)

            k_prev, k_after = k-1, max(len(anchors)-1, k+1)
            k_prev_offset, _ = calc_offset(ob, k_prev)
            # prev_exploration_step = min(exploration_step-1, max(0, round((k_prev+1+k_prev_offset) / (len(anchors)+total_offset) * path_length) - 1))
            # k_after_offset, _ = calc_offset(ob, k_after)
            # after_exploration_step = max(exploration_step+1, max(0, round((k_after+1+k_after_offset) / (len(anchors)+total_offset) * path_length) - 1))
            
            # assert prev_exploration_step + 1 < after_exploration_step
            # ob['exploration_labels'][prev_exploration_step+1:after_exploration_step] = 1 # when number of landmarks < path length

            exploration_start = max(0, round((k_prev+1+k_prev_offset) * unit) - 1)            
            if exploration_start == exploration_loc:
                ob['exploration_labels'][exploration_start:exploration_loc+1] = 1
            elif exploration_start < exploration_loc:
                ob['exploration_labels'][exploration_start+1:exploration_loc+1] = 1
            else:
                raise AssertionError

            ob['anchor_offset'] = ob['anchor_offset'] + [k] if 'anchor_offset' in ob else [k] # NOTE: ob['anchor_offset].append(k) results in None, why?

        noisy_instrs = np.array(noisy_instrs)
        return torch.from_numpy(noisy_instrs).cuda()

    def insert_anchors(self, obs):
        noisy_instrs = []
        for ob in obs:
            anchors = ob['anchor_index']
            instr_enc = ob['speaker_instr_encoding']
            # extract an anchor-based phrase by random
            # j = np.random.randint(0, high=len(obs))
            # n_noise = min(len(obs), np.random.randint(1, 4)) # TODO: determine n ank tokens
            n_noise = 1
            js = np.random.choice(len(obs), size=n_noise)
            for j in js:
                anchors = obs[j]['anchor_index']
                while len(anchors) == 0:
                    j = np.random.randint(0, high=len(obs))
                    anchors = obs[j]['anchor_index']

                k = np.random.randint(0, high=len(anchors))
            
                start = anchors[k-1] + 1 if k > 0 else 1
                end = anchors[k] + 1

                phrase = obs[j]['anchored_encoding'][start:end] # e.g.: on the wall / walk up the stairs
                if phrase[0] == self.tok.word_to_index['.']:
                    phrase = phrase[1:]

                # insert into original encodings
                if len(anchors) == 0:
                    insert_loc = 1
                else:
                    insert_loc = np.random.choice(anchors) + 1
                max_len = len(instr_enc)
                instr_enc = np.insert(instr_enc, insert_loc, phrase)[:max_len]
            
            if instr_enc[-1] != self.tok.word_to_index['<PAD>'] and instr_enc[-1] != self.tok.word_to_index['<EOS>']:
                instr_enc[-1] = self.tok.word_to_index['<EOS>']
            noisy_instrs.append(instr_enc)
        
        noisy_instrs = np.array(noisy_instrs)
        return torch.from_numpy(noisy_instrs).cuda()

    def add_anchors(self, obs):
        # get noisy instructions
        if args.train == 'validspeaker' or 'listener' in args.train:
            noisy_instr = self.replace_with_anchors(obs)
        else:
            noisy_instr = self.insert_anchors(obs)
        return noisy_instr

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.encoder.train()
            self.decoder.train()
            self.goal_decoder.train()
            self.lazy_decoder.train()
            pdb.set_trace()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.goal_decoder.eval()
            self.lazy_decoder.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)
        # This code block is only used for the featdrop.
        if featdropmask is not None:
            img_feats[..., :-args.angle_feat_size] *= featdropmask
            can_feats[..., :-args.angle_feat_size] *= featdropmask

        # Encoder
        ctx, ctx_mask = self.encoder(can_feats, img_feats, lengths,
                           already_dropfeat=(featdropmask is not None))
        # ctx_mask = utils.length2mask(lengths)

        if self.tasks == 'all':
            # get noisy instructions
            noisy_instr = self.add_anchors(obs)
            # Encoder again
            multimodal_ctx, multimodal_ctx_mask = self.encoder(can_feats, img_feats, lengths, instr_encoding=noisy_instr, multimodal_input=True)

        words_all = {}
        log_probs_all = {}
        hidden_states_all = {}
        entropies_all = {}
        for name, decoder in zip(self.decoder_names, [self.decoder, self.goal_decoder, self.lazy_decoder]):
            if name == 'speaker':
                words, log_probs, hidden_states, entropies = self.infer_by_decoder(decoder, ctx, ctx_mask, sampling, train)
            else:
                if self.tasks == 'all':
                    words, log_probs, hidden_states, entropies = self.infer_by_decoder(decoder, multimodal_ctx, multimodal_ctx_mask, sampling, train)
                else:
                    continue
            words_all[name] = words
            log_probs_all[name] = log_probs
            hidden_states_all[name] = hidden_states
            entropies_all[name] = entropies

        if train and sampling:
            return words_all, log_probs_all, hidden_states_all, entropies_all
        else:
            return words_all      # [(b), (b), (b), ...] --> [b, l]

    def score_candidates(self, img_feats, can_feats, lengths, instrs):
        self.encoder.eval()
        self.decoder.eval()

        instr_encodings = np.array([self.tok.encode_sentence(instr)[1:] for instr in instrs]) # do not include BOS
        instr_encodings = torch.from_numpy(instr_encodings).long().cuda()

        # Encoder
        ctx, ctx_mask = self.encoder(can_feats, img_feats, lengths, already_dropfeat=False)

        # Decoder
        batch_size = len(ctx)
        words = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        # ended = np.zeros(batch_size, np.bool)
        ended = torch.zeros(batch_size, dtype=torch.bool).cuda()
        word = np.ones(batch_size, np.int64) * self.tok.word_to_index['<BOS>']    # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()
        sequence_scores = torch.zeros(batch_size).cuda()
        outputs = torch.zeros(batch_size).cuda()
        for i in range(args.maxDecode):
            # Decode Step
            logits, h_t, c_t = self.decoder(word.view(-1, 1), ctx, ctx_mask, h_t, c_t)      # Decode, logits: (b, 1, vocab_size) 

            target = instr_encodings[:, i].contiguous()
            word = target

            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            log_probs = F.log_softmax(logits, dim=1)
            word_scores = -F.nll_loss(log_probs, word, ignore_index=self.tok.pad_token_id, reduce=False)
            sequence_scores += word_scores
            
            outputs = (ended * outputs) + (~ended * sequence_scores)
            # ended = np.logical_or(ended, word == self.tok.sep_token_id)
            ended = ended | (word == self.tok.sep_token_id)
            if ended.all():
                break

        return outputs

    def underspecify_instrs(self, sampling=False):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        """
        self.encoder.eval()
        self.decoder.eval()
        self.goal_decoder.eval()
        self.lazy_decoder.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        speak_goal = False if np.random.rand() > 1/3 else True
        if speak_goal:
            words = [ob['goals'] for ob in obs]
            for ob in obs:
                if isinstance(ob['exploration_labels'], list):
                    ob['exploration_labels'] = np.array(ob['exploration_labels'])
                ob['exploration_labels'][:-1] = 1
            exploration_labels =  [ob['exploration_labels'] for ob in obs]
            return words, exploration_labels

        # Get feature
        (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)

        n_iters = np.random.randint(2, 4)
        for i in range(n_iters):
            # get noisy instructions
            noisy_instr = self.replace_with_anchors(obs)

            # Encoder
            multimodal_ctx, multimodal_ctx_mask = self.encoder(can_feats, img_feats, lengths, instr_encoding=noisy_instr, multimodal_input=True)
            
            words, _, _, _ = self.infer_by_decoder(self.lazy_decoder, multimodal_ctx, multimodal_ctx_mask, sampling, train=False)

            for j, (ob, inst) in enumerate(zip(obs, words)):
                ob['instructions'] = self.tok.decode_sentence(self.tok.shrink(inst, strict=False)) # update speaker_instr_encoding!!!
                ob['speaker_instr_encoding'] = self.tok.encode_sentence(ob['instructions'])
                anchor_index, anchored_encoding, anchor_mask = self.env.get_anchor_index(ob['instructions'].lower())
                ob['anchor_index'] = anchor_index
                ob['anchored_encoding'] = anchored_encoding
                ob['anchor_mask'] = anchor_mask

        exploration_labels = [ob['exploration_labels'] for ob in obs]

        words = [self.tok.decode_sentence(self.tok.shrink(word, strict=False)) for word in words]

        return words, exploration_labels    # [(b), (b), (b), ...] --> [b, l]

    def infer_by_decoder(self, decoder, ctx, ctx_mask, sampling, train):
        if decoder.__class__.__name__ == 'SpeakerDecoder':
            words, log_probs, hidden_states, entropies = self.infer_batch_lstm(decoder, ctx, ctx_mask, sampling, train)
        else:
            words, log_probs, entropies = self.infer_batch_transformer(decoder, ctx, ctx_mask, sampling, train)
            hidden_states = None
        return words, log_probs, hidden_states, entropies
        
    def infer_batch_lstm(self, decoder, ctx, ctx_mask, sampling, train):
        # Decoder
        batch_size = len(ctx)
        words = []
        log_probs = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        ended = np.zeros(batch_size, np.bool)
        word = np.ones(batch_size, np.int64) * self.tok.word_to_index['<BOS>']    # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()
        for i in range(args.maxDecode):
            # Decode Step
            logits, h_t, c_t = decoder(word, ctx, ctx_mask, h_t, c_t)      # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits.max(1)
            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.tok.word_to_index['<PAD>']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break
        return np.stack(words, 1), log_probs, hidden_states, entropies

    def infer_batch_transformer(self, decoder, ctx, ctx_mask, sampling, train):
        # Decoder
        batch_size = len(ctx)
        words = torch.ones((batch_size, args.maxDecode+1), dtype=torch.long, device=ctx.device) * self.tok.word_to_index['<PAD>']
        log_probs = []
        entropies = []
        ended = np.zeros(batch_size, np.bool)
        word = torch.ones((batch_size, 1), dtype=torch.long, device=ctx.device) * self.tok.word_to_index['<BOS>']
        for i in range(args.maxDecode):
            words[:, i] = word.view(-1) # resize (b,)

            # Decode Step
            logits, _, _ = decoder(words, ctx, ctx_mask)      # Decode, logits: (b, maxDecode, vocab_size)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            logits[:, :, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits[:, i, :], -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits[:, i, :].max(1)
            # Append the word
            word[ended] = self.tok.word_to_index['<PAD>']
            cpu_word = word.cpu().numpy()

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break
        
        words[:, i+1] = word.view(-1)
        words = words[:, 1:].cpu().numpy() # throw <BOS>
        return words, log_probs, entropies

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("goal_decoder", self.goal_decoder, self.decoder_optimizer),
                     ("lazy_decoder", self.lazy_decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("goal_decoder", self.goal_decoder, self.decoder_optimizer),
                     ("lazy_decoder", self.lazy_decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

    def set_to_eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.goal_decoder.eval()
        self.lazy_decoder.eval()

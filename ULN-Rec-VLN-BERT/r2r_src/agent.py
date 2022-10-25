# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from asyncore import loop
from cProfile import label
import json
import os
from re import L
import sys
from typing_extensions import final
from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
from networkx.algorithms.efficiency_measures import global_efficiency
from nltk import text
import numpy as np
import random
import math
import time
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import copy
from sklearn.preprocessing import power_transform

import torch
from torch._C import dtype
import torch.nn as nn
from torch.autograd import Variable
from torch import is_complex, optim
import torch.nn.functional as F
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss

from env import R2RBatch
from model import E2E, CrossModalPositionalEmbedding, InstructionClassifier
import utils
from utils import padding_idx, print_progress
import model_OSCAR, model_PREVALENT
import param
from param import args
from collections import defaultdict
from tqdm import tqdm
import pdb


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in tqdm(range(iters)):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
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

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        if args.vlnbert == 'oscar':
            self.vln_bert = model_OSCAR.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_OSCAR.Critic().cuda()
        elif args.vlnbert == 'prevalent':
            self.vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_PREVALENT.Critic().cuda()
        self.exploration = E2E(64, self.vln_bert.vln_bert.config.hidden_size, 0.5, feature_size=self.feature_size + args.angle_feat_size).cuda()
        self.explore_critic = model_PREVALENT.Critic().cuda()
        self.classifier = InstructionClassifier(self.vln_bert.vln_bert.config.hidden_size, n_layers=0, n_classes=2).cuda()        
        self.models = (self.vln_bert, self.critic, self.exploration, self.explore_critic, self.classifier)

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.exploration_optimizer = args.optimizer(self.exploration.parameters(), lr=args.lr)
        self.explore_critic_optimizer = args.optimizer(self.explore_critic.parameters(), lr=args.lr)
        self.classifier_optimizer = args.optimizer(self.classifier.parameters(), lr=args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer, self.exploration_optimizer, self.explore_critic_optimizer, self.classifier_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.ndtw_criterion = utils.ndtw_initialize()
        self.bce = BCEWithLogitsLoss()
        self.ce = CrossEntropyLoss()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs, key='instr_encoding'):
        seq_tensor = np.array([ob[key] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != padding_idx)

        token_type_ids = torch.zeros_like(mask)

        if key == 'goal_instr_encoding':
            text_type_ids = []
            for instr in seq_tensor:
                ids = torch.zeros_like(instr)
                cls_locs = torch.where(instr == self.tok.tokenizer.cls_token_id)[0]
                goal_locs = torch.where(instr == self.tok.goal_token_id)[0]
                instr_locs = torch.where(instr == self.tok.instr_token_id)[0]
                # for with only [CLS]
                # assert len(cls_locs) == 2
                # if len(cls_locs) == 2:
                #     instr_begin = cls_locs[-1].item() # NOTE: goal first or instr first?
                #     ids[:instr_begin] = 1
                # elif len(cls_locs) == 1:
                #     pass
                # else:
                #     raise ValueError
                assert len(goal_locs) == 1
                assert len(instr_locs) == 1
                ids[goal_locs:instr_locs] = 1

                text_type_ids.append(ids)
            text_type_ids = torch.stack(text_type_ids, dim=0)
        else:
            text_type_ids = torch.zeros_like(mask)
        # text_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx), text_type_ids.long().cuda()

    def _get_exploration_input(self, obs):
        seq_tensor = []
        for ob in obs:
            p = np.random.rand()
            if p > 0.7:
                seq_tensor.append(ob['instr_encoding'])
            else:
                seq_tensor.append(ob['goal_encoding'])

        seq_tensor = np.array(seq_tensor)
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != padding_idx)

        token_type_ids = torch.zeros_like(mask)

        # text_type_ids = []
        # for instr in seq_tensor:
        #     ids = torch.zeros_like(instr)
        #     goal_locs = torch.where(instr == self.tok.goal_token_id)[0]
        #     instr_locs = torch.where(instr == self.tok.instr_token_id)[0]
        #     assert len(goal_locs) == 1
        #     assert len(instr_locs) == 1
        #     ids[goal_locs:instr_locs] = 1
        #     text_type_ids.append(ids)
        # text_type_ids = torch.stack(text_type_ids, dim=0)
        text_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx), text_type_ids.long().cuda()

    # def _get_exploration_labels(self, obs):
    #     labels = []
    #     masks = []
    #     max_len = max([len(ob['exploration_labels']) for ob in obs])
    #     for ob in obs:
    #         length = len(ob['exploration_labels'])
    #         labels.append(list(ob['exploration_labels']) + [0] * (max_len-length))
    #         masks.append([1] * length + [0] * (max_len-length))
    #     labels = torch.tensor(labels).long().cuda()
    #     masks = torch.tensor(masks).float().cuda()
    #     return labels, masks

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended, from_env=True, results=None, t=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if from_env:
                teacher_viewpoint = ob['teacher']
            else:
                teacher_viewpoint = results[i][t+1] if t <= len(results[i]) - 2 else results[i][-1]

            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == teacher_viewpoint:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert teacher_viewpoint == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None, exp=False):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name, exp):
            if type(name) is int:       # Go to the next view
                if not exp:
                    self.env.env.sims[idx].makeAction(name, 0, 0)
                else:
                    self.env.env.exp_sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                if not exp:
                    self.env.env.sims[idx].makeAction(*self.env_actions[name])
                else:
                    self.env.env.exp_sims[idx].makeAction(*self.env_actions[name])

        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            if exp:
                env_sims = self.env.env.exp_sims
            else:
                env_sims = self.env.env.sims

            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up', exp=exp)
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down', exp=exp)
                    src_level -= 1
                while env_sims[idx].getState().viewIndex != trg_point:    # Turn right until the target # "ROTATE"!
                    take_action(i, idx, 'right', exp=exp)

                if select_candidate['viewpointId'] != \
                       env_sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId:
                    pdb.set_trace()

                assert select_candidate['viewpointId'] == \
                       env_sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'], exp=exp)

                state = env_sims[idx].getState()
                if traj is not None:
                    if exp:
                        k = len(perm_idx) // len(traj)
                        traj[i//k]['path'].append((state.location.viewpointId, state.heading, state.elevation))
                    else:
                        traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def make_backward_action(self, obs, old_viewpointId, traj=None):
        assert len(obs) == 1 # batch size 1
        a_t = None
        for i, cand in enumerate(obs[0]['candidate']):
            if cand['viewpointId'] == old_viewpointId:
                a_t = [i]
        assert a_t is not None # must find way back        
        # move backward, no need to recover heading & elevation
        self.make_equiv_action(a_t, obs, traj=traj) 

    def make_backward_action_batch(self, perm_obs, perm_idx, old_viewpoints, prev_a_t, traj=None):
        batch_size = len(perm_obs)
        a_t = np.array([-1] * batch_size)

        for i, (ob, a) in enumerate(zip(perm_obs, prev_a_t)):
            if a == -1:
                continue
            for j, cand in enumerate(ob['candidate']):
                if cand['viewpointId'] == old_viewpoints[i]:
                    a_t[i] = j

        self.make_equiv_action(a_t, perm_obs, perm_idx, traj)

    # NOTE: idle
    def getStates(self, perm_idx, exp=False):
        states = []
        if not exp:
            states = [self.env.env.sims[idx].getState() for idx in perm_idx]
        else:
            num_exp = len(self.env.env.exp_sims) // self.env.batch_size
            states = [self.env.env.exp_sims[idx*self.env.batch_size+j].getState() for idx in perm_idx for j in range(num_exp)]
        return states

    # # NOTE: idle
    # def collect_future_feats(self, perm_obs, perm_idx, candidate_leng):
    #     batch_size = len(perm_obs)
    #     ended = np.array([False] * batch_size)
    #     max_leng = max(candidate_leng)
    #     old_viewpoints = [ob['viewpoint'] for ob in perm_obs]
        
    #     # return (B, max_leng, 36, D)
    #     pano_feats = []
    #     angle_feats = []
    #     cand_feats = []
    #     cand_lens = []
    #     for i in range(max_leng):
    #         cpu_a_t = np.array([i] * batch_size, dtype=np.int64) * ~ended + np.array([-1] * batch_size, dtype=np.int64) * ended
            
    #         self.make_equiv_action(cpu_a_t, perm_obs, perm_idx)
    #         explore_obs = np.array(self.env._get_obs())[perm_idx]
    #         angle_feat, cand_feat, cand_len = self.get_input_feat(explore_obs) # we can also return other features
    #         # pano_f_t = self._feature_variable(explore_obs) # (B, 36, D)
    #         # pano_feats.append(pano_f_t)
    #         angle_feats.append(angle_feat)
    #         cand_feats.append(cand_feat)
    #         cand_lens.append(cand_len)

    #         self.make_backward_action_batch(explore_obs, perm_idx, old_viewpoints, cpu_a_t)
            
    #         states = self.getStates(perm_idx)
    #         recovered_viewpoints = [state.location.viewpointId for state in states]
            
    #         assert utils.equal_by_element(old_viewpoints, recovered_viewpoints)
    #         for k in range(batch_size):
    #             perm_obs[k]['viewIndex'] = states[k].viewIndex
            
    #         ended = ended | (np.array(candidate_leng) == (i + 2))

    #     cand_lens_flat = utils.flat_list_of_list_vertical(cand_lens)
    #     max_future_leng = max(cand_lens_flat)

    #     # pano_feats = torch.stack(pano_feats, dim=1) # (B, max_leng, 36, D)
    #     angle_feats = torch.stack(angle_feats, dim=1) # (B, max_leng, D)
    #     future_cand_feats = torch.zeros((batch_size, max_leng, max_future_leng, cand_feat.size(-1)), dtype=torch.float32).cuda()
    #     for i, (cand_feat, lengs) in enumerate(zip(cand_feats, cand_lens)):
    #         future_cand_feats[:, i, :max(lengs), :] = cand_feat # (B, max_leng, future_max_leng, D)
        
    #     candidate_mask = utils.length2mask(candidate_leng).unsqueeze(-1)
    #     future_candidate_mask = utils.length2mask(cand_lens_flat)
    #     candidate_mask = candidate_mask | future_candidate_mask.view(batch_size, max_leng, max_future_leng)
        
    #     return pano_feats, angle_feats, future_cand_feats, candidate_mask


    def train_classifier(self, n_iters, test=False, use_gt=False, **kwargs):
        ''' Train for a given number of iterations '''
        if n_iters == None:
            n_iters = len(self.env) // args.batchSize + 1

        if use_gt:
            assert test

        self.vln_bert.eval()
        self.critic.eval()
        self.exploration.eval()
        self.explore_critic.eval()
        self.classifier.train()

        self.losses = []
        self.classification_results = []
        self.accuracies = []
        self.ids_labels = {}
        looped = False

        for iter in (range(1, n_iters + 1)):
            self.classifier_optimizer.zero_grad()
            self.loss = 0
            
            obs = np.array(self.env.reset())
            seq_tensor = []
            targets = []
            for ob in obs:
                if not test:
                    key = 'instr_encoding' if np.random.rand() > 0.5 else 'goal_encoding'
                    label = 0 if key == 'instr_encoding' else 1
                else:
                    key = 'instr_encoding'
                    if 'level' in ob:
                        label = 0 if ob['level'] < 3 else 1
                    else:
                        label = 0
                seq_tensor.append(ob[key])
                targets.append(label)
            seq_tensor = np.array(seq_tensor)
            targets = np.array(targets)
            seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
            seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

            seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
            targets = torch.from_numpy(targets).long().cuda()

            token_type_ids = torch.zeros_like(seq_tensor).long().cuda()
            
            seq_embeddings = self.vln_bert.vln_bert.get_word_embeddings(seq_tensor, token_type_ids)
            pred = self.classifier(seq_embeddings.permute(0, 2, 1))
            
            self.loss = self.ce(pred, targets)
            self.accuracies.append((pred.argmax(1) == targets).float().mean().item())
            self.losses.append(self.loss.item())

            if not test:
                self.loss.backward()
                torch.nn.utils.clip_grad_norm(self.classifier.parameters(), 40.) # TODO: need?
                self.classifier_optimizer.step()
            else:
                pred_labels = pred.argmax(1).tolist()
                for i, (ob, label) in enumerate(zip(obs, pred_labels)):
                    if ob['instr_id'] in self.ids_labels:
                        looped = True
                    else:
                        self.ids_labels[ob['instr_id']] = targets[i] if use_gt else label
                if looped:
                    assert len(self.ids_labels) == len(self.env)
                    break
        
            print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

        self.ids_labels = [[k, v] for k, v in self.ids_labels.items()]
        self.accuracies = np.mean(self.accuracies)     
        return self.losses


    def rollout_exploration(self, train_ml=None, train_rl=True, reset=True, speaker=None, test=False):
        assert train_rl == False
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)
        
        if not test:
            trim_sentence, trim_lang_att_mask, trim_token_type_ids, seq_lengths, perm_idx, trim_text_type_ids = self._get_exploration_input(obs)
        else:
            trim_sentence, trim_lang_att_mask, trim_token_type_ids, seq_lengths, perm_idx, trim_text_type_ids = self._sort_batch(obs, key='instr_encoding')
        perm_obs = obs[perm_idx]

        full_instr_result = [ob['result'] for ob in perm_obs]

        #####################################
        # Trimmed Forward
        #####################################
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        ''' Language BERT for exploration'''
        language_inputs = {'mode':        'language',
                        'sentence':       trim_sentence,
                        'attention_mask': trim_lang_att_mask,
                        'lang_mask':      trim_lang_att_mask,
                        'token_type_ids': trim_token_type_ids}
        if args.vlnbert == 'oscar':
            lang_feats_e2e = self.vln_bert(**language_inputs)
        elif args.vlnbert == 'prevalent':
            h_t_e2e, lang_feats_e2e = self.vln_bert(**language_inputs) # TODO: how should we deal with the state vector? 

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # rewards = []
        
        # policy_log_probs = []
        # masks = []
        # entropys = []
        
        # h_0, c_0 = None, None
        # NOTE: extra line here
        # hidden_states = []

        ml_loss = 0
        explore_masks = torch.tensor(~ended.reshape(-1, 1), dtype=torch.long, requires_grad=False).cuda()

        # h_t = None
        # c_t = h_t_e2e.clone()
        # h1 = self.exploration.init_h1(lang_feats_e2e[:, 1:, :], trim_lang_att_mask[:, 1:])
        # h1 = h_t_e2e.clone()


        for t in range(self.episode_len):

            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert=='prevalent'):
                lang_feats_e2e = torch.cat((h_t_e2e.unsqueeze(1), lang_feats_e2e[:,1:,:]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()            

            visual_attention_mask = torch.cat((trim_lang_att_mask, visual_temp_mask), dim=-1)

            # decoder
            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           lang_feats_e2e,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          trim_lang_att_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     trim_token_type_ids,
                            'action_feats':       input_a_t,
                            'cand_feats':         candidate_feat.clone(),
                            'return_attn':        True}
            h_t_e2e, e2e_logit, attn_lang, attn_vis, attn_lang_probs, attn_vis_probs, lang_state_scores = self.vln_bert(**visual_inputs)

            uncertain = self.exploration(e2e_logit, lang_state_scores, attn_vis, attn_lang)
            # hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)  
            e2e_logit.masked_fill_(candidate_mask, -float('inf'))
            # target = self._teacher_action(perm_obs, ended) # fake target
            target = self._teacher_action(perm_obs, ended, from_env=False, results=full_instr_result, t=t)

            _, a_t_before_exp = e2e_logit.max(1)

            exp_label = torch.ones(batch_size, dtype=torch.long).cuda()
            exp_label = (exp_label & (a_t_before_exp == target)).long()

            ml_loss += self.criterion(uncertain, exp_label)
            self.accuracies.append((uncertain.max(1)[1] == exp_label).float().mean().item())
            
            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target   # normal action when apply_speaker, else true target #TODO: real target
            else:
                raise ValueError

            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            prev_obs = perm_obs
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx=perm_idx, traj=traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            explore_masks = torch.cat([explore_masks, torch.tensor(~ended.reshape(-1, 1), dtype=torch.long).cuda()], dim=-1)

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())      

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
            pdb.set_trace()
        else:
            self.losses.append(self.loss.item() / self.episode_len)  # This argument is useless.
        # print(num_zeros, num_ones, pred_ones, num_zeros+num_ones)
        return traj

    def make_exploration_batch(self, decisions, perm_obs, perm_idx, candidate_leng,
                            language_features, language_attention_mask, token_type_ids, logit, h_t,
                            traj=None, k=5, s=1):
        batch_size = len(decisions)
        final_a_t = torch.zeros((batch_size), dtype=torch.long, requires_grad=False).cuda()

        visual_mask_root = (utils.length2mask(candidate_leng) == 0).long()

        # # Mask outputs where agent can't move forward
        # # Here the logit is [b, max_candidate]
        candidate_mask = utils.length2mask(candidate_leng)
        logit.masked_fill_(candidate_mask, -float('inf'))

        # softmax_probs = F.softmax(logit, 1)
        softmax_probs = logit
        num_exp = min(k, softmax_probs.size(1))
        probs, a_t = softmax_probs.topk(num_exp, dim=1)        
        probs = probs.detach()
        a_t = a_t.detach()
        cpu_a_t = a_t.cpu().numpy()

        for i, next_ids in enumerate(cpu_a_t):
            for j, next_id in enumerate(next_ids):
                if next_id >= (candidate_leng[i]-1) or next_id == args.ignoreid:    # The last action is <end>
                    cpu_a_t[i][j] = -1             # Change the <end> and ignore action to -1

        explore_mask = ~(decisions.bool()) # (B, num_exp)

        final_a_t = a_t[:, 0].clone().detach()
        
        if decisions.sum().item() == batch_size or cpu_a_t[:, 0].sum().item() == -1 * batch_size:
            return final_a_t
        else:
            ended = np.array([False] * batch_size * num_exp)[..., np.newaxis]
            ended[:, 0] = np.logical_or(ended[:, 0], (cpu_a_t.reshape(-1) == -1))
            
            all_exp_probs = []
            perm_idx_exp = [idx*k+i for idx in perm_idx for i in range(num_exp)]

            # get exploration simulators
            obs_exp = np.array(self.env._get_exp_obs(resync=True))
            perm_obs_exp = obs_exp[perm_idx_exp]

            # make one step exploration
            self.make_equiv_action(cpu_a_t.reshape(-1), perm_obs_exp, perm_idx_exp, exp=True, traj=traj)

            obs_exp = np.array(self.env._get_exp_obs())
            perm_obs_exp = obs_exp[perm_idx_exp]

            # prepare expanded language features
            if (args.vlnbert=='prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
            n_tokens = language_features.size(1)
            language_features = language_features.unsqueeze(1).repeat((1, num_exp, 1, 1)).view(batch_size*num_exp, n_tokens, -1)
            language_attention_mask = language_attention_mask.unsqueeze(1).repeat((1, num_exp, 1)).view(batch_size*num_exp, -1)
            token_type_ids = token_type_ids.unsqueeze(1).repeat((1, num_exp, 1)).view(batch_size*num_exp, -1)

            for t in range(s):
                exp_input_a_t, exp_cand_feat, exp_cand_len_flat = self.get_input_feat(perm_obs_exp)
                max_future_leng = max(exp_cand_len_flat)

                exp_candidate_mask = (visual_mask_root[:, :num_exp] * explore_mask.unsqueeze(-1)).unsqueeze(-1).bool()
                future_candidate_mask = (utils.length2mask(exp_cand_len_flat) == 0).bool()
                exp_candidate_mask = exp_candidate_mask.view(batch_size*num_exp, -1) & future_candidate_mask
                
                if t >= 1 and args.vlnbert == 'prevalent':
                    language_features = torch.cat([h_t.unsqueeze(1), language_features[:, 1:, :]], dim=1)
                
                visual_temp_mask = exp_candidate_mask.long()
                visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)
            
                self.vln_bert.vln_bert.config.directions = max(exp_cand_len_flat)
                ''' Visual BERT '''
                visual_inputs = {'mode':              'visual',
                                'sentence':           language_features,
                                'attention_mask':     visual_attention_mask,
                                'lang_mask':          language_attention_mask,
                                'vis_mask':           visual_temp_mask,
                                'token_type_ids':     token_type_ids,
                                'action_feats':       exp_input_a_t,
                                'cand_feats':         exp_cand_feat}
                h_t, exp_logit = self.vln_bert(**visual_inputs)

                exp_logit.masked_fill_(~exp_candidate_mask, -float('inf'))

                exp_softmax_probs = exp_logit.view(batch_size, num_exp, max_future_leng) # NOTE: view to align with probs
                exp_probs, exp_a_t = exp_softmax_probs.max(-1)        # student forcing - argmax
                
                invalid_cand_mask = (exp_candidate_mask.sum(-1)==0).view(batch_size, num_exp)
                exp_probs = exp_probs.detach()
                exp_probs.masked_fill_(invalid_cand_mask, 0)

                all_exp_probs.append(exp_probs)
                
                exp_a_t = exp_a_t.view(-1).detach()
                cpu_a_t_exp = exp_a_t.cpu().numpy()
                for i, next_id in enumerate(cpu_a_t_exp):
                    if ended[i, -1] or next_id == (exp_cand_len_flat[i]-1) or next_id == args.ignoreid:
                        cpu_a_t_exp[i] = -1

                self.make_equiv_action(cpu_a_t_exp, perm_obs_exp, perm_idx_exp, exp=True, traj=traj)
                obs_exp = np.array(self.env._get_exp_obs())
                perm_obs_exp = obs_exp[perm_idx_exp]

                ended_t = np.logical_or(ended[:, -1], (cpu_a_t_exp == -1))
                ended = np.concatenate([ended, ended_t[..., np.newaxis]], axis=1)

                if ended_t.all():
                    break

            all_exp_probs = torch.stack(all_exp_probs, dim=2)
            ended = ended.reshape(batch_size, num_exp, -1)

            # # collect features by going back and forth many times
            # for i in range(cpu_a_t.shape[1]):

            #     # make one step exploration and collect features
            #     self.make_equiv_action(cpu_a_t[:, i], perm_obs, perm_idx, traj)
            #     explored_obs = np.array(self.env._get_obs())[perm_idx]
            #     angle_feat, cand_feat, cand_len = self.get_input_feat(explored_obs)
            #     exp_input_a_t.append(angle_feat)
            #     exp_cand_feat.append(cand_feat)
            #     exp_cand_len.append(cand_len)

            #     # move backward
            #     # this is to set self.env.env.sim back to the original point
            #     # need to update obs['viewIndex']?
            #     self.make_backward_action_batch(explored_obs, perm_idx, old_viewpoints, cpu_a_t[:, i], traj) 
            #     states = self.getStates(perm_idx)
            #     recovered_viewpoints = [state.location.viewpointId for state in states]

            #     assert utils.equal_by_element(old_viewpoints, recovered_viewpoints)
            #     for k in range(batch_size):
            #         perm_obs[k]['viewIndex'] = states[k].viewIndex
            # exp_cand_len_flat = utils.flat_list_of_list_vertical(exp_cand_len)

            # max_future_leng = max(exp_cand_len_flat)

            # fake_batch_size = num_exp
            # exp_input_a_t = torch.stack(exp_input_a_t, dim=1) # (B, num_exp, D)
            # exp_cand_feat_padded = torch.zeros((batch_size, num_exp, max_future_leng, self.feature_size + args.angle_feat_size), dtype=torch.float32, requires_grad=False).cuda()
            # for i, feat in enumerate(exp_cand_feat):
            #     length = feat.size(1)
            #     exp_cand_feat_padded[:, i, :length, :] = feat
            
            # exp_candidate_mask = (visual_temp_mask[:, :num_exp] * explore_mask.unsqueeze(-1)).unsqueeze(-1).bool()
            # future_candidate_mask = (utils.length2mask(exp_cand_len_flat) == 0).bool()
            # exp_candidate_mask = exp_candidate_mask & future_candidate_mask.view(batch_size, num_exp, max_future_leng)

            # reshape all
            # exp_input_a_t = exp_input_a_t.view(batch_size*num_exp, -1) 
            # exp_cand_feat_padded = exp_cand_feat_padded.view(batch_size*num_exp, max_future_leng, -1)
            # exp_candidate_mask = exp_candidate_mask.view(batch_size*num_exp, max_future_leng)

            # if (args.vlnbert=='prevalent'):
            #     language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
            # n_tokens = language_features.size(1)
            # language_features = language_features.unsqueeze(1).repeat((1, fake_batch_size, 1, 1)).view(batch_size*num_exp, n_tokens, -1)
            # language_attention_mask = language_attention_mask.unsqueeze(1).repeat((1, fake_batch_size, 1)).view(batch_size*num_exp, -1)
            # token_type_ids = token_type_ids.unsqueeze(1).repeat((1, fake_batch_size, 1)).view(batch_size*num_exp, -1)

            # visual_temp_mask = exp_candidate_mask.long()
            # visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)
            
            # self.vln_bert.vln_bert.config.directions = max_future_leng
            # ''' Visual BERT '''
            # visual_inputs = {'mode':              'visual',
            #                 'sentence':           language_features,
            #                 'attention_mask':     visual_attention_mask,
            #                 'lang_mask':          language_attention_mask,
            #                 'vis_mask':           visual_temp_mask,
            #                 'token_type_ids':     token_type_ids,
            #                 'action_feats':       exp_input_a_t,
            #                 # 'pano_feats':         f_t,
            #                 'cand_feats':         exp_cand_feat_padded}
            # _, exp_logit = self.vln_bert(**visual_inputs)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            # exp_logit.masked_fill_(~exp_candidate_mask, -float('inf'))

            # exp_softmax_probs = exp_logit.view(batch_size, num_exp, max_future_leng)
            # exp_probs, _ = exp_softmax_probs.max(-1)        # student forcing - argmax
            
            # valid_cand_mask = ~(exp_candidate_mask.sum(-1)==0).view(batch_size, num_exp)
            # exp_probs = exp_probs.detach()
            # exp_probs.masked_fill_(~valid_cand_mask, 0)
            
            lam = 1.2
            # best_scores = np.zeros(batch_size)
            # for i in range(num_exp):
            #     p1 = probs[:, i].cpu().numpy()
            #     p2 = exp_probs[:, i].cpu().numpy()
            #     is_stop = (cpu_a_t[:, i] == -1)
                
            #     score = (p1 + (p2*~is_stop + p1*is_stop)*lam) * valid_cand_mask[:, i].cpu().numpy()

            #     is_better = torch.from_numpy(score >= best_scores).cuda() * valid_cand_mask[:, i]
            #     best_scores = score*is_better.cpu().numpy() + best_scores*~is_better.cpu().numpy()
            #     final_a_t = a_t[:, i]*is_better + final_a_t*~is_better
            # final_a_t = a_t[:, 0]*(decisions.bool()) + final_a_t*~(decisions.bool())

            for i in range(batch_size):
                if not decisions[i]:
                    p_s = torch.cat([probs[i].unsqueeze(1), all_exp_probs[i]], dim=1) # (num_exp, s+1)

                    stop_idx = torch.tensor((~ended[i]).sum(-1).reshape(-1, 1)).cuda() # (num_exp, )

                    power = torch.arange(0, p_s.size(1)).unsqueeze(0).repeat(p_s.size(0), 1).cuda()
                    
                    stop_mask = power >= stop_idx

                    power = power * ~stop_mask + stop_idx * stop_mask

                    stop_idx = stop_idx * (stop_idx < ended[i].shape[1]) + (stop_idx - 1) * (stop_idx == ended[i].shape[1])
                    p_s = p_s * ~stop_mask + p_s.gather(1, stop_idx).repeat(1, p_s.size(1)) * stop_mask
                    
                    # is_stop = torch.from_numpy((cpu_a_t[i, :] == -1)).cuda()
                    
                    scores = (lam**power * p_s).sum(-1).masked_fill_(~(visual_mask_root[i, :num_exp].bool()), -float('inf')) # (num_exp, )
                    
                    # scores = (p1 + (lam*p2*~is_stop + p1*is_stop)).masked_fill_(~valid_cand_mask[i, :], -float('inf'))
                    _, best_a_t_idx = scores.max(0)
                    final_a_t[i] = a_t[i, best_a_t_idx]
            
            return final_a_t

    def speaker_score(self, decisions, perm_obs, perm_idx, candidate_leng, img_features, cand_feats, lengths, logit, speaker, instrs, k=5):
        batch_size = len(decisions)
        final_a_t = torch.zeros((batch_size), dtype=torch.long, requires_grad=False).cuda()

        candidate_mask = utils.length2mask(candidate_leng)
        logit.masked_fill_(candidate_mask, -float('inf'))

        
        softmax_probs = F.softmax(logit, 1)
        num_exp = min(k, softmax_probs.size(1))
        probs, a_t = softmax_probs.topk(num_exp, dim=1)        
        probs = probs.detach()
        a_t = a_t.detach()
        cpu_a_t = a_t.cpu().numpy()

        log_probs = F.log_softmax(logit, 1)
        log_probs, _ = log_probs.topk(num_exp, dim=1)        

        for i, next_ids in enumerate(cpu_a_t):
            for j, next_id in enumerate(next_ids):
                if next_id >= (candidate_leng[i]-1) or next_id == args.ignoreid:    # The last action is <end>
                    cpu_a_t[i][j] = -1             # Change the <end> and ignore action to -1

        explore_mask = ~(decisions.bool()) # (B, num_exp)

        final_a_t = a_t[:, 0].clone().detach()

        if decisions.sum().item() == batch_size or cpu_a_t[:, 0].sum().item() == -1 * batch_size:
             # no need to explore
            return final_a_t
        else:
            img_features = torch.stack(img_features, dim=1) # (B, n_prev, 36, 2052)
            img_features = torch.repeat_interleave(img_features, num_exp, dim=0)
            lengths = np.repeat(lengths, num_exp)
            instrs = [instr for instr in instrs for _ in range(num_exp)]
            
            assert len(cand_feats) > 0
            cand_feats = torch.stack(cand_feats, dim=1) # (B, n_prev, 2052)
            cand_feats = torch.repeat_interleave(cand_feats, num_exp, dim=0) # (B*k, n_prev, 2052)
            next_cand_feat = np.zeros((batch_size*num_exp, self.feature_size+args.angle_feat_size), dtype=np.float32)            
            for i, cpu_a_t_row in enumerate(cpu_a_t):
                n_candidate = len(perm_obs[i]['candidate'])
                for j, a in enumerate(cpu_a_t_row):
                    if a == -1:
                        pass
                    else:
                        next_cand_feat[i*num_exp+j, :] = perm_obs[i]['candidate'][a]['feature']
            next_cand_feat = torch.from_numpy(next_cand_feat).cuda().unsqueeze(1)
            cand_feats = torch.cat([cand_feats, next_cand_feat], dim=1)
            scores = speaker.score_candidates(img_features, cand_feats, lengths, instrs)
            scores = scores.view(batch_size, num_exp)
            
            speaker_std = np.std(scores.cpu().detach().numpy())
            follower_mask = np.isinf(log_probs.cpu().detach().numpy())
            follower_std = np.ma.masked_array(log_probs.cpu().detach().numpy(), mask=follower_mask).std()
            lam = 0.2
            speaker_weight = lam / speaker_std
            follower_weight = (1-lam) / follower_std

            best_scores = np.zeros(batch_size)        
            _, final_a_t = (log_probs*follower_weight + scores*speaker_weight).max(1)
            final_a_t = a_t[:, 0]*(decisions.bool()) + final_a_t*~(decisions.bool())
            return final_a_t
      
    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Language input
        # if args.mix_text_input:
        #     sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx, text_type_ids = self._sort_batch(obs, key='goal_instr_encoding')
        # else:
            # only input full instructions
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx, text_type_ids = self._sort_batch(obs, key='instr_encoding')
        perm_obs = obs[perm_idx]

        label = None
        if hasattr(self.env, "label"):
            label = self.env.label

        ''' Language BERT '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,
                        'attention_mask': language_attention_mask,
                        'lang_mask':      language_attention_mask,
                        'token_type_ids': token_type_ids, 
                        'label':          label}
        if args.vlnbert == 'oscar':
            language_features = self.vln_bert(**language_inputs)
        elif args.vlnbert == 'prevalent':
            h_t, language_features = self.vln_bert(**language_inputs)
        h_t_last = h_t

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in perm_obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        f_t_history = []
        cand_feat_history = []
        lengths = np.zeros(batch_size, dtype=np.int64)
        accumulate_exp = torch.zeros(batch_size).cuda()

        
        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert=='prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            
            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          language_attention_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            # 'pano_feats':         f_t,
                            'cand_feats':         candidate_feat,
                            'return_attn':        True}
            h_t, logit, attn_lang, attn_vis, attn_lang_probs, attn_vis_probs, lang_state_scores = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            if args.e2e and self.feedback == 'argmax':
                lengths += (1-ended)
                #####################################
                # EXPLORATION!!!
                #####################################
                uncertain = self.exploration(logit, lang_state_scores, attn_vis, attn_lang)
                _, decisions = F.softmax(uncertain).max(1)
                # decisions = (F.softmax(uncertain)[:, 1] > 1).type(torch.int64)
                decisions = torch.ones_like(decisions) if t==0 else decisions
                
                accumulate_exp += (1-decisions)
                decisions = ((accumulate_exp > 3) | decisions.bool()).float() # only explore 3 times

                if args.state_freeze:
                    h_t_last = h_t*decisions.view(-1,1).bool() + h_t_last*~(decisions.view(-1,1).bool())
                else:
                    h_t_last = h_t
                

                exp_traj = [{
                    'instr_id': ob['instr_id'],
                    'path': [],
                } for ob in perm_obs]

                if speaker is None:
                    a_t_after_exp = self.make_exploration_batch(decisions, perm_obs, perm_idx, candidate_leng,
                                                                        language_features, language_attention_mask, token_type_ids, 
                                                                        logit, h_t_last, traj=exp_traj, k=args.k, s=args.s)
                else:
                    f_t = self._feature_variable(perm_obs)
                    f_t_history.append(f_t)
                    instrs = [ob['instructions'] for ob in perm_obs]
                    a_t_after_exp = self.speaker_score(decisions, perm_obs, perm_idx, candidate_leng, f_t_history, cand_feat_history, lengths, logit, speaker, instrs)
            
            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)
            
            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                if args.e2e:
                    a_t = a_t_after_exp
                    cand_feats = np.zeros((batch_size, self.feature_size+args.angle_feat_size), np.float32)
                    for i, a in enumerate(a_t):
                        if a < len(perm_obs[i]['candidate']) and not ended[i]:
                            cand_feats[i, :] = perm_obs[i]['candidate'][a]['feature']
                    cand_feat_history.append(torch.from_numpy(cand_feats).cuda())

                    _, a_t_before_exp = logit.max(1)

                    necessary_exp = ((a_t_before_exp != target) | (a_t_before_exp != a_t_after_exp))

                    for i, (nss, is_exp, e) in enumerate(zip(necessary_exp, ~decisions.bool(), ended)):
                        current_vp = perm_obs[i]['viewpoint']
                        hist_vps = [vp_tuple[0] for vp_tuple in traj[i]['path']]

                        if is_exp and not e:
                            if a_t[i] < len(perm_obs[i]['candidate']):
                                next_vp = perm_obs[i]['candidate'][a_t[[i]]]['viewpointId']
                            else:
                                continue
                                next_vp = ''
                            exp_path = exp_traj[i]['path']
                            exp_path = [vp_tuple for vp_tuple in exp_path if vp_tuple[0] != next_vp and vp_tuple[0] != current_vp and vp_tuple[0] not in hist_vps]

                            if a_t[i] >= len(perm_obs[i]['candidate']):
                                exp_path.append(traj[i]['path'][-1])
                            traj[i]['path'] += exp_path
                    # h_t_last = h_t
                    # h_t_last = h_t * (a_t_before_exp == a_t_after_exp).view(-1, 1) + h_t_last * (a_t_before_exp != a_t_after_exp).view(-1, 1)

                    # necessary_exp = ~(decisions.bool()) & ((a_t_before_exp != target) | (a_t_before_exp != a_t_after_exp))

                else:
                    _, a_t = logit.max(1)        # student forcing - argmax

                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]            # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] < 3.0:                             # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          language_attention_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            # 'pano_feats':         f_t,
                            'cand_feats':         candidate_feat}
            last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, speaker=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.exploration.train()
            self.critic.train()
            self.explore_critic.train()
        else:
            self.vln_bert.eval()
            self.exploration.eval()
            self.critic.eval()
            self.explore_critic.eval()
        if iters == None:
            iters = len(self.env) // self.env.batch_size + 1
        super(Seq2SeqAgent, self).test(iters, speaker=speaker)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.exploration.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.exploration_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def train_explorer(self, n_iters, feedback='teacher', test=False, **kwargs):
        ''' Train for a given number of iterations '''

        self.feedback = feedback
        if n_iters == None:
            n_iters = len(self.env) // args.batchSize + 2
        looped = False

        self.vln_bert.eval()
        self.critic.eval()

        if not test:
            self.exploration.train()
            self.explore_critic.train()
        else:
            self.exploration.eval()
            self.explore_critic.eval()

        self.losses = []
        self.results = []
        self.accuracies = []

        for iter in (range(1, n_iters + 1)):
            self.exploration_optimizer.zero_grad()
            self.explore_critic_optimizer.zero_grad()
            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                traj = self.rollout_exploration(train_ml=1.0, train_rl=False, test=test, **kwargs)
            else:
                assert False

            if not test:
                self.loss.backward()
                torch.nn.utils.clip_grad_norm(self.exploration.parameters(), 40.) # TODO: need?
                self.exploration_optimizer.step()
                self.explore_critic_optimizer.step()
            print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

        self.accuracies = np.mean(self.accuracies)     
        return self.losses

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
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer),
                     ("exploration", self.exploration, self.exploration_optimizer),
                     ("explore_critic", self.explore_critic, self.explore_critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def save_explorer(self, epoch, path):
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("exploration", self.exploration, self.exploration_optimizer),
                     ("explore_critic", self.explore_critic, self.explore_critic_optimizer),]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def save_classifier(self, epoch, path):
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("classifier", self.classifier, self.classifier_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer),] # TODO: load exploration elsewhere
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1

    def load_explorer(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("exploration", self.exploration, self.exploration_optimizer)] # TODO: load exploration elsewhere
        for param in all_tuple:
            recover_state(*param)
        return states['exploration']['epoch'] - 1
    
    def load_classifier(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("classifier", self.classifier, self.classifier_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['classifier']['epoch'] - 1
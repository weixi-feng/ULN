''' Batched Room-to-Room navigation environment '''

from lib2to3.pgen2 import token
import sys
# sys.path.append('buildpy36')
sys.path.append('/data2/weixifeng/Matterport3DSimulator_old/build/')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args
from nltk import pos_tag, word_tokenize
from tqdm import tqdm

from utils import load_datasets, load_nav_graphs, pad_instr_tokens
import pdb

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('    Image features not provided - in testing mode')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

        self.batch_size = batch_size
        self.exp_sims = []
        self.exp_size = args.k

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

    def setExpSims(self):
        for i in range(self.batch_size): 
            state = self.sims[i].getState()
            for j in range(self.exp_size):
                sim = MatterSim.Simulator()
                sim.setRenderingEnabled(False)
                sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
                sim.setCameraResolution(self.image_w, self.image_h)
                sim.setCameraVFOV(math.radians(self.vfov))
                sim.init()
                sim.newEpisode(state.scanId, state.location.viewpointId, state.heading, state.elevation)
                self.exp_sims.append(sim)

    def syncEpisodes(self):
        if len(self.exp_sims) == 0:
            self.setExpSims()
        
        for i in range(self.batch_size):
            state = self.sims[i].getState()
            for j in range(self.exp_size):
                self.exp_sims[i*self.exp_size+j].newEpisode(state.scanId, state.location.viewpointId, state.heading, state.elevation)

    def getExpStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.exp_sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    # direction_tokens = ['left', 'right', 'back', 'forward', 'straight', 'around']
    # mistagged_verbs = ['turn', 'walk', 'wait', 'exit', 'step', 'front', 'behind', 'see', 'go', 'stop', 'next', 'middle']
    # object_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    # VPs = ['go to', 'walk to', 'stop in', 'wait in']

    def __init__(self, feature_store, tokenizer, stok, batch_size=100, seed=10, splits=['train'],
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        else:
            self.feature_size = 2048
        self.data = []

        # NOTE: always pass tokenizer
        self.tok = tokenizer
        self.stok = stok
        
        def tokenize_instr(instr):
            instr_tokens = tokenizer.tokenize(instr)
            padded_instr_tokens, num_words = pad_instr_tokens(instr_tokens, args.maxInput)
            return tokenizer.convert_tokens_to_ids(padded_instr_tokens)
        
        scans = []
        for split in splits:
            tokenized = "/" in split
            for i_item, item in enumerate(tqdm(load_datasets([split]))):
                if args.test_only and i_item == 64:
                    break
                
                if 'instr_encoding' in item:
                    new_item = dict(item)
                    new_item['instr_id'] = item['path_id']
                    new_item['instructions'] = item['instructions'][0]
                    new_item['instr_encoding'] = item['instr_enc']
                    if new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
                    continue

                for j, instr in enumerate(item['instructions']):
                    new_item = dict(item)
                    new_item['instr_id'] = item['path_id'] if "/" in split else '%s_%d' % (item['path_id'], j) 
                    new_item['instructions'] = instr.replace(" ##", "") if tokenized else instr

                    ''' BERT tokenizer '''
                    # new_item['instr_encoding'] = tokenizer.encode_sentence(new_item['instructions'], tokenized=False) # replace bare BertTokenizer with wrapped version
                    new_item['instr_encoding'] = tokenize_instr(new_item['instructions'])


                    if 'val' in split or 'test' in split:
                        if 'sample' not in split:
                            sents = new_item['instructions'].strip().rstrip(".").split(".")
                            new_item['goals'] = sents[-1] + "." # use last sentence as goal
                            if len(new_item['goals'].split()) <= 3 and len(sents) > 1:
                                new_item['goals'] = sents[-2] + sents[-1] + "."
                        else:
                            if item['levels'][j] != 1:
                                sents = new_item['instructions'].strip().rstrip(".").split(".")
                                new_item['goals'] = sents[-1] + "." # use last sentence as goal
                                if len(new_item['goals'].split()) <= 3 and len(sents) > 1:
                                    new_item['goals'] = sents[-2] + sents[-1] + "."
                            else:
                                new_item['goals'] = item['goals'][j]
                    else:
                        new_item['goals'] = item['goals'][j]

                    # new_item['goal_encoding'] = tokenizer.encode_sentence(new_item['goals'], tokenized=False) # replace bare BertTokenizer with wrapped version
                    new_item['goal_encoding'] = tokenize_instr(new_item['goals'])

                    if 'levels' in item:
                        new_item['level'] = item['levels'][j]

                    # for speaker
                    # new_item['speaker_instr_encoding'] = self.stok.encode_sentence(new_item['instructions'])

                    if 'result' in item:
                        try:
                            new_item['result'] = [v[0] for v in item['result'][j]]
                        except:
                            new_item['result'] = [v[0] for v in item['result'][0]]

                    if new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))


    def __len__(self):
        return len(self.data)

    def size(self):
        return len(self.data)

    def reload_data(self, ids):
        new_data = []
        for d in self.data:
            if d['instr_id'] in ids:
                new_data.append(d)
        assert len(new_data) == len(ids)
        self.data = new_data
        self.reset_epoch(shuffle=False)

    def set_label(self, label):
        self.label = label

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = np.zeros((36, 2048))

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'gt_path' : item['path'],
                'path_id' : item['path_id'],
                'instr_encoding' : item['instr_encoding'],
                'goals': item['goals'],
                'goal_encoding': item['goal_encoding'],
            })
            # 'speaker_instr_encoding' : item['speaker_instr_encoding'],
            if 'result' in item:
                obs[-1]['result'] = item['result']

            if 'level' in item:
                obs[-1]['level'] = item['level']

            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def _get_exp_obs(self, resync=False):
        if resync:
            self.env.syncEpisodes()

        obs = []
        for i, (feature, state) in enumerate(self.env.getExpStates()):
            item = self.batch[i//self.env.exp_size]
            base_view_id = state.viewIndex

            if feature is None:
                feature = np.zeros((36, 2048))

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'gt_path' : item['path'],
                'path_id' : item['path_id'],
            })
            
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats

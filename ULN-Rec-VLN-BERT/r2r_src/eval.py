''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint

from torch._C import import_ir_module_from_buffer
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs, ndtw_graphload, DTW
from agent import BaseAgent


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.underspec = False
        self.levels = []
        for split in splits:
            data = load_datasets([split])
            if 'levels' in next(iter(data)):
                self.underspec = True
                all_levels = [l for d in data for l in d['levels']]
                self.min_level = min(all_levels)
                self.max_level = max(all_levels)
            for item in data:
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['path_id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        with open("/mnt/sshd/weixifeng/r2r_data/scan_viewpoint_info.json", 'r') as file:
            self.viewpoint_label = json.load(file)
        

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        gt = self.gt[instr_id.split('_')[-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]  # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0  # length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )
        if self.underspec:
            self.scores['levels'].append(gt['levels'][int(instr_id.split("_")[-1])])
            final_label = self.viewpoint_label[gt['scan']][final_position]['label']
            goal_label = self.viewpoint_label[gt['scan']][goal]['label']
            success = (self.distances[gt['scan']][final_position][goal] < self.error_margin)
            if goal_label == 'no label':
                self.scores['region_sr'].append(success)
            else:
                self.scores['region_sr'].append(success or (goal_label == final_label and self.distances[gt['scan']][final_position][goal] < 6.0) )

    def get_navigation_error(self, output_file):
        self.ne = []
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file

        print('result length', len(results))
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                instr_id = item['instr_id']
                path = item['trajectory']
                
                gt = self.gt[instr_id.split('_')[-2]]
                start = gt['path'][0]
                assert start == path[0][0], 'Result trajectories should include the start position'
                goal = gt['path'][-1]
                final_position = path[-1][0]  # the first of [view_id, angle, vofv]
                self.ne.append({'instr_id': instr_id, 'ne': self.distances[gt['scan']][final_position][goal]})
        return self.ne

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file

        print('result length', len(results))
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item['trajectory'])

        if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                           % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
            assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths'])
        }
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        # oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        # score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))

        spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
            for error, p, l in
            zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)
        # score_summary['region_sr'] = np.average(self.scores['region_sr'])

        if self.underspec:
            summary_by_levels = defaultdict(dict)
            for i in range(self.min_level, self.max_level+1):
                key = f"level{i}"
                mask = np.array(self.scores['levels']) == i
                summary_by_levels[key] = {
                    'nav_error': np.ma.array(self.scores['nav_errors'], mask=~mask).mean(),
                    'steps': np.ma.array(self.scores['trajectory_steps'], mask=~mask).mean(),
                    'lengths': np.ma.array(self.scores['trajectory_lengths'], mask=~mask).mean()
                }
                num_successes = len([e for i, e in enumerate(self.scores['nav_errors']) if e < self.error_margin and mask[i]])
                summary_by_levels[key]['success_rate'] = float(num_successes) / float(np.sum(mask))
                # oracle_successes = len([e for i, e in enumerate(self.scores['oracle_errors']) if e < self.error_margin and mask[i]])
                # summary_by_levels[key]['oracle_rate'] = float(oracle_successes) / float(np.sum(mask))

                summary_by_levels[key]['spl'] = np.ma.array(spl, mask=~mask).mean()
                # summary_by_levels[key]['region_sr'] = np.ma.array(self.scores['region_sr'], mask=~mask).mean()

            score_summary['levels'] = summary_by_levels

        return score_summary, self.scores

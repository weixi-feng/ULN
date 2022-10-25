from numpy.lib.arraysetops import isin
import torch

import os
import time
import json
import random
import numpy as np
from collections import defaultdict

from torch.nn.modules import loss

from utils import BTokenizer, read_vocab, write_vocab, build_vocab, padding_idx, timeSince, read_img_features, print_progress
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer
from utils import Tokenizer
from speaker import Speaker
import copy
import pdb

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'data/prevalent_vocab/train_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

VALID_EVERY = 2

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES

feedback_method = args.feedback  # teacher or sample

print(args); print('')


''' train the listener '''
def train(train_env, tok, n_iters, log_every=2000, val_envs={}, aug_env=None, stok=None):
    valid_counter = 1

    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    record_file = open('./logs/' + args.name + '.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    start_iter = 0
    if args.load is not None:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))

    speaker = None
    if args.speaker:
        speaker = Speaker(train_env, listner, tok=stok)
        if args.speaker_snap is not None:
            speaker.load(args.speaker_snap)
    # no need to cuda here

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", 'update':False}, }
    env_names = list(val_envs.keys())
    for env_name in env_names:
        if env_name != 'val_seen' and not env_name in best_val:
            best_val[env_name] = {"spl": 0., "sr": 0., "state":"", 'update':False}
            
    counter = 0
    # additional_val_every = 10000 // log_every
    additional_val_every = 1

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=feedback_method, speaker=speaker)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method, speaker=speaker)

                # Train with Augmented data
                listner.env = aug_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method, speaker=speaker)

                print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        # print("total_actions", total, ", max_length", length)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            env_name_components = env_name.split("_")
            suffix = '' if len(env_name_components) == 2 else env_name_components[-1]
            if suffix:
                if counter < additional_val_every - 1:
                    counter += 1
                    continue
                else:
                    counter = 0

            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if isinstance(val, dict):
                    continue # TODO
                if metric in ['spl']:
                    writer.add_scalar("spl/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['spl']: #TODO
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.4f' % (metric, val)

        record_file = open('./logs/' + args.name + '.txt', 'a')
        record_file.write(loss_str + '\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
            else:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                            iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open('./logs/' + args.name + '.txt', 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}, stok=None):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    if args.e2e and args.load_explorer is not None:
        print("Loaded the explorer at iter %d from %s" % (agent.load_explorer(args.load_explorer), args.load_explorer))

    if args.classify_first and args.load_classifier is not None:
        print("Loaded the classifier at iter %d from %s" % (agent.load_classifier(args.load_classifier), args.load_classifier))

    speaker = None
    if args.speaker:
        speaker = Speaker(train_env, agent, tok=stok)
        if args.speaker_snap is not None:
            speaker.load(args.speaker_snap)

    if args.classify_first:
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            _ = agent.train_classifier(None, test=True)
            ids_labels = agent.ids_labels
            sub_vals = []
            # reconstruct envs and evals
            all_labels = list(set([x for _, x in ids_labels]))
            print(all_labels)
            for label in all_labels:
                instr_ids_subset = [id for id, y in ids_labels if y == label]
                sub_env = copy.copy(env)
                sub_env.reload_data(instr_ids_subset)
                sub_env.set_label(label)
                sub_vals.append(sub_env)
            val_envs[env_name] = (sub_vals, evaluator)
        print("\nInstruction classification done!")
    
    for env_name, (env, evaluator) in val_envs.items():
        t1 = time.time()
        agent.logs = defaultdict(list)
        iters = None
        if isinstance(env, list):
            result = {}
            #  [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
            for sub_env in env:
                agent.env = sub_env
                agent.test(use_dropout=False, feedback='argmax', iters=iters, speaker=speaker)
                sub_result = agent.results
                new_keys = set(sub_result.keys())
                existing_keys = set(result.keys())
                assert existing_keys.isdisjoint(new_keys)
                result.update(sub_result)
            result = [{'instr_id': k, 'trajectory': v} for k, v in result.items()]
        else:
            agent.env = env
            agent.test(use_dropout=False, feedback='argmax', iters=iters, speaker=speaker)
            result = agent.get_results()
        
        t2 = time.time()
        print(f"{env_name}: {t2-t1}s")

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                if isinstance(val, dict):
                    continue
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

            if "levels" in score_summary.keys():
                loss_str = ""
                for key, value in score_summary['levels'].items():
                    loss_str += f"\t {key}: "
                    for metric, val in value.items():
                        loss_str += "%s: %.4f, " % (metric, val)
                    loss_str += "\n"
                print(loss_str)
        
        ne = evaluator.get_navigation_error(result)
        result_dict = {d['instr_id']: d['trajectory'] for d in result}
        with open(f'data/R2R_{env_name}.json', 'r') as file:
            data = json.load(file)
        for d in data:
            d.pop('result', None)
        data = {d['path_id']: d for d in data}
        for item in ne:
            instr_id = item['instr_id']
            success = item['ne'] < 3.0
            path_id = int(instr_id.split('_')[0])
            idx = int(instr_id.split('_')[1])
            traj = result_dict[instr_id]
            if 'result' in data[path_id].keys():
                data[path_id]['result'][idx] = traj
                data[path_id]['success'][idx] = success
            else:
                data[path_id]['result'] = [None] * len(data[path_id]['instructions'])
                data[path_id]['result'][idx] = traj
                data[path_id]['success'] = [0] * len(data[path_id]['instructions'])
                data[path_id]['success'][idx] = success

        # json.dump(
        #     list(data.values()),
        #     open(os.path.join('data', 'universal_speaker', f'R2R_{env_name}.json'), 'w'),
        #     sort_keys=True, indent=4, separators=(',', ': ')
        # )
        if 'ne' in args.save:
            json.dump(
                ne,
                open(os.path.join(log_dir, 'submit_%s_ne.json' % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ':')
            )

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)

def train_val(test_only=False):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    # tok = BTokenizer(args)
    tok = get_tokenizer(args)

    speaker_vocab = read_vocab(TRAIN_VOCAB)
    stok = Tokenizer(vocab=speaker_vocab, encoding_length=args.maxInput)
    
    feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = [f'val_unseen{args.suffix}', f'val_seen{args.suffix}']
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok, stok=stok)
    from collections import OrderedDict

    # if args.submit:
    #     val_env_names.append('test')
    # else:
    #     pass

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok, stok=stok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs, stok=stok)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs, stok=stok)
    else:
        assert False

def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    # tok_bert = BTokenizer(args)
    tok_bert = get_tokenizer(args)

    # Load the speaker vocab for SentDrop
    speaker_vocab = read_vocab(TRAIN_VOCAB)
    stok = Tokenizer(vocab=speaker_vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_seen', 'val_unseen', 'val_unseen_underspec']

    # Load the augmentation data
    aug_path = args.aug
    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok_bert, stok=stok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok_bert, name='aug', stok=stok)

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok_bert, stok=stok),
                Evaluation([split], featurized_scans, tok_bert))
                for split in val_env_names}

    # Start training
    train(train_env, tok_bert, args.iters, val_envs=val_envs, aug_env=aug_env, stok=stok)


def train_val_exploration(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    # tok_bert = BTokenizer(args)
    tok_bert = get_tokenizer(args)

    # Load the speaker vocab for SentDrop
    speaker_vocab = read_vocab(TRAIN_VOCAB)
    stok = Tokenizer(vocab=speaker_vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        # val_env_names = ['val_train_seen']
        val_env_names = ['val_seen_uln']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_unseen_uln']

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train_fg'], tokenizer=tok_bert, stok=stok)
    aug_env = None

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok_bert, stok=stok),
                Evaluation([split], featurized_scans, tok_bert))
                for split in val_env_names}

    # Start training
    train_explorer(train_env, tok_bert, args.iters, val_envs=val_envs, aug_env=aug_env, stok=stok)


def train_explorer(train_env, tok, n_iters, log_every=200, val_envs={}, aug_env=None, stok=None):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    save_prefix = os.path.join("snap", f"{args.name}", "state_dict")
    record_filename = './logs/' + f"{args.name}" + '.txt'
    record_file = open(record_filename, 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    start_iter = 0
    if args.load is not None and not args.test_only:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration {}".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration {}".format(args.load, load_iter))

    speaker = None
    if args.speaker:
        speaker = Speaker(train_env, listner, tok=stok)
        if args.speaker_snap is not None:
            speaker.load(args.speaker_snap)
    # no need to cuda here

    start = time.time()
    print('\n Explorer training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_seen_uln': {"spl": 0., "sr": 0., "acc": 0., "state":"", 'update':False}, 
                'val_unseen_uln': {"spl": 0., "sr": 0., "acc": 0., "state":"", 'update':False}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = log_every
        iter = idx + interval
        
        assert aug_env is None
        listner.env = train_env
        
        losses = listner.train_explorer(interval, feedback=feedback_method, speaker=speaker)  # Train interval iters
        train_loss = np.mean(losses)

        train_acc = listner.accuracies
        print("iter {}, train loss: {}, train acc: {}".format(idx+interval, train_loss, train_acc))
        

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, _) in val_envs.items():
            listner.env = env
            losses = listner.train_explorer(None, feedback='teacher', test=True)  # Train interval iters
            accuracy = listner.accuracies
            loss_str += ', %s accuracy: %.4f' %(env_name, accuracy)
            if env_name in best_val:
                if accuracy > best_val[env_name]['acc']:
                    best_val[env_name]['acc'] = accuracy
                    best_val[env_name]['update'] = True
        # for env_name, (env, evaluator) in val_envs.items():
        #     listner.env = env
        #     listner.test(use_dropout=False, feedback='argmax', iters=None)
        #     # Get validation distance from goal under test evaluation conditions
        #     # num_correct, num_steps, losses = listner.train_explorer(None, feedback='argmax', speaker=speaker, test=True)
        #     result = listner.get_results()
        #     score_summary, _ = evaluator.score(result)
        #     loss_str += ", %s " % env_name

        #     for metric, val in score_summary.items():
        #         if isinstance(val, dict):
        #             continue # TODO
        #         if metric in ['spl']:
        #             writer.add_scalar("spl/%s" % env_name, val, idx)
        #             if env_name in best_val:
        #                 if val > best_val[env_name]['spl']: #TODO
        #                     best_val[env_name]['spl'] = val
        #                     best_val[env_name]['update'] = True
        #                 elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
        #                     best_val[env_name]['spl'] = val
        #                     best_val[env_name]['update'] = True
        #         loss_str += ', %s: %.4f' % (metric, val)

        # record_file = open(record_filename, 'a')
        # record_file.write(loss_str + '\n')
        # record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save_explorer(idx, os.path.join(save_prefix, "best_%s" % (env_name)))
            else:
                listner.save_explorer(idx, os.path.join(save_prefix, "latest_dict"))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                            iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open(record_filename, 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

    listner.save_explorer(idx, os.path.join(save_prefix, "LAST_iter%d" % (idx)))


def train_val_classifier(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    # tok_bert = BTokenizer(args)
    tok_bert = get_tokenizer(args)

    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        # val_env_names = ['val_train_seen']
        val_env_names = ['val_seen_underspec']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_unseen_underspec']

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok_bert, stok=None)
    aug_env = None

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok_bert, stok=None),
                Evaluation([split], featurized_scans, tok_bert))
                for split in val_env_names}

    # Start training
    train_classifier(train_env, tok_bert, args.iters, val_envs=val_envs, aug_env=aug_env)


def train_classifier(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None, stok=None):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    save_prefix = os.path.join("snap", f"{args.name}", "state_dict")
    record_filename = './logs/' + f"{args.name}" + '.txt'
    record_file = open(record_filename, 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    start_iter = 0
    if args.load is not None and not args.test_only:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration {}".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration {}".format(args.load, load_iter))

    start = time.time()
    print('\n Classifier training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen_underspec': {"acc": 0., "state":"", 'update':False}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = log_every
        iter = idx + interval
        
        assert aug_env is None
        listner.env = train_env
        
        losses = listner.train_classifier(interval)  # Train interval iters
        train_loss = np.mean(losses)

        train_acc = listner.accuracies
        print("iter {}, train loss: {}, train acc: {}".format(idx+interval, train_loss, train_acc))
        

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, _) in val_envs.items():
            listner.env = env
            losses = listner.train_classifier(None, test=True)  # Train interval iters
            accuracy = listner.accuracies
            loss_str += ', %s accuracy: %.4f' %(env_name, accuracy)
            if env_name in best_val:
                if accuracy > best_val[env_name]['acc']:
                    best_val[env_name]['acc'] = accuracy
                    best_val[env_name]['update'] = True

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save_classifier(idx, os.path.join(save_prefix, "best_%s" % (env_name)))
            else:
                listner.save_classifier(idx, os.path.join(save_prefix, "latest_dict"))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                            iter, float(iter)/n_iters*100, loss_str)))

        if iter % log_every == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open(record_filename, 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

    listner.save_classifier(idx, os.path.join(save_prefix, "LAST_iter%d" % (idx)))



if __name__ == "__main__":
    if args.train in ['listener', 'validlistener']:
        train_val(test_only=args.test_only)
    elif args.train == 'auglistener':
        train_val_augment(test_only=args.test_only)
    elif 'explorer' in args.train:
        train_val_exploration(test_only=args.test_only)
    elif 'classifier' in args.train:
        train_val_classifier(test_only=args.test_only)
    else:
        assert False

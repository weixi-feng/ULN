import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--test_only', type=int, default=0, help='fast mode for testing')

        self.parser.add_argument('--iters', type=int, default=300000, help='training iterations')
        self.parser.add_argument('--name', type=str, default='default', help='experiment id')
        self.parser.add_argument('--vlnbert', type=str, default='oscar', help='oscar or prevalent')
        self.parser.add_argument('--train', type=str, default='listener')
        self.parser.add_argument('--description', type=str, default='no description\n')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')

        # Augmented Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.20)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--features", type=str, default='places365')

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        # for speaker
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument("--bidir", type=bool, default=True) 
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")

        self.parser.add_argument("--dim_feedforward", type=int, default=512)
        self.parser.add_argument("--num_layers", type=int, default=2)
        self.parser.add_argument("--layer_norm_eps", type=float, default=0.1)
        self.parser.add_argument("--nhead", type=int, default=4)
        self.parser.add_argument("--speaker_type", type=str, choices=['lstm', 'transformer'], default='transformer')
        self.parser.add_argument("--quick_test", action='store_true')
        self.parser.add_argument("--encoder_drop", type=float, default=0.6)
        self.parser.add_argument("--speaker_drop", type=float, default=0.6)
        self.parser.add_argument("--goal_drop", type=float, default=0.6)
        self.parser.add_argument("--lazy_drop", type=float, default=0.1)
        self.parser.add_argument("--tasks", default='all', choices=['all', 'speaker'])
        self.parser.add_argument("--loss_weight", type=float, default=1.0)

        # for agent
        self.parser.add_argument("--suffix", default="", type=str, help="suffix of the name of dataset")
        self.parser.add_argument("--speaker", action='store_true', help="if use speaker to edit instructions")
        self.parser.add_argument("--speaker_snap", type=str, default=None, help="speaker snapshot")
        self.parser.add_argument("--e2e", action='store_true')
        self.parser.add_argument("--load_explorer", default=None, help='path of the trained model')
        self.parser.add_argument("--aemb", type=int, default=64)
        self.parser.add_argument("--save", type=str, nargs='+', default='')
        self.parser.add_argument("--k", type=int, default=3)
        self.parser.add_argument("--s", type=int, default=1)
        self.parser.add_argument("--load_classifier", type=str, default=None)
        self.parser.add_argument("--classify_first", action='store_true')
        self.parser.add_argument("--load_partial", type=str, default=None)
        self.parser.add_argument("--state_freeze", action='store_true', help='whether freeze the state when explore')

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args

args.description = args.name
args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.log_dir = 'snap/%s' % args.name

if args.suffix == 'none':
    args.suffix = ''

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')

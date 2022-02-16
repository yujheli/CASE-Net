import argparse
from util.torch_util import str_to_bool

class ArgumentParser():
    def __init__(self, mode='train'):
        self.parser = argparse.ArgumentParser(description='Unsupervised Domain Adaptation for Person Re-Identification')
        self.add_base_parameters()

        if mode == 'train':
            self.add_train_parameters()
            self.add_dataset_parameters()

        elif mode == 'eval':
            self.add_eval_parameters()
        
    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        base_params.add_argument('--model-dir', type=str, default='trained_models', help='Directory for saving trained models')
        base_params.add_argument('--model-path', type=str, default='', help='Pre-trained model path')
        base_params.add_argument('--extractor-path', type=str, default='', help='Pre-trained extractor path')
        base_params.add_argument('--decoder-path', type=str, default='', help='Pre-trained decoder path')
        base_params.add_argument('--generator-path', type=str, default='', help='Pre-trained generator path')
        base_params.add_argument('--classifier-path', type=str, default='', help='Pre-trained classifier path')
        base_params.add_argument('--discriminator-path', type=str, default='', help='Pre-trained discriminator path')
        base_params.add_argument('--var-path', type=str, default='', help='Pre-trained var path')
        base_params.add_argument('--D1-path', type=str, default='', help='Pre-trained discriminator path')
        base_params.add_argument('--D2-path', type=str, default='', help='Pre-trained discriminator path')
        base_params.add_argument('--acgan-path', type=str, default='', help='Pre-trained ACGAN discriminator path')
        base_params.add_argument('--backbone', type=str, default='resnet-101', help='Backbone for feature extractor')                
        base_params.add_argument('--gpu', type=int, default=0, help='Which GPU to use?')
        base_params.add_argument('--num-workers', type=int, default=4, help='Number of workers')
        base_params.add_argument('--batch-size', type=int, default=8, help='Batch size')


    def add_dataset_parameters(self):
        dataset_params = self.parser.add_argument_group('dataset')
        dataset_params.add_argument('--source-dataset', type=str, default='Duke', help='Source dataset')                
        dataset_params.add_argument('--target-dataset', type=str, default='Duke', help='Target dataset')                
        dataset_params.add_argument('--random-crop', type=str_to_bool, nargs='?', const=True, default=True, help='Use random crop?')                

    def add_train_parameters(self):
        train_params = self.parser.add_argument_group('train')
        train_params.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
        train_params.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
        train_params.add_argument('--power', type=float, default=0.9, help='power constant')
        train_params.add_argument('--num-steps', type=int, default=10, help='number of training steps')
        train_params.add_argument('--eval-steps', type=int, default=10, help='number of evaluation steps')
        train_params.add_argument('--image-steps', type=int, default=10, help='number of writing image steps')
        train_params.add_argument('--iter-size', type=int, default=1)
        train_params.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
        train_params.add_argument('--seed', type=int, default=1234, help='Random seed')
        train_params.add_argument('--rec-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use reconstruction loss?')
        train_params.add_argument('--cls-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use classification loss?')
        train_params.add_argument('--acgan-cls-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use acgan classification loss?')
        train_params.add_argument('--adv-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use adversarial loss?')
        train_params.add_argument('--acgan-adv-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use ACGAN adversarial loss?')
        train_params.add_argument('--acgan-dis-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use ACGAN adversarial loss?')
        train_params.add_argument('--dis-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use discriminator loss?')
        train_params.add_argument('--KL-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use KL divergence loss?')
        train_params.add_argument('--triplet-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use triplet loss?')
        train_params.add_argument('--gp-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use triplet loss?')
        train_params.add_argument('--diff-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use difference loss?')
        train_params.add_argument('--w-rec', type=float, default=0, help='weight for reconstruction loss')
        train_params.add_argument('--w-cls', type=float, default=0, help='weight for classification loss')
        train_params.add_argument('--w-acgan-cls', type=float, default=0, help='weight for acgan classification loss')
        train_params.add_argument('--w-adv', type=float, default=0, help='weight for adversarial loss')
        train_params.add_argument('--w-acgan-adv', type=float, default=0, help='weight for ACGAN adversarial loss')
        train_params.add_argument('--w-acgan-dis', type=float, default=0, help='weight for ACGAN adversarial loss')
        train_params.add_argument('--w-dis', type=float, default=0, help='weight for discriminator loss')
        train_params.add_argument('--w-KL', type=float, default=0, help='weight for KL divergence loss')
        train_params.add_argument('--w-global', type=float, default=0, help='weight for global triplet loss')
        train_params.add_argument('--w-local', type=float, default=0, help='weight for local triplet loss')
        train_params.add_argument('--w-gp', type=float, default=0, help='weight for gp loss')
        train_params.add_argument('--w-diff', type=float, default=0, help='weight for gp loss')
        train_params.add_argument('--eval-metric', type=str, default='rank', help='Eval metric rank/mAP')                
        train_params.add_argument('--dist-metric', type=str, default='L2', help='Retrieval distance metric') 
        train_params.add_argument('--rank', type=int, default=20, help='Rank')                


    def add_eval_parameters(self):
        eval_params = self.parser.add_argument_group('eval')
        eval_params.add_argument('--dataset', type=str, default='Duke', help='Eval dataset')                
        eval_params.add_argument('--eval-metric', type=str, default='rank', help='Eval metric rank/mAP')                
        eval_params.add_argument('--dist-metric', type=str, default='L2', help='Retrieval distance metric') 
        eval_params.add_argument('--rank', type=int, default=1, help='Rank')


    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict={ a.dest: getattr(args,a.dest,None) for a in group._group_actions }
            arg_groups[group.title] = group_dict
        return (args, arg_groups)

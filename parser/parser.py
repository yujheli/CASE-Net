import argparse
from util.torch_util import str_to_bool

class ArgumentParser():
    def __init__(self, mode='train'):
        self.parser = argparse.ArgumentParser(description='Unsupervised Domain Adaptation for Person Re-Identification')
        self.add_base_parameters()

        if mode == 'train':
            self.add_train_parameters()
            self.add_dataset_parameters()
        
    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        base_params.add_argument('--model-path', type=str, default='', help='Pre-trained model path')
        base_params.add_argument('--model-dir', type=str, default='', help='Directory for saving trained models')
        base_params.add_argument('--discriminator-path', type=str, default='', help='Pre-trained discriminator path')
        base_params.add_argument('--gpu', type=int, default=0, help='Which GPU to use?')
        base_params.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    
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
        train_params.add_argument('--iter-size', type=int, default=1)
        train_params.add_argument('--batch-size', type=int, default=8, help='training batch size')
        train_params.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
        train_params.add_argument('--seed', type=int, default=1, help='Random seed')
        train_params.add_argument('--rec-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use reconstruction loss?')        
        train_params.add_argument('--cls-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use classification loss?')        
        train_params.add_argument('--adv-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use adversarial loss?')        
        train_params.add_argument('--contra-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use contrastive loss?')        
        train_params.add_argument('--w-rec', type=float, default=0, help='weight for reconstruction loss')
        train_params.add_argument('--w-cls', type=float, default=0, help='weight for classification loss')
        train_params.add_argument('--w-adv', type=float, default=0, help='weight for adversarial loss')
        train_params.add_argument('--w-contra', type=float, default=0, help='weight for contrastive loss')
        
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

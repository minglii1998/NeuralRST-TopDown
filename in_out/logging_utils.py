import os
import os.path as osp

import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix

from in_out.util import get_logger

from torch.utils.tensorboard import SummaryWriter  

'''
the structure of model
the args parser config
tensorboard log
traing, validating and testing log
best validating result and corresponding testing result

specified by different seed
'''

def get_suffix_without_seed(ver, args):
    if args.use_dynamic_oracle:
        dynamic = '.dyn.'
    else:
        dynamic = '.sta.'
    suffix = ver + '.' + args.model + dynamic + 'bs.' + str(args.batch_size) + args.word_embedding + 'dim.' + str(args.word_dim) + \
        args.decode_layer + '.wd.' + str(args.gamma)
    
    if len(args.milestone) > 0:
        md_str = '.md'
        for v in args.milestone:
            md_str += str(v)
        suffix = suffix + md_str
    
    if args.keep_lstm:
        suffix = suffix + 'keep_lstm'
    
    suffix = suffix + args.special_tag
    
    if args.special_tag == 'no_record':
        suffix = 'no_record'

    return suffix

class SelfLogger(object):
    def __init__(self, ver:str, config):
        super(SelfLogger, self).__init__()
        # ver: ori/lm
        self.ver = ver
        self.out_dir = config.model_path
        self.seed = config.seed
        self.args = config

        # used as the directory of this model
        # if exist, can mkdir directly
        self.suffix_without_seed = get_suffix_without_seed(ver, self.args)
        self.model_dir = osp.join(self.out_dir,self.suffix_without_seed)
        if not osp.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # used as the dir of this model in this specific seed
        # if exist, then means this seed has been tested, then examin mannuly
        self.model_seed_dir = osp.join(self.model_dir,'seed'+str(self.seed))
        if osp.exists(self.model_seed_dir):
            print(self.model_seed_dir)
            print('The dir of the model of this seed has already exsist, please check it! Delete this dir or change a seed.')
            # raise AssertionError
        if not osp.exists(self.model_seed_dir):
            os.makedirs(self.model_seed_dir)

        # used as tensorboard file name
        self.suffix_with_seed = str(self.seed) + '.' + self.suffix_without_seed 
        self.tensorboard_file = osp.join(self.model_seed_dir,self.suffix_with_seed)
        self.tb_writer = SummaryWriter(self.tensorboard_file)

        self.args_config_file = osp.join(self.model_dir,'args_config.txt')
        self.args_config_pt = osp.join(self.model_dir,'args_config.pt')
        if osp.exists(self.args_config_pt):
            saved_args = torch.load(self.args_config_pt)
            saved_args.seed = self.seed
            # same then no need to save, not same need to check
            if saved_args != self.args :
                print('Please check args config! Mis match happens!')
                print('Saved args: \t',saved_args)
                print('Now args: \t', self.args)
                # raise AssertionError
        else:
            temp_args = self.args
            temp_args.seed = 0
            torch.save(temp_args,self.args_config_pt)

            self.write_dict(vars(temp_args),self.args_config_file)

        self.model_config_path = osp.join(self.model_dir,'model_config.txt')
        # self.training_logs_path = osp.join(self.model_seed_dir,'traing_logs.txt')

        # self.seed_acc_path = osp.join(self.model_dir,'seed_acc.txt')
        # self.seed_acc_pt_path = osp.join(self.model_dir,'seed_acc.pt')

        self.logger = get_logger("RSTParser", self.args.use_dynamic_oracle, self.model_seed_dir)

        self.loss_txt_p = osp.join(self.model_seed_dir,'loss_log')

        self.checkpoint_p = osp.join(self.model_seed_dir,'checkpoint')
        if not osp.exists(self.checkpoint_p):
            os.makedirs(self.checkpoint_p)
        self.checkpoint_f = osp.join(self.checkpoint_p,'model.pt')

        print('Logger init completed')


    def write_dict(self, dict_needed, path):
        with open(path,'w+') as f:
            for k in dict_needed.keys():
                f.write(str(k)+':'+str(dict_needed[k])+'\n')

    def write_model(self, model):
        if not osp.exists(self.model_config_path):
            with open(self.model_config_path,'w') as f:
                f.write(str(model))

    def info(self, content):
        self.logger.info(content)

    def get_alphabet_path(self,):
        return os.path.join(self.model_dir, 'alphabets/')

    def get_no_seed_path(self,):
        return self.model_dir

    def record_loss(self,losses,tag,epoch):
        with open(self.loss_txt_p,'a')as f:
            content = str(epoch)+ ':' + tag + ':' +str(losses) + '\n'
            f.write(content)

    def get_model_path(self,):
        return osp.join(self.model_seed_dir,'network.pt')

    def save_checkpoint(self, net, optimizer, epoch):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, self.checkpoint_f)
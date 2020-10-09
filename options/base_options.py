import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        
        parser.add_argument('--batchSize', type=int, default=2, help=argparse.SUPPRESS)
        parser.add_argument('--loadSize', type=int, default=550, help=argparse.SUPPRESS)
        parser.add_argument('--fineSize', type=int, default=512, help=argparse.SUPPRESS)
        parser.add_argument('--input_nc', type=int, default=3, help=argparse.SUPPRESS)
        parser.add_argument('--output_nc', type=int, default=3, help=argparse.SUPPRESS)
        parser.add_argument('--ngf', type=int, default=64, help=argparse.SUPPRESS)
        parser.add_argument('--which_model_netG', type=str, default='encoderVggDecoderResNet', help=argparse.SUPPRESS)
        parser.add_argument('--gpu_ids', type=str, default='0', help=argparse.SUPPRESS)
        parser.add_argument('--name', type=str, default='facefh_dictionary', help=argparse.SUPPRESS)
        parser.add_argument('--dataset_mode', type=str, default='aligned', help=argparse.SUPPRESS)
        parser.add_argument('--model', type=str, default='faceDict', help=argparse.SUPPRESS)
        parser.add_argument('--which_direction', type=str, default='BtoA', help=argparse.SUPPRESS)
        parser.add_argument('--nThreads', default=8, type=int, help=argparse.SUPPRESS)
        parser.add_argument('--checkpoints_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)) + '/../checkpoints', help=argparse.SUPPRESS)
        parser.add_argument('--norm', type=str, default='instance', help=argparse.SUPPRESS)
        parser.add_argument('--serial_batches', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--display_winsize', type=int, default=256, help=argparse.SUPPRESS)
        parser.add_argument('--display_id', type=int, default=1, help=argparse.SUPPRESS)
        parser.add_argument('--display_server', type=str, default="http://localhost", help=argparse.SUPPRESS)
        parser.add_argument('--display_env', type=str, default='main', help=argparse.SUPPRESS)
        parser.add_argument('--display_port', type=int, default=8097, help=argparse.SUPPRESS)
        parser.add_argument('--no_dropout', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help=argparse.SUPPRESS)
        parser.add_argument('--resize_or_crop', type=str, default='degradation', help=argparse.SUPPRESS)
        parser.add_argument('--no_flip', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--init_type', type=str, default='kaiming', help=argparse.SUPPRESS)
        parser.add_argument('--init_gain', type=float, default=0.02, help=argparse.SUPPRESS)
        parser.add_argument('--verbose', action='store_true', help=argparse.SUPPRESS)
        parser.add_argument('--suffix', default='', type=str, help=argparse.SUPPRESS)
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        
        opt, _ = parser.parse_known_args()
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode

        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

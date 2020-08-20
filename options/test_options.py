from .base_options import BaseOptions
import os

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot', type=str, default='/home/Data/AllDataImages/2018_FaceFH', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--dataroot', type=str, default='', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='', help='train, val, test, etc')
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--is_real', type=int, default=0, help='0:for test with degradation, 1: for real without degradation')
        parser.add_argument('--partroot', type=str, default='datasets/GRMouthVGG2/MergeGRVGG2WebFace/', help='path to roialign part locations')

        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))


        parser.add_argument('--p1', type=str, default='', help='')
        parser.add_argument('--p2', type=str, default='', help='')
        parser.add_argument('--p3', type=str, default='', help='')
        parser.add_argument('--p4', type=int, default=0, help='')
        parser.add_argument('--p5', type=str, default='', help='')
        parser.add_argument('--p6', type=str, default='', help='')

        # DeepVooDoo options
        parser.add_argument('--test_name', type=str, default='deepvoodoo', help='Name of the dataset.')
        parser.add_argument('--test_dir', type=str, default='/ParkCounty/home/DFDNet_data/deepvoodoo', help='Directory to import testing images.')
        parser.add_argument('--gpu_id', type=int, default=0, help="Which gpu to run on.  default is 0")
        parser.add_argument('--upscale', type=int, default=4, help="The upsample scale")
        parser.add_argument('--only_final', action="store_true", default=None, help='Only save the final output')
        parser.add_argument('--aligned_dir', type=str, default='', help='Directory to import aligned dst images. This is needed for running DFD on raw_predict merged data.')
        parser.add_argument('--aligned_postfix', type=str, default='_0.jpg', help='Postfix to replace the png format part in image path. Default is set to _0.jpg')
        self.isTrain = False
        return parser

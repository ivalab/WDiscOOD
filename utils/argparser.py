import argparse
import os, sys
import torch

class BaseArgs():
    """The base argument class contains the shared arguments
    """
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser("The argument for the OOD detection experiments")

        # ------------------- Train
        self.parser.add_argument("--train_save_root", type=str, default="./checkpoints")
        self.parser.add_argument("--train_exp_name", type=str, default="DiscTrain",
                                help="The subfolder name storing training outcomes/process")
        self.parser.add_argument("--save_freq", type=int, default=1)
        self.parser.add_argument("--print_freq", type=int, default=10)
        self.parser.add_argument("--train_bs", type=int, default=128,
                                help="The training batch size")
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument("--lr", type=float, default=1e-1)
        self.parser.add_argument("--lr_decay_period", type=float, default=30)
        self.parser.add_argument("--lr_decay_factor", type=float, default=0.1)
        self.parser.add_argument("--momentum", type=float, default=0.9)
        self.parser.add_argument("--weight_decay", type=float, default=1e-4)
        self.parser.add_argument("--epochs", type=int, default=10)
        self.parser.add_argument("--resume", type=str, default=None,
                                help="The folder which stores the info to resume the training from.")

        # ------------------- Test
        self.parser.add_argument('--test_bs', type=int, default=200)
        
        # ------------------- Datasets
        self.parser.add_argument('--id_dset', type=str, default="imagenet", choices=["imagenet"],
                                help="The in distribution dataset, which is where the network was trained on / will be trained on.")
        self.parser.add_argument('--ood_dsets', default=[], nargs="*", type=str, 
                                help="['inat', 'textures', 'openimage_o', 'imagenet_o', 'sun', 'places']")

        # ------------------- Network
        self.parser.add_argument('--arch', type=str, default="resnet50", 
                            choices=["resnet50", "resnet50_supcon", "resnet50_clip", "vit_b"])
        # self.parser.add_argument('--load_folder', type=str, default='', \
        #                     help="The model loading folder, in which the latest model with args.method_name will be loaded")
        self.parser.add_argument('--load_file', type=str, default=None, 
                            help="Provide the model file directly. Take higher priority than the args.load_folder option")
        self.parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
        self.parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

    
    def preprocess_args(self, args):
        # get the in distribution class number
        if args.id_dset == "imagenet":
            args.num_classes = 1000
            args.id_train_size_total = 1281167
        else:
            raise NotImplementedError

        # cuda
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load_file
        if args.load_file == "None":
            args.load_file = None
            
        return args

    def get_args(self):
        args = self.parser.parse_args()
        args = self.preprocess_args(args)
        return args
    
class FeatExtractArgs(BaseArgs):
    def __init__(self) -> None:
        super().__init__()

        file_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(file_dir)

        # ------------------- Save directory
        self.parser.add_argument('--save_folder', default=os.path.join(root_dir, "featCache"), type=str, 
                            help="The directory to store the features. ")

        self.parser.add_argument('--large_scale', action="store_true", default=False)
        self.parser.add_argument('--id_train_num', type=int, default=200000,
                            help="The in-distribution training data number for parameter fitting.")
        self.parser.add_argument('--rerun', action="store_true", 
                            help="Force rerun even if the file already exists.")
    
    def preprocess_args(self, args):
        super().preprocess_args(args)
        return args


class OODArgs(FeatExtractArgs):
    def __init__(self) -> None:
        super().__init__()

        self.parser.add_argument('--run_together', action="store_true",
                                help="Run all data in a dataset together. Default is run one-by-one.")
        # -------------------- investigation related
        self.parser.add_argument('--fig_save_folder', type=str, default=None, 
                            help="Figure save subfolder. If None, will save to figs/")
        self.parser.add_argument('--draw_curves', action='store_true')

        # ------------------- OOD score type and parameters. It only contains the baseline parameters in the base class.
        self.parser.add_argument('--score',type=str, default='MSP', 
                            choices = ["MSP", "Energy", "ODIN", "Maha", 
                            "maxLogit", "KLMatch", 
                            "WDiscOOD",
                            "MahaVanilla",
                            "ReAct",
                            "KNN", "VIM", "Residual"],
                            help="score type.")
        self.parser.add_argument('--feat_norm', type=int, default=0,
                            help="Option to normalize the feature. 0 for no and 1 for yes")

        # ------------------- For disc ood
        self.parser.add_argument('--alpha', type=float, default=1e-5, 
                            help="The within class scatter matrix positive bias")
        self.parser.add_argument('--mode', type=str, default="disc", help="'disc' or 'scatter'" )
        self.parser.add_argument('--feat_space', type=int, default=0, 
                            )
        self.parser.add_argument('--proj_error_th', type=float, default=0, 
                                help="The tolerance for the feature mapping")        
        self.parser.add_argument('--center_style', type=str, default="CENTER", 
                                choices=[None, "CENTER", "VIM"],
                                help="Center the features following the vim style.")

        self.parser.add_argument("--whiten_data", type=int, default=1,
                                help="Option to whiten data. 1 for yes; 0 for no")

        self.parser.add_argument('--topk_principal', type=int, default=1000,
                                help="The topk number of principle directions.")
        self.parser.add_argument("--num_disc", type=int, default=None,
                                help="The number of discriminants for the discriminative space")
        self.parser.add_argument("--num_disc_res", type=int, default=None,
                                help="The number of discriminants that is orthogonal to the residual space. If None, will use the same number as num_disc")
        self.parser.add_argument('--score_g', type=str, default="ClsEucl",
                            choices=["ClsEucl", "CenterEucl", "ClsMaha", "CenterMaha", "CenterClsMaha", "Norm"],
                            help="Distance choices. CenterClsMaha corresponds to distance w.r.t all center, but scaled using class precisions.")
        self.parser.add_argument('--score_h', type=str, default="CenterEucl",
                            choices=["ClsEucl", "CenterEucl", "ClsMaha", "CenterMaha", "CenterClsMaha", "Norm"])
        self.parser.add_argument('--res_dist_weight', type=float, default=None,
                                help="scaling factor of the residual space distant")
        self.parser.add_argument('--invs_train_dists', action="store_true")

    
    def preprocess_args(self, args):
        super().preprocess_args(args)
        args.fig_save_folder = f"figs/{args.fig_save_folder}" if args.fig_save_folder is not None else "figs/"
        args.feat_norm = (args.feat_norm == 1)
        args.whiten_data = (args.whiten_data == 1)
        args.num_disc_res = args.num_disc if args.num_disc_res is None else args.num_disc_res
        return args

if __name__ == "__main__":
    arg_parser = FeatExtractArgs()    
    args = arg_parser.get_args()


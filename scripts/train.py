"""
Basic training script for PyTorch
"""
import argparse
import os
import torch
import numpy as np
import sys
import json

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
sys.path.append(os.path.dirname(os.getcwd()))
from copy import deepcopy
from config.config import CONF
from lib.dataset import make_dataloader
from lib.solver import Solver
from models.visposnet import VisPosNet
from torch.utils.data import DataLoader
from datetime import datetime
from utils.model_util_scannet import ScannetDatasetConfig


torch.set_printoptions(precision=12)

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

DC = ScannetDatasetConfig()


def get_dataloader(CONF, args, scanrefer, all_scene_list, split):
    dataset, sent_per_batch = make_dataloader(
        CONF,
        scanrefer=scanrefer[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
    )
    dataloader = DataLoader(dataset, batch_size=sent_per_batch, shuffle=True, num_workers=CONF.DATALOADER.NUM_WORKERS)

    return dataset, dataloader


def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(args, scanrefer_train, scanrefer_val, num_scenes):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1:
            num_scenes = len(train_scene_list)  # 562
        else:
            assert len(train_scene_list) >= num_scenes

        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def train(CONF, args):
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.
                                                                                                              no_height)

    model = VisPosNet(input_feature_dim=input_channels,
                      mean_size_arr=DC.mean_size_arr,
                      num_proposal=args.num_proposals,
                      use_bidir=args.use_bidir,
                      no_reference=args.no_reference,
                      no_detection=args.no_detection,
                      use_lang_classifier=(not args.no_lang_cls),
                      num_class=DC.num_class,
                      num_heading_bin=DC.num_heading_bin,
                      num_size_cluster=DC.num_size_cluster)

    if args.use_pretrained:
        # load pretrained model
        print("loading pretrained VoteNet...")
        pretrained_model = VisPosNet(input_feature_dim=input_channels,
                                     mean_size_arr=DC.mean_size_arr,
                                     num_proposal=args.num_proposals,
                                     use_bidir=args.use_bidir,
                                     no_reference=True,
                                     no_detection=args.no_detection,
                                     use_lang_classifier=(not args.no_lang_cls),
                                     num_class=DC.num_class,
                                     num_heading_bin=DC.num_heading_bin,
                                     num_size_cluster=DC.num_size_cluster)
        pretrained_path = os.path.join(CONF.PATH.OUTPUT, "pretrained_model/model.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

    if args.no_detection:
        # freeze backbone
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        # freeze voting
        for param in model.vgen.parameters():
            param.requires_grad = False

        # freeze proposal
        for param in model.proposal.parameters():
            param.requires_grad = False

    device = torch.device("cuda")
    model.to(device)

    # choose an optimizer
    lr = CONF.SOLVER.BASE_LR
    if CONF.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=CONF.SOLVER.MOMENTUM, weight_decay=args.wd)
    elif CONF.SOLVER.TYPE == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=args.wd)
    else:
        raise NotImplementedError

    # load the saved model
    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(args, SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    # dataloader
    train_dataset, train_data_loader = get_dataloader(CONF, args, scanrefer, all_scene_list, "train")
    val_dataset, val_data_loader = get_dataloader(CONF, args, scanrefer, all_scene_list, "val")

    data_loader = {
        "train": train_data_loader,
        "val": val_data_loader
    }

    LR_DECAY_STEP = CONF.SOLVER.STEPS if args.no_reference else None
    LR_DECAY_RATE = CONF.SOLVER.GAMMA if args.no_reference else None
    BN_DECAY_STEP = 20 if args.no_reference else None
    BN_DECAY_RATE = 0.5 if args.no_reference else None

    solver = Solver(
        model,
        data_loader,
        optimizer,
        stamp,
        args.val_step,
        not args.no_detection,
        not args.no_reference,
        not args.no_lang_cls,
        LR_DECAY_STEP,
        LR_DECAY_RATE,
        BN_DECAY_STEP,
        BN_DECAY_RATE
    )

    num_params = get_num_params(model)
    return solver, num_params, root, train_dataset, val_dataset


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="PyTorch 3D Object Detection Training")
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="3")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=30)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the lanfe module.")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print("Using {} GPUs".format(num_gpus))

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # CONF.merge_from_list(args.opts)
    # CONF.freeze()

    print("initializing...")
    solver, num_params, root, train_dataset, val_dataset = train(CONF, args)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    main()

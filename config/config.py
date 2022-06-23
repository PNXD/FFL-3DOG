import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
# change it to your own directory
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/FMT/code/FFL-3DOG"
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.PARSER = os.path.join(CONF.PATH.BASE, "data_parser")
CONF.PATH.SNG = os.path.join(CONF.PATH.PARSER, "sng_parser")
CONF.PATH.P_DATA = os.path.join(CONF.PATH.SNG, "_data")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
CONF.PATH.LANGUAGE = os.path.join(CONF.PATH.DATA, "language")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# data
CONF.SCANNET_DIR = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.SCANNET_FRAMES_PATH = os.path.join(CONF.PATH.BASE, "frames_square")
CONF.ENET_FEATURES_ROOT = os.path.join(CONF.PATH.BASE, "enet_features")
CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}")
CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy")
CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_PATH, "{}/{}")
CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
# CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

# scannet
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 126
CONF.TRAIN.SEED = 42

# DataLoader
CONF.DATALOADER = EasyDict()
CONF.DATALOADER.NUM_WORKERS = 4

# model
CONF.LANREF = EasyDict()
CONF.LANREF.recognition_dim = 128
CONF.LANREF.Phrase_embed_dim = 128

CONF.MODEL = EasyDict()

CONF.MODEL.TOPN = 25

# phrase embedding
CONF.MODEL.VOCAB_FILE = "datasets/skip-thoughts/vocab.json"

# union-box embedding
CONF.MODEL.SceneBoxEmb = EasyDict()
CONF.MODEL.SceneBoxEmb.IN_DIM = 3
CONF.MODEL.SceneBoxEmb.OUT_DIM = 128

CONF.MODEL.BBOX_REG_WEIGHTS = (10., 10., 10., 5., 5., 5.)

# loss
CONF.MODEL.FG_IOU_THRESHOLD = 0.5
CONF.MODEL.BG_IOU_THRESHOLD = 0.3
CONF.MODEL.FG_REG_IOU_THRESHOLD = 0.5
CONF.MODEL.CLS_LOSS_TYPE = 'Softmax'
CONF.MODEL.PHRASE_SELECT_TYPE = 'Mean'

# Solver
CONF.SOLVER = EasyDict()
CONF.SOLVER.TYPE = 'SGD'
CONF.SOLVER.BASE_LR = 0.001
CONF.SOLVER.STEPS = [2, 20, 25]
CONF.SOLVER.GAMMA = 0.1
CONF.SOLVER.MOMENTUM = 0.9
CONF.SOLVER.REF_LR_FACTOR = 0.1

CONF.SOLVER.SENT_PER_BATCH = 6
CONF.SOLVER.REGLOSS_FACTOR = 0.1

CONF.SOLVER.TEST = EasyDict()
CONF.SOLVER.TEST.SENT_PER_BATCH = 6
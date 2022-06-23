import sys
import os

from lib.build_data import BuildDataset

sys.path.append(os.path.join(os.getcwd()))


def make_dataloader(CONF, scanrefer, scanrefer_all_scene, split='train', num_points=40000, use_height=False,
                    use_color=False, use_normal=False, use_multiview=False):
    if split == 'train':
        sent_per_batch = CONF.SOLVER.SENT_PER_BATCH
    else:
        sent_per_batch = CONF.SOLVER.TEST.SENT_PER_BATCH

    dataset = scanrefer
    datasets = BuildDataset(dataset, scanrefer_all_scene, split, num_points, use_height, use_color, use_normal,
                            use_multiview)

    return datasets, sent_per_batch

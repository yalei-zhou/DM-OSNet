#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
# from thop import profile
# from thop import clever_format
# from fastreid.modeling.backbones.dmosnet import build_osnet_backbone
# from argparse import Namespace
# from torchstat import stat

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)
        
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args = Namespace(config_file='/home/zyl/fast-reid/configs/osnet/bagtricks_R50-ibn.yml', dist_url='tcp://127.0.0.1:50152', eval_only=False, machine_rank=0, num_gpus=1, num_machines=1, opts=['MODEL.DEVICE', 'cuda:0'], resume=False)
    # args = Namespace(config_file='/home/zyl/fast-reid/configs/osnet/bagtricks_R50-ibn.yml', dist_url='tcp://127.0.0.1:50152', eval_only=False, machine_rank=0, num_gpus=1, num_machines=1, opts=['MODEL.DEVICE', 'cuda:0'], resume=False)
    # args = Namespace(config_file='/home/zyl/fast-reid/configs/Market1501/bagtricks_R50.yml', dist_url='tcp://127.0.0.1:49152', eval_only=True, machine_rank=0, num_gpus=1, num_machines=1, opts=['MODEL.WEIGHTS', '/home/zyl/fast-reid/lup_moco_r50.pth', 'MODEL.DEVICE', 'cuda:0'], resume=False)
    # model = build_osnet_backbone()
    # stat(model, (3,256,128))
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

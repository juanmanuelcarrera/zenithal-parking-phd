"""
Fine-tune Mask R-CNN R101-FPN from COCO on ZenithalParking.

SGD, lr 0.0025, momentum 0.9, frozen early backbone stages,
early stopping after 10 epochs without improvement on val segm AP.
"""
import argparse
import os

import detectron2.data.transforms as T
from detectron2.data import (DatasetCatalog, DatasetMapper, build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from .dataset import register_detectron2
from .maskrcnn import build_cfg


class StopTraining(Exception):
    pass


class EarlyStopping(HookBase):
    def __init__(self, period: int, patience: int = 10):
        self.period = period
        self.patience = patience
        self.best = -1.0
        self.bad_epochs = 0

    def after_step(self):
        if (self.trainer.iter + 1) % self.period:
            return
        cfg = self.trainer.cfg
        evaluator = COCOEvaluator("zenithal_val", output_dir=cfg.OUTPUT_DIR)
        loader = build_detection_test_loader(cfg, "zenithal_val")
        ap = inference_on_dataset(self.trainer.model, loader, evaluator)["segm"]["AP"]

        if ap > self.best:
            self.best, self.bad_epochs = ap, 0
            self.trainer.checkpointer.save("model_best")
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                raise StopTraining


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.RandomRotation([-15, 15]),
            T.ResizeScale(min_scale=0.8, max_scale=1.2, target_height=683, target_width=1024),
            T.FixedSizeCrop((683, 1024)),
            T.RandomBrightness(0.7, 1.3),
            T.RandomFlip(),
        ]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ZenithalParking")
    parser.add_argument("--out", default="runs/maskrcnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    register_detectron2(args.data)

    cfg = build_cfg(num_classes=1, device=args.device)
    cfg.DATASETS.TRAIN = ("zenithal_train",)
    cfg.DATASETS.TEST = ("zenithal_val",)
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.OUTPUT_DIR = args.out

    iters_per_epoch = max(1, len(DatasetCatalog.get("zenithal_train")) // args.batch)
    cfg.SOLVER.MAX_ITER = iters_per_epoch * args.epochs
    cfg.SOLVER.STEPS = []
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    early_stop = EarlyStopping(period=iters_per_epoch, patience=10)
    trainer.register_hooks([early_stop])
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except StopTraining:
        print(f"early stop, best segm AP {early_stop.best:.2f}")


if __name__ == "__main__":
    main()

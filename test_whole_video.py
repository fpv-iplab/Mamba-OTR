import os
import sys
import time
import torch
import pickle
import warnings
import numpy as np
import os.path as osp
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.models import build_model
from tools.generate_targets import generate_target
from rekognition_online_action_detection.models.transformer.position_encoding import PositionalEncoding
from rekognition_online_action_detection.evaluation import compute_result





class WholeVideoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, video_id):
        self.cfg = cfg
        self.video_id = video_id
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.object_feature = cfg.INPUT.OBJECT_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME

        self.frames = [np.load(osp.join(self.data_root, self.visual_feature, self.video_id + '.npy'), mmap_mode='r')]
        self.optical_flow = [np.load(osp.join(self.data_root, self.motion_feature, self.video_id + '.npy'), mmap_mode='r')]
        self.objects = [np.load(osp.join(self.data_root, self.object_feature, self.video_id + '.npy'), mmap_mode='r')] 
        self.target = [np.load(osp.join(self.data_root, "TARGET", self.target_perframe, self.video_id + '.npy'))]
        self.verb_target = [np.load(osp.join(self.data_root, "TARGET", self.target_perframe.replace('target', 'verb'), video_id + '.npy'))]
        self.noun_target = [np.load(osp.join(self.data_root, "TARGET", self.target_perframe.replace('target', 'noun'), video_id + '.npy'))]


    def __getitem__(self, index):
        visual = torch.from_numpy(self.frames[index].astype(np.float32))
        optical = torch.from_numpy(self.optical_flow[index].astype(np.float32))
        objects = torch.from_numpy(self.objects[index].astype(np.float32))
        target = self.target[index].astype(np.float32)
        verb_target = self.verb_target[index].astype(np.float32)
        noun_target = self.noun_target[index].astype(np.float32)
        return visual, optical, objects, (target, verb_target, noun_target)


    def __len__(self):
        return len(self.frames)





def test(cfg):
    times = []
    results = []
    video_length = []
    sessions = getattr(cfg.DATA, 'TEST_SESSION_SET')

    # Setup configurations
    generate_target(cfg)
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger = setup_logger(cfg, phase='test')

    # Build model
    model = build_model(cfg, device)
    logger.info("")
    logger.info("MODEL STRUCTURE:")
    logger.info(model)
    logger.info("")

    max_len = 0
    for video_id in sessions:
        video_len = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.VISUAL_FEATURE, video_id + '.npy'), mmap_mode='r').shape[0]
        if video_len > max_len:
            max_len = video_len
    max_len_rounded = int(np.ceil(max_len / 1000.0)) * 1000
    logger.info(f"Max video length: {max_len} (rounded up: {max_len_rounded})")

    checkpointer.load(model)
    pos_encoding = PositionalEncoding(model.d_model, model.dropout, max_len_rounded)
    model.pos_encoding = pos_encoding
    model.pos_encoding = model.pos_encoding.to(device)
    model.eval()


    video_preds = []
    for video in tqdm(sessions):
        dataset = WholeVideoDataset(cfg, video)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        for visual, optical, objects, target in data_loader:
            visual = visual.to(device)
            optical = optical.to(device)
            objects = objects.to(device)
            target, _, _ = target

            video_length.append(visual.shape[1])

            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                score = model(visual, optical, objects)

            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

            score, _, _ = score
            score = score.softmax(dim=-1).cpu().numpy()

            score = score.squeeze(0)
            target = target.squeeze(0)
            video_preds.append((score, target))


    all_scores = []
    all_targets = []
    for _, res in enumerate(video_preds):
        score, target = res
        all_scores.append(np.array(score))
        all_targets.append(np.array(target))

    all_scores = np.concatenate(all_scores, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    result_det = compute_result["perpoint"](
        cfg,
        all_targets,
        all_scores,
        class_names=cfg.DATA.VERB_NAMES
    )
    
    print(f"Mean Average Precision: {result_det['mp_mAP']:.5f}")

    perframe_time = [time / length for time, length in zip(times, video_length)]
    print(f"Mean Time per frame: {np.mean(np.array(perframe_time)):.5f}")

    fps = [length / time for time, length in zip(times, video_length)]
    print(f"Mean FPS: {np.mean(np.array(fps)):.5f}")





if __name__ == '__main__':
    test(load_cfg())

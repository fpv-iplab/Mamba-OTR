# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
from tqdm import tqdm
import pandas as pd

import torch
import numpy as np
import pickle as pkl

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result


def do_perframe_det_batch_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()
    cfg.MODEL.CRITERIONS = [['MCE', {}]]
    
    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect scores and targets
    pred_scores = {}
    pred_scores_verb = {}
    pred_scores_noun = {}
    gt_targets = {}
    vrb_target = {}
    nn_target = {}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-4]
            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                target, verb_target, noun_target = target

            score = model(*[x.to(device) for x in data[:-4]])
            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                score, score_verb, score_noun = score
                score = score.softmax(dim=-1).cpu().numpy()
                score_verb = score_verb.softmax(dim=-1).cpu().numpy()
                score_noun = score_noun.softmax(dim=-1).cpu().numpy()
                cfg.DATA.NUM_VERBS = 126 if cfg.DATA.DATA_NAME == 'EK55' else 98 if not cfg.DATA.TK_ONLY else cfg.DATA.NUM_VERBS
                cfg.DATA.NUM_NOUNS = 353 if cfg.DATA.DATA_NAME == 'EK55' else 301 if not cfg.DATA.TK_ONLY else cfg.DATA.NUM_NOUNS
            else:
                score = score.softmax(dim=-1).cpu().numpy()
            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        pred_scores_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))
                        pred_scores_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        vrb_target[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))
                        nn_target[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))

                if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                    pred_scores[session][query_indices] = score[bs]
                    gt_targets[session][query_indices] = target[bs]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        pred_scores_verb[session][query_indices] = score_verb[bs]
                        pred_scores_noun[session][query_indices] = score_noun[bs]

                        vrb_target[session][query_indices] = verb_target[bs]
                        nn_target[session][query_indices] = noun_target[bs]
                else:
                    pred_scores[session][query_indices[-1]] = score[bs][-1]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        pred_scores_verb[session][query_indices[-1]] = score_verb[bs][-1]
                        pred_scores_noun[session][query_indices[-1]] = score_noun[bs][-1]

                    gt_targets[session][query_indices[-1]] = target[bs][-1]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                        vrb_target[session][query_indices[-1]] = verb_target[bs][-1]
                        nn_target[session][query_indices[-1]] = noun_target[bs][-1]

    # Save scores and targets
    # pkl.dump({
    #     'cfg': cfg,
    #     'perframe_pred_scores': pred_scores,
    #     'perframe_gt_targets': gt_targets,
    # }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.pkl', 'wb'))

    # Compute results
    print("Computing all results")
    result_det = compute_result[cfg.EVALUATION.METHOD](
        cfg,
        np.concatenate(list(gt_targets.values()), axis=0),
        np.concatenate(list(pred_scores.values()), axis=0),
        class_names = cfg.DATA.CLASS_NAMES
    )
    if cfg.MODEL.LSTR.V_N_CLASSIFIER:
        result_verb = compute_result[cfg.EVALUATION.METHOD](
            cfg,
            np.concatenate(list(vrb_target.values()), axis=0),
            np.concatenate(list(pred_scores_verb.values()), axis=0),
            class_names = cfg.DATA.VERB_NAMES,
        )
        result_noun = compute_result[cfg.EVALUATION.METHOD](
            cfg,
            np.concatenate(list(nn_target.values()), axis=0),
            np.concatenate(list(pred_scores_noun.values()), axis=0),
            class_names = cfg.DATA.NOUN_NAMES
        )
    if cfg.EVALUATION.METHOD == "perpoint":
        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
            logger.info(f'Action perframe detection m{cfg.DATA.METRICS}: {result_det["mp_mAP"]:.5f},\
                verb m{cfg.DATA.METRICS}: {result_verb["mp_mAP"]:.5f},\
                noun m{cfg.DATA.METRICS}: {result_noun["mp_mAP"]:.5}')
        else:
            logger.info(f'Action perframe detection m{cfg.DATA.METRICS}: {result_det["mp_mAP"]:.5f}')
    else:
        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
            logger.info(f'Action perframe detection m{cfg.DATA.METRICS}: {result_det["mean_AP"]:.5f},\
                verb m{cfg.DATA.METRICS}: {result_verb["mean_AP"]:.5f},\
                noun m{cfg.DATA.METRICS}: {result_noun["mean_AP"]:.5}')
        else:
            logger.info(f'Action perframe detection m{cfg.DATA.METRICS}: {result_det["mean_AP"]:.5f}')
    return #! return here to avoid the following code

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import os.path as osp

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result


def execute_epoch(cfg, model, device, logger, data_loader):
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
                if cfg.OUTPUT.MODALITY == "action":
                    score_verb = score_verb.softmax(dim=-1).cpu().numpy()
                    score_noun = score_noun.softmax(dim=-1).cpu().numpy()
            else:
                score = score.softmax(dim=-1).cpu().numpy()

            if cfg.OUTPUT.MODALITY == "action":
                size = cfg.DATA.NUM_CLASSES
            elif cfg.OUTPUT.MODALITY == "verb":
                size = cfg.DATA.NUM_VERBS
            elif cfg.OUTPUT.MODALITY == "noun":
                size = cfg.DATA.NUM_NOUNS

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    pred_scores[session] = np.zeros((num_frames, size))
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                        pred_scores_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))
                        pred_scores_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, size))
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                        vrb_target[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS))
                        nn_target[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS))

                if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                    pred_scores[session][query_indices] = score[bs]
                    gt_targets[session][query_indices] = target[bs]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                        pred_scores_verb[session][query_indices] = score_verb[bs]
                        pred_scores_noun[session][query_indices] = score_noun[bs]

                        vrb_target[session][query_indices] = verb_target[bs]
                        nn_target[session][query_indices] = noun_target[bs]
                else:
                    pred_scores[session][query_indices[-1]] = score[bs][-1]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                        pred_scores_verb[session][query_indices[-1]] = score_verb[bs][-1]
                        pred_scores_noun[session][query_indices[-1]] = score_noun[bs][-1]

                    gt_targets[session][query_indices[-1]] = target[bs][-1]
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                        vrb_target[session][query_indices[-1]] = verb_target[bs][-1]
                        nn_target[session][query_indices[-1]] = noun_target[bs][-1]

    if cfg.SAVE != "":
        cfg.SAVE = osp.join(cfg.SAVE, cfg.DATA.DATA_NAME)

        if not osp.exists(cfg.SAVE):
            os.makedirs(cfg.SAVE)

        for session in pred_scores:
            np.save(osp.join(cfg.SAVE, f"{cfg.OUTPUT.MODALITY}_{session}.npy"), pred_scores[session])
            if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                np.save(osp.join(cfg.SAVE, f"verb_{session}.npy"), pred_scores_verb[session])
                np.save(osp.join(cfg.SAVE, f"noun_{session}.npy"), pred_scores_noun[session])

    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
        vrb_target = np.concatenate(list(vrb_target.values()), axis=0)
        pred_scores_verb = np.concatenate(list(pred_scores_verb.values()), axis=0)

        nn_target = np.concatenate(list(nn_target.values()), axis=0)
        pred_scores_noun = np.concatenate(list(pred_scores_noun.values()), axis=0)
    gt_targets = np.concatenate(list(gt_targets.values()), axis=0)
    pred_scores = np.concatenate(list(pred_scores.values()), axis=0)

    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
        return gt_targets, pred_scores, pred_scores_verb, vrb_target, pred_scores_noun, nn_target
    return gt_targets, pred_scores



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

    res = execute_epoch(cfg, model, device, logger, data_loader)

    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
        gt_targets, pred_scores, pred_scores_verb,vrb_target, pred_scores_noun, nn_target = res
    else:
        gt_targets, pred_scores = res

    # Save scores and targets
    # toSave = cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action"
    # pkl.dump({
    #     'cfg': cfg,
    #     'perframe_pred_scores': pred_scores,
    #     'perframe_gt_targets': gt_targets,
    #     "perframe_pred_scores_verb": pred_scores_verb if toSave else None,
    #     "perframe_pred_scores_noun": pred_scores_noun if toSave else None,
    #     "perframe_vrb_target": vrb_target if toSave else None,
    #     "perframe_nn_target": nn_target if toSave else None
    # }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.pkl', 'wb'))

    # Compute results
    if cfg.OUTPUT.MODALITY == "action":
        class_names = cfg.DATA.CLASS_NAMES
    elif cfg.OUTPUT.MODALITY == "verb":
        class_names = cfg.DATA.VERB_NAMES
    elif cfg.OUTPUT.MODALITY == "noun":
        class_names = cfg.DATA.NOUN_NAMES

    print("Computing all results")
    result_det = compute_result["perpoint"](
        cfg,
        gt_targets,
        pred_scores,
        class_names = class_names
    )
    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
        result_verb = compute_result["perpoint"](
            cfg,
            vrb_target,
            pred_scores_verb,
            class_names = cfg.DATA.VERB_NAMES,
        )
        result_noun = compute_result["perpoint"](
            cfg,
            nn_target,
            pred_scores_noun,
            class_names = cfg.DATA.NOUN_NAMES,
        )

    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
        logger.info(f'Action perframe detection mp_mAP: {result_det["mp_mAP"]:.5f},\
            verb mp_mAP: {result_verb["mp_mAP"]:.5f},\
            noun mp_mAP: {result_noun["mp_mAP"]:.5}')
    else:
        logger.info(f'{cfg.OUTPUT.MODALITY.capitalize()} perframe detection mp_mAP: {result_det["mp_mAP"]:.5f}')

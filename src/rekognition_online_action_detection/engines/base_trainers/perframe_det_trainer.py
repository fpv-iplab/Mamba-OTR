# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from tqdm import tqdm

import torch
import torch.nn as nn

from rekognition_online_action_detection.evaluation import compute_result
from rekognition_online_action_detection.engines.base_inferences.perframe_det_batch_inference import execute_epoch



def preprocess_end_label(data, delta=3):
    for b in range(data.shape[0]):
        for i in range(data.shape[1]):
            idx = torch.where(data[b, i, :] == 1)[0]
            if idx[0] > 0:
                if i - delta < 0:
                    data[b, :i, idx] = 1
                    data[b, :i, 0] = 0
                else:
                    data[b, i - delta:i, idx] = 1
                    data[b, i - delta:i, 0] = 0
    return data



def do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          device,
                          checkpointer,
                          logger):
    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        det_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        verb_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        noun_losses = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        det_pred_scores = []
        det_gt_targets = []
        verb_pred_scores = []
        verb_gt_targets = []
        noun_pred_scores = []
        noun_gt_targets = []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)

            with torch.set_grad_enabled(training):
                if training:
                    pbar = tqdm(data_loaders[phase],
                                desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                    for batch_idx, data in enumerate(pbar, start=1):
                        batch_size = data[0].shape[0]
                        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                            det_target, verb_target, noun_target = data[-1]
                            if cfg.MODEL.FRAME_DELTA > 0 and "end" in cfg.INPUT.TARGET_PERFRAME:
                                det_target = preprocess_end_label(det_target, cfg.MODEL.FRAME_DELTA)
                                verb_target = preprocess_end_label(verb_target, cfg.MODEL.FRAME_DELTA)
                                noun_target = preprocess_end_label(noun_target, cfg.MODEL.FRAME_DELTA)

                                det_target = det_target.to(device)
                                verb_target = verb_target.to(device)
                                noun_target = noun_target.to(device)
                            else:
                                det_target = det_target.to(device)
                                verb_target = verb_target.to(device)
                                noun_target = noun_target.to(device)
                        else:
                            if cfg.MODEL.FRAME_DELTA > 0 and "end" in cfg.INPUT.TARGET_PERFRAME:
                                det_target = preprocess_end_label(data[-1], cfg.MODEL.FRAME_DELTA)
                                det_target = det_target.to(device)
                            else:
                                det_target = data[-1].to(device)

                        loss_names = list(zip(*cfg.MODEL.CRITERIONS))[0]
                        det_score = model(*[x.to(device) for x in data[:-1]])

                        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                            det_score, verb_score, noun_score = det_score 
                        else:
                            verb_score, noun_score = None, None

                        if cfg.OUTPUT.MODALITY == "action":
                            reshape_size = cfg.DATA.NUM_CLASSES
                        elif cfg.OUTPUT.MODALITY == "verb":
                            reshape_size = cfg.DATA.NUM_VERBS
                        elif cfg.OUTPUT.MODALITY == "noun":
                            reshape_size = cfg.DATA.NUM_NOUNS

                        det_score = det_score.reshape(-1, reshape_size)
                        det_target = det_target.reshape(-1, reshape_size)

                        if cfg.DATA.DATA_NAME == "EK100" and cfg.OUTPUT.MODALITY == "action":
                            det_loss = criterion["MCE_EQL"](det_score, det_target)
                        else:
                            det_loss = criterion[loss_names[0]](det_score, det_target)
                        det_losses[phase] += det_loss.item() * batch_size

                        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                            if cfg.OUTPUT.MODALITY == "action":
                                verb_score = verb_score.reshape(-1, cfg.DATA.NUM_VERBS)
                                verb_target = verb_target.reshape(-1, cfg.DATA.NUM_VERBS)
                                verb_loss = criterion['MCE'](verb_score, verb_target)
                                verb_losses[phase] += verb_loss.item() * batch_size

                            if cfg.OUTPUT.MODALITY == "action":
                                noun_score = noun_score.reshape(-1, cfg.DATA.NUM_NOUNS)
                                noun_target = noun_target.reshape(-1, cfg.DATA.NUM_NOUNS)
                                noun_loss = criterion['MCE'](noun_score, noun_target)
                                noun_losses[phase] += noun_loss.item() * batch_size

                        # Output log for current batch
                        pbar.set_postfix({
                            'lr': '{:.7f}'.format(scheduler.get_last_lr()[0]),
                            'det_loss': '{:.5f}'.format(det_loss.item()),
                        })

                        if training:
                            optimizer.zero_grad()
                            loss = det_loss
                            if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                                loss += noun_loss + verb_loss
                            if loss.item() != 0:
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                else:
                    res = execute_epoch(cfg, model, device, logger, data_loaders[phase])
                    if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                        det_gt_targets, det_pred_scores, verb_pred_scores, verb_gt_targets, noun_pred_scores, noun_gt_targets = res
                    else:
                        det_gt_targets, det_pred_scores = res
        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        text = "det" if cfg.OUTPUT.MODALITY == "action" else cfg.OUTPUT.MODALITY
        log.append('train {}_loss: {:.5f}'.format(text,
            det_losses['train'] / len(data_loaders['train'].dataset),
        ))
        if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
            log.append('train verb_loss: {:.5f}'.format(
                verb_losses['train'] / len(data_loaders['train'].dataset),
            ))
            log.append('train noun_loss: {:.5f}'.format(
                noun_losses['train'] / len(data_loaders['train'].dataset),
            ))
        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                verb_result = compute_result["perpoint"](
                    cfg,
                    verb_gt_targets,
                    verb_pred_scores,
                    class_names=cfg.DATA.VERB_NAMES,
                )
                noun_result = compute_result["perpoint"](
                    cfg,
                    noun_gt_targets,
                    noun_pred_scores,
                    class_names=cfg.DATA.NOUN_NAMES,
                )

            if cfg.OUTPUT.MODALITY == "verb":
                class_names = cfg.DATA.VERB_NAMES
            elif cfg.OUTPUT.MODALITY == "noun":
                class_names = cfg.DATA.NOUN_NAMES
            else:
                class_names = cfg.DATA.CLASS_NAMES

            det_result = compute_result["perpoint"](
                cfg,
                det_gt_targets,
                det_pred_scores,
                class_names=class_names,
            )

            log.append('{}_mp_mAP: {:.5f}'.format(text,
                det_result['mp_mAP']
            ))
            if cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                log.append('test verb_loss: {:.5f}'.format(
                    verb_losses['test'] / len(data_loaders['test'].dataset),
                ))
                log.append('test noun_loss: {:.5f}'.format(
                    noun_losses['test'] / len(data_loaders['test'].dataset),
                ))
                log.append('test verb_mp_mAP: {:.5f}'.format(
                    verb_result['mp_mAP'],
                ))
                log.append('test noun_mp_mAP: {:.5f}'.format(
                    noun_result['mp_mAP'],
                ))
        log.append('running time: {:.2f} sec'.format(
            end - start,
        ))
        logger.info(' | '.join(log))

        # Save checkpoint for model and optimizer
        if epoch % cfg.SOLVER.SAVE_EVERY == 0:
            if cfg.DATA.TK_ONLY and cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY == "action":
                checkpointer.save(epoch, model, optimizer, verb_result['mp_mAP'])
            else:
                checkpointer.save(epoch, model, optimizer, det_result['mp_mAP'])

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()

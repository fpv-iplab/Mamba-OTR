# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from multiprocessing import Pool
from collections import OrderedDict

import numpy as np
from sklearn.metrics import average_precision_score


def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> float:
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        prec (np.ndarray): Precision array.
        rec (np.ndarray): Recall array.

    Returns:
        float: Interpolated AP.
    """
    
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def temporal_offset(target_AS: float, candidate_AS: np.ndarray) -> np.ndarray: 
    """Compute the temporal offset between a target AS and all the test AS.
    Taken from: https://github.com/rosarioscavo/actionformer_release/blob/main/metric/odas_enigma_clean.py

    Args:
        target_AS (float): Ground truth action start considered as target.
        candidate_AS (np.ndarray): Predicted action starts considered as candidates.
    Returns:
        np.ndarray: Temporal offset between a target AS and all the test AS.
    """

    result = np.absolute(candidate_AS - target_AS)
    return result


def point_average_precision(ground_truth: np.ndarray,
                                    prediction: np.ndarray,
                                    tOffset_thresholds: np.ndarray,
                                    fps: float = 4.0) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and
    predictions. If multiple predictions occurs for the same
    predicted segment, only the one with smallest offset is matches as
    true positive.
    Inspired by: https://github.com/rosarioscavo/actionformer_release/blob/main/metric/odas_enigma_clean.py

    Args:
        ground_truth (np.ndarray): Ground truth of action starts.
        prediction (np.ndarray): Predictions of action starts.
        tOffset_thresholds (np.ndarray): Temporal offset thresholds in seconds.
        fps (float): Frame rate of the video.

    Returns:
        np.ndarray: Average precision score for each tOffset_threshold.
    """

    tOffset_thresholds = tOffset_thresholds * fps

    num_pos = float(len(ground_truth))
    size = (len(tOffset_thresholds), len(prediction))
    lock_gt = np.full((len(tOffset_thresholds), len(ground_truth)), -1)

    tp = np.zeros(size)
    fp = np.zeros(size)
    pred_idx = prediction.argsort(axis=0)[::-1]
    prediction = prediction[pred_idx]

    for idx, this_pred in enumerate(prediction):
        t_off = temporal_offset(this_pred, ground_truth)
        t_off_sorted_idx = t_off.argsort()

        for tidx, toff_thr in enumerate(tOffset_thresholds):
            for jdx in t_off_sorted_idx:
                if t_off[jdx] > toff_thr:
                    fp[tidx, idx] = 1
                    break

                if lock_gt[tidx, jdx] >= 0:
                    continue

                tp[tidx, idx] = 1
                lock_gt[tidx, jdx] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)

    recall_cumsum = tp_cumsum / num_pos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = np.zeros(len(tOffset_thresholds))
    for tidx in range(len(tOffset_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])
    return ap.mean()


def convert_to_timestamp(data: np.ndarray) -> np.ndarray:
    """Convert action start frame to timestamp.

    Args:
        data (np.ndarray): Action start frame.

    Returns:
        np.ndarray: Action start timestamp.
    """
    timestamp = [i for i in range(len(data)) if data[i] != 0] #! != 0, no threshold applied. Future work available
    return np.array(timestamp)


def preprocess_pred(data: np.ndarray, threshold: float) -> np.ndarray:
    """Preprocess prediction data. Applies thresholding to remove low
    confidence predictions and convert to timestamp.

    Args:
        data (np.ndarray): Prediction data.
        threshold (float): Threshold value.

    Returns:
        np.ndarray: Preprocessed prediction data.
    """
    pred = np.where(data > threshold, data, 0)
    return convert_to_timestamp(pred)



def perframe_average_precision(ground_truth,
                               prediction,
                               class_names,
                               ignore_index,
                               metrics,
                               postprocessing):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)
    tOffset_thresholds = np.linspace(1.0, 10.0, 10)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    elif metrics == 'pAP':
        compute_score = point_average_precision
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                if metrics == 'pAP':
                    gt = convert_to_timestamp(ground_truth[:, idx])
                    pred = preprocess_pred(prediction[:, idx], threshold=0.005) #! Fixed threshold value for 
                                                                                #! "timestamp regression" emulation

                    result['per_class_AP'][class_name] = compute_score(
                        gt, prediction[:, idx], tOffset_thresholds)
                else:
                    result['per_class_AP'][class_name] = compute_score(
                        ground_truth[:, idx], prediction[:, idx])
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))

    return result


def get_stage_pred_scores(gt_targets, pred_scores, perc_s, perc_e):
    starts = []
    ends = []
    stage_gt_targets = []
    stage_pred_scores = []
    for i in range(len(gt_targets)):
        if gt_targets[i] == 0:
            stage_gt_targets.append(gt_targets[i])
            stage_pred_scores.append(pred_scores[i])
        else:
            if i == 0 or gt_targets[i - 1] == 0:
                starts.append(i)
            if i == len(gt_targets) - 1 or gt_targets[i + 1] == 0:
                ends.append(i)
    if len(starts) != len(ends):
        raise ValueError('starts and ends cannot pair!')

    action_lens = [ends[i] - starts[i] for i in range(len(starts))]
    stage_starts = [starts[i] + int(action_lens[i] * perc_s) for i in range(len(starts))]
    stage_ends = [max(stage_starts[i] + 1, starts[i] + int(action_lens[i] * perc_e)) for i in range(len(starts))]
    for i in range(len(starts)):
        stage_gt_targets.extend(gt_targets[stage_starts[i]: stage_ends[i]])
        stage_pred_scores.extend(pred_scores[stage_starts[i]: stage_ends[i]])
    return np.array(stage_gt_targets), np.array(stage_pred_scores)


def perstage_average_precision(ground_truth,
                               prediction,
                               class_names,
                               ignore_index,
                               metrics,
                               postprocessing):
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    for perc_s in range(10):
        perc_e = perc_s + 1
        stage_name = '{:2}%_{:3}%'.format(perc_s * 10, perc_e * 10)
        result[stage_name] = OrderedDict({'per_class_AP': OrderedDict()})
        for idx, class_name in enumerate(class_names):
            if idx not in ignore_index:
                stage_gt_targets, stage_pred_scores = get_stage_pred_scores(
                    (ground_truth[:, idx] == 1).astype(int),
                    prediction[:, idx],
                    perc_s / 10,
                    perc_e / 10,
                )
                result[stage_name]['per_class_AP'][class_name] = \
                    compute_score(stage_gt_targets, stage_pred_scores)
        result[stage_name]['mean_AP'] = \
            np.mean(list(result[stage_name]['per_class_AP'].values()))

    return result

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections import OrderedDict


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
                            prediction: OrderedDict[np.ndarray, np.ndarray]) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and
    predictions. If multiple predictions occurs for the same
    predicted segment, only the one with smallest offset is matches as
    true positive.
    Inspired by: https://github.com/rosarioscavo/actionformer_release/blob/main/metric/odas_enigma_clean.py

    Args:
        ground_truth (np.ndarray): Ground truth of actiona.
        prediction (OrderedDict): Predictions of actions.

    Returns:
        np.ndarray: Average precision score for each tOffset_threshold.
    """
    pred_ts = prediction["timestamp"]
    tOffset_thresholds = np.linspace(1.0, 10.0, 10)

    ap = np.zeros(len(tOffset_thresholds))
    if pred_ts.shape[0] == 0:
        return ap

    num_pos = float(len(ground_truth))
    size = (len(tOffset_thresholds), len(pred_ts))
    lock_gt = np.full((len(tOffset_thresholds), len(ground_truth)), -1)

    tp = np.zeros(size)
    fp = np.zeros(size)
    pred_idx = prediction["score"].argsort(axis=0)[::-1]
    pred_ts = pred_ts[pred_idx]

    for idx, this_pred in enumerate(pred_ts):
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

    for tidx in range(len(tOffset_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])
    return ap


def preprocess_pred(data: np.ndarray, threshold: float = 0.005, fps: float = 4.0) -> np.ndarray:
    """Preprocess prediction data. Applies thresholding to remove low
    confidence predictions and convert to timestamp.

    Args:
        data (np.ndarray): Prediction data.
        threshold (float): Very small threshold value to remove very low confidence predictions.
        fps (float): Frame rate of the video.

    Returns:
        np.ndarray: Preprocessed prediction data.
    """
    pred = OrderedDict()
    pred["timestamp"] = np.arange(data.shape[0]) / fps
    pred["score"] = data

    idx = np.where(data > threshold)
    pred["timestamp"] = pred["timestamp"][idx]
    pred["score"] = pred["score"][idx]
    return pred


def perframe_perpoint_average_precision(ground_truth,
                               prediction,
                               class_names,
                               ignore_index,
                               fps,
                               postprocessing):
    """Compute (frame-level) perpoint average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Ignore backgroud class
    ignore_index = set([0, *ignore_index])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                    gt = np.where(ground_truth[:, idx] != 0)[0] / fps
                    pred = preprocess_pred(prediction[:, idx], threshold=0.005, fps=fps)
                    result['per_class_AP'][class_name] = point_average_precision(gt, pred)

    result["p_mAP"] = np.mean(list(result['per_class_AP'].values()), axis=1)
    result["mp_mAP"] = np.mean(result["p_mAP"])
    return result

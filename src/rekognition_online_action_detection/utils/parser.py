# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import argparse
import json

from rekognition_online_action_detection.config.defaults import get_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Rekognition Face Anti-Spoofing')
    parser.add_argument(
        '--config_file',
        default='',
        type=str,
        help='path to yaml config file',
    )
    parser.add_argument(
        '--gpu',
        default='0',
        type=str,
        help='specify visible devices'
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="save evaluation outputs to disk",
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs='*',
        help='modify config options using the command-line',
    )
    return parser.parse_args()


def assert_and_infer_cfg(cfg, args):
    # Setup the visible devices
    cfg.GPU = args.gpu
    cfg.SAVE = args.save
    cfg.CONFIG_FILE = args.config_file

    # Infer data info
    with open(cfg.DATA.DATA_INFO, 'r') as f:
        data_info = json.load(f)[cfg.DATA.DATA_NAME]

    cfg.DATA.DATA_ROOT = data_info['data_root'] if cfg.DATA.DATA_ROOT is None else cfg.DATA.DATA_ROOT
    cfg.DATA.CLASS_NAMES = data_info['class_names'] if cfg.DATA.CLASS_NAMES is None else cfg.DATA.CLASS_NAMES
    cfg.DATA.VERB_NAMES = data_info['verb_names'] if cfg.DATA.VERB_NAMES is None and cfg.DATA.DATA_NAME == "EK100" else cfg.DATA.VERB_NAMES
    cfg.DATA.NOUN_NAMES = data_info['noun_names'] if cfg.DATA.NOUN_NAMES is None and cfg.DATA.DATA_NAME == "EK100" else cfg.DATA.NOUN_NAMES
    cfg.DATA.NUM_VERBS = data_info['num_verbs'] if cfg.DATA.NUM_VERBS is None and cfg.DATA.DATA_NAME == "EK100" else cfg.DATA.NUM_VERBS
    cfg.DATA.NUM_NOUNS = data_info['num_nouns'] if cfg.DATA.NUM_NOUNS is None and cfg.DATA.DATA_NAME == "EK100" else cfg.DATA.NUM_NOUNS
    cfg.DATA.NUM_CLASSES = data_info['num_classes'] if cfg.DATA.NUM_CLASSES is None else cfg.DATA.NUM_CLASSES
    cfg.DATA.TK_IDXS = data_info['tk_idxs'] if cfg.DATA.TK_IDXS is None and cfg.DATA.DATA_NAME == "EK100" else cfg.DATA.TK_IDXS
    if cfg.DATA.TK_ONLY:
        cfg.DATA.NUM_VERBS = 3
        cfg.DATA.NUM_CLASSES = len(cfg.DATA.TK_IDXS)
        cfg.DATA.VERB_NAMES = cfg.DATA.VERB_NAMES[:3]
        cfg.DATA.CLASS_NAMES = [cfg.DATA.CLASS_NAMES[i] for i in cfg.DATA.TK_IDXS]

    cfg.DATA.IGNORE_INDEX = data_info['ignore_index'] if cfg.DATA.IGNORE_INDEX is None else cfg.DATA.IGNORE_INDEX
    cfg.DATA.FPS = data_info['fps'] if cfg.DATA.FPS is None else cfg.DATA.FPS
    cfg.DATA.TRAIN_SESSION_SET = data_info['train_session_set'] if cfg.DATA.TRAIN_SESSION_SET is None else cfg.DATA.TRAIN_SESSION_SET
    cfg.DATA.TEST_SESSION_SET = data_info['test_session_set'] if cfg.DATA.TEST_SESSION_SET is None else cfg.DATA.TEST_SESSION_SET

    print(f"Data root: {cfg.DATA.DATA_ROOT}")
    cfg.DATA.DATA_ROOT = osp.join(osp.dirname(cfg.DATA.DATA_ROOT), f"{cfg.DATA.FPS}fps")
    print(f"Data root: {cfg.DATA.DATA_ROOT}")

    # Input assertions
    if cfg.INPUT.MODALITY == 'twostream':
        cfg.INPUT.MODALITY = 'visual+motion'
    if cfg.INPUT.MODALITY == 'threestream':
        cfg.INPUT.MODALITY = 'visual+motion+object'
    assert cfg.INPUT.MODALITY in ['visual', 'motion', 'object',
                                  'visual+motion', 'visual+motion+object']

    # Output assertions
    assert cfg.OUTPUT.MODALITY in ["verb", "noun", "action"]
    if not cfg.MODEL.LSTR.V_N_CLASSIFIER and cfg.OUTPUT.MODALITY != "action":
        cfg.OUTPUT.MODALITY = "action" #! default to action if V-N classifier is not used and output is wrongly set to verb/noun

    if cfg.OUTPUT.MODALITY == "verb":
        cfg.INPUT.TARGET_PERFRAME = cfg.INPUT.TARGET_PERFRAME.replace('target', 'verb' if not cfg.DATA.TK_ONLY else "verb_tk")
    if cfg.OUTPUT.MODALITY == "noun":
        cfg.INPUT.TARGET_PERFRAME = cfg.INPUT.TARGET_PERFRAME.replace('target', 'noun' if not cfg.DATA.TK_ONLY else "noun_tk")

    # Infer memory
    if cfg.MODEL.MODEL_NAME in ['LSTR']:
        cfg.MODEL.LSTR.AGES_MEMORY_LENGTH = cfg.MODEL.LSTR.AGES_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.LONG_MEMORY_LENGTH = cfg.MODEL.LSTR.LONG_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.WORK_MEMORY_LENGTH = cfg.MODEL.LSTR.WORK_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.TOTAL_MEMORY_LENGTH = \
            cfg.MODEL.LSTR.AGES_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.LONG_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        assert cfg.MODEL.LSTR.AGES_MEMORY_LENGTH % cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.LONG_MEMORY_LENGTH % cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.WORK_MEMORY_LENGTH % cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE == 0
        cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.AGES_MEMORY_LENGTH // cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH // cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.TOTAL_MEMORY_NUM_SAMPLES = \
            cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

        assert cfg.MODEL.LSTR.INFERENCE_MODE in ['batch', 'stream']

    # Infer output dir
    config_name = osp.splitext(args.config_file)[0].split('/')[1:]
    cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, *config_name)
    if cfg.SESSION:
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.SESSION)


def load_cfg(args = None):
    if args is None:
        args = parse_args()
    cfg = get_cfg()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    assert_and_infer_cfg(cfg, args)
    return cfg

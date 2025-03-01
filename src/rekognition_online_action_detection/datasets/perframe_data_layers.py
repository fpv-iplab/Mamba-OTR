# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd

from .datasets import DATA_LAYERS as registry
from rekognition_online_action_detection.utils.ek_utils import (action_to_noun_map, action_to_verb_map)


@registry.register('LSTRENIGMA')
@registry.register('LSTRTHUMOS')
@registry.register('LSTRTVSeries')
@registry.register('LSTREK55')
@registry.register('LSTREK100')
class LSTRDataLayer(data.Dataset):

    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_name = cfg.DATA.DATA_NAME
        self.data_root = cfg.DATA.DATA_ROOT
        self.clip_mixup_rate = cfg.DATA.CLIP_MIXUP_RATE
        self.clip_mixup_sample = cfg.DATA.CLIP_MIXUP_SAMPLE
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.object_feature = cfg.INPUT.OBJECT_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.training = phase == 'train'

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        if self.data_name.startswith('EK'):
            if self.data_name == 'EK55':
                path_to_data = 'external/rulstm/RULSTM/data/ek55/'
            elif self.data_name == 'EK100':
                path_to_data = 'external/rulstm/RULSTM/data/ek100/'
            else:
                raise ValueError
            segment_list = pd.read_csv(osp.join(path_to_data,
                                                'training.csv' if self.training else 'validation.csv'),
                                       names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                       skipinitialspace=True)
            segment_list['len_f'] = segment_list['end_f'] - segment_list['start_f']

            if self.cfg.DATA.TK_ONLY:
                self.segment_by_cls = {cls: segment_list[segment_list['action'] == cls]
                                   for cls in self.cfg.DATA.TK_IDXS}
            else:
                self.segment_by_cls = {cls: segment_list[segment_list['action'] == cls]
                                   for cls in range(0, self.cfg.DATA.NUM_CLASSES - 1)}
            for session in self.sessions:
                target = np.load(osp.join(self.data_root, "TARGET", self.target_perframe, session + '.npy'))
                if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
                    verb_target = np.load(osp.join(self.data_root, "TARGET", self.target_perframe.replace('target', 'verb'), session + '.npy'))
                    noun_target = np.load(osp.join(self.data_root, "TARGET", self.target_perframe.replace('target', 'noun'), session + '.npy'))
                else:
                    verb_target = target
                    noun_target = target
                segments_per_session = segment_list[segment_list['video'] == session]
                for segment in segments_per_session.iterrows():
                    start_tick = int(segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                    end_tick = int(segment[1]['end_f'] / 30 * self.cfg.DATA.FPS)
                    start_tick = min(start_tick, end_tick)
                    work_end = start_tick
                    work_start = work_end - self.work_memory_length
                    segments_before_current = segments_per_session[segments_per_session['end_f'] < segment[1]['start_f']]
                    if work_start < 0:
                        continue
                    self.inputs.append([
                        session, work_start, work_end,
                        target[work_start: work_end],
                        verb_target[work_start: work_end],
                        noun_target[work_start: work_end],
                        segments_before_current,
                    ])
        else:
            if self.data_name == 'ENIGMA':
                for video_id in ["126", "107", "49", "66", "69", "104", "111", "145", "156"]:
                    try:
                        self.sessions.remove(video_id)
                    except:
                        pass
            for session in self.sessions:
                target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
                if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
                    verb_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'verb'), session + '.npy'))
                    noun_target = np.load(osp.join(self.data_root, self.target_perframe.replace('target', 'noun'), session + '.npy'))
                else:
                    verb_target = target
                    noun_target = target
                seed = np.random.randint(self.work_memory_length) if self.training else 0
                for work_start, work_end in zip(
                    range(seed, target.shape[0], self.work_memory_length),
                    range(seed + self.work_memory_length, target.shape[0], self.work_memory_length)):
                    self.inputs.append([
                        session, work_start, work_end,
                        target[work_start: work_end],
                        verb_target[work_start: work_end],
                        noun_target[work_start: work_end],
                        None,
                    ])



    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        (session, work_start, work_end, target,
         verb_target, noun_target, segments_before) = self.inputs[index]
        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')
        object_inputs = np.load(
            osp.join(self.data_root, self.object_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = np.concatenate((target[:self.work_memory_length:self.work_memory_sample_rate],
                                 target[self.work_memory_length::]),
                                 axis=0)
        if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
            verb_target = np.concatenate((verb_target[:self.work_memory_length:self.work_memory_sample_rate],
                                          verb_target[self.work_memory_length::]),
                                          axis=0)
            noun_target = np.concatenate((noun_target[:self.work_memory_length:self.work_memory_sample_rate],
                                          noun_target[self.work_memory_length::]),
                                          axis=0)

        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]
        work_object_inputs = object_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            # long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]
            long_object_inputs = object_inputs[long_indices]
            # Clip-level mixup
            if self.training and self.clip_mixup_rate > 0:
                assert segments_before is not None
                segments_before = segments_before[segments_before['end_f'] / 30 * self.cfg.DATA.FPS > long_start]
                if self.clip_mixup_sample == 'uniform':
                    prob = None
                elif self.clip_mixup_sample == 'by_length':
                    prob = (segments_before['len_f']).tolist()
                    prob = prob / np.sum(prob)
                else:
                    raise ValueError
                num_clip = int(len(segments_before) * self.clip_mixup_rate)
                segments_to_mixup = segments_before.sample(num_clip, replace=False,
                                                           weights=prob)
                if self.cfg.DATA.TK_ONLY:
                    segments_to_mixup = segments_to_mixup[segments_to_mixup['action'].isin(self.cfg.DATA.TK_IDXS)]
                for old_segment in segments_to_mixup.iterrows():
                    old_start_tick = int(old_segment[1]['start_f'] / 30 * self.cfg.DATA.FPS)
                    old_end_tick = int(np.ceil(old_segment[1]['end_f'] / 30 * self.cfg.DATA.FPS))
                    old_action = old_segment[1]['action']
                    old_vid = old_segment[1]['video']
                    assert old_vid == session
                    segment_same_class = self.segment_by_cls[old_action]
                    segment_same_class = segment_same_class[segment_same_class['video'] != old_vid]
                    valid_segments = segment_same_class[segment_same_class['len_f'] > old_segment[1]['len_f']]
                    if len(valid_segments) == 0:
                        continue
                    else:
                        sample_segment = valid_segments.sample(1)
                        new_vid = sample_segment['video'].values[0]
                        new_start_tick = int(sample_segment['start_f'] / 30 * self.cfg.DATA.FPS)
                        new_end_tick = int(sample_segment['end_f'] / 30 * self.cfg.DATA.FPS)

                        new_visual_inputs = np.load(
                            osp.join(self.data_root, self.visual_feature, new_vid + '.npy'), mmap_mode='r')
                        new_motion_inputs = np.load(
                            osp.join(self.data_root, self.motion_feature, new_vid + '.npy'), mmap_mode='r')
                        new_object_inputs = np.load(
                            osp.join(self.data_root, self.object_feature, new_vid + '.npy'), mmap_mode='r')
                        
                        sel_indices = np.where((long_indices >= old_start_tick) & (long_indices <= old_end_tick))

                        shift = np.random.randint(new_end_tick - new_start_tick - len(sel_indices) + 1)
                        long_visual_inputs[sel_indices] = new_visual_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)]
                        long_motion_inputs[sel_indices] = new_motion_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)]
                        long_object_inputs[sel_indices] = new_object_inputs[new_start_tick + shift:new_start_tick + shift + len(sel_indices)]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            long_object_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if (long_visual_inputs is not None and
            long_motion_inputs is not None and
            long_object_inputs is not None):
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
            fusion_object_inputs = np.concatenate((long_object_inputs, work_object_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs
            fusion_object_inputs = work_object_inputs


        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        fusion_object_inputs = torch.as_tensor(fusion_object_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
            verb_target = torch.as_tensor(verb_target.astype(np.float32))
            noun_target = torch.as_tensor(noun_target.astype(np.float32))
            target = (target, verb_target, noun_target)

        assert fusion_visual_inputs.ndim == fusion_motion_inputs.ndim
        if fusion_visual_inputs.ndim == 4:
            # (L,C,H,W) -> (L,H,W,C)
            fusion_visual_inputs = fusion_visual_inputs.permute(0, 2, 3, 1)
            fusion_motion_inputs = fusion_motion_inputs.permute(0, 2, 3, 1)
            fusion_visual_inputs = fusion_visual_inputs.mean((1, 2))
            fusion_motion_inputs = fusion_motion_inputs.mean((1, 2))

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs,
                    fusion_object_inputs, memory_key_padding_mask, target)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs,
                    fusion_object_inputs, fusion_object_inputs, target)

    def __len__(self):
        return len(self.inputs)


@registry.register('LSTRBatchInferenceENIGMA')
@registry.register('LSTRBatchInferenceTHUMOS')
@registry.register('LSTRBatchInferenceTVSeries')
@registry.register('LSTRBatchInferenceEK55')
@registry.register('LSTRBatchInferenceEK100')
class LSTRBatchInferenceDataLayer(data.Dataset):

    def __init__(self, cfg, phase='test'):
        self.cfg = cfg
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.object_feature = cfg.INPUT.OBJECT_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

        assert phase == 'test', 'phase must be `test` for batch inference, got {}'

        self.inputs = []
        if 'ENIGMA' in self.data_root:
            for video_id in ["126", "107", "49", "66", "69", "104", "111", "145", "156"]:
                try:
                    self.sessions.remove(video_id)
                except:
                    pass
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, "TARGET", self.target_perframe, session + '.npy'))
            if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
                verb_target = np.load(osp.join(self.data_root, "TARGET", self.target_perframe.replace('target', 'verb'), session + '.npy'))
                noun_target = np.load(osp.join(self.data_root, "TARGET", self.target_perframe.replace('target', 'noun'), session + '.npy'))
            else:
                verb_target = target
                noun_target = target
            for work_start, work_end in zip(
                range(0, target.shape[0] + 1),
                range(self.work_memory_length, target.shape[0] + 1)):
                self.inputs.append([
                    session, work_start, work_end,
                    target[work_start: work_end],
                    verb_target[work_start: work_end],
                    noun_target[work_start: work_end],
                    target.shape[0]
                ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target, verb_target, noun_target, num_frames = self.inputs[index]
        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')
        object_inputs = np.load(
            osp.join(self.data_root, self.object_feature, session + '.npy'), mmap_mode='r')

        # Get target
        target = target[::self.work_memory_sample_rate]
        verb_target = verb_target[::self.work_memory_sample_rate]
        noun_target = noun_target[::self.work_memory_sample_rate]
        
        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]
        work_object_inputs = object_inputs[work_indices]

        # Get long memory
        if self.long_memory_num_samples > 0:
            # long_start, long_end = max(0, work_start - self.long_memory_length), work_start - 1
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]
            long_object_inputs = object_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            long_object_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
            fusion_object_inputs = np.concatenate((long_object_inputs, work_object_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs
            fusion_object_inputs = work_object_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        fusion_object_inputs = torch.as_tensor(fusion_object_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        if self.cfg.MODEL.LSTR.V_N_CLASSIFIER:
            verb_target = torch.as_tensor(verb_target.astype(np.float32))
            noun_target = torch.as_tensor(noun_target.astype(np.float32))
            target = (target, verb_target, noun_target)

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs,
                    fusion_object_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs,
                    fusion_object_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)

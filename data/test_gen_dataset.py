import random

import torch
from torch.utils.data.dataset import Dataset
from data.keyframe_dataset import ActionDataset
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
from configs import cfg
from scipy.spatial.transform import Rotation as sRot
from utils import data_utils
from utils.bvh import Bvh
from utils.debug_util import run_cmd
from utils.skeleton_torch import SkeletonMotionTorch
from utils.skeleton import SkeletonMotion
from utils import heuristic


class TrajHintDataset(ActionDataset):
    def __init__(self, base_cfg, give_hint=False, split='all'):
        self.traj = []
        self.g_positions = []
        self.first_frames = []
        self.keyframes = []
        self.give_hint = give_hint
        self.max_kf_num = 0
        super().__init__(**base_cfg, split=split)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        traj = data['g_positions'][:, 0]
        g_pos = data['g_positions']
        if g_pos.shape[0] < 219:
            return 0
        self.traj.append(traj)
        self.g_positions.append(g_pos)
        self.first_frames.append(data_utils.preprocess_frame({
            'g_position': g_pos[9],
            'rotations': data['rotations'][9],
            'velocity': traj[9] - traj[8],
        }))
        self.max_kf_num = max(self.max_kf_num, data['keyframes'].shape[0])
        self.keyframes.append(data['keyframes'])
        return 1

    def __getitem__(self, item):
        hint_mask = np.zeros_like(self.g_positions[item]).astype(np.float32)
        kf = np.ones((self.max_kf_num), dtype=int)
        kf[:len(self.keyframes[item])] = self.keyframes[item]

        return  self.traj[item].astype(np.float32), \
                self.g_positions[item].astype(np.float32), \
                hint_mask, \
                np.array(self.get_action(item), dtype=np.float32)[None], \
                self.first_frames[item], \
                kf


class MotionExtractorDataset(ActionDataset):
    def __init__(self, base_cfg, split='train', data_repr='gpos', augment=False):
        self.max_frames = 0
        self.data_repr = data_repr
        self.augment = augment
        self.g_positions = []
        self.keyframes = []
        self.poses = []
        self.l_positions = []
        self.max_frames = 0
        super().__init__(**base_cfg, split = split)

    def _load_data(self, path):
        data = np.load(path, allow_pickle=True)
        if data['g_positions'].shape[0] < 219:
            return 0
        self.g_positions.append(data['g_positions'])
        self.keyframes.append(data['keyframes'])
        if data['keyframes'].shape[0] > self.max_frames:
            self.max_frames = data['keyframes'].shape[0]
        self.poses.append(data['rotations'])
        self.l_positions.append(data['l_positions'])
        return 1

    def __getitem__(self, item):
        keyframes = self.keyframes[item].tolist()
        if self.augment:
            sample = np.random.choice(np.arange(keyframes[0], keyframes[-1] + 1), len(keyframes) - 2, replace=False)
            keyframes = [keyframes[0], keyframes[-1]] + sample.tolist()
        keyframes.sort()
        frame_list = []
        gt = {
            'rotations': self.poses[item].astype(np.float32).copy(),
            'l_positions': self.l_positions[item].astype(np.float32).copy(),
            'g_positions': self.g_positions[item].astype(np.float32).copy(),
        }
        first_frame = data_utils.preprocess_frame({
            'g_position': self.g_positions[item][keyframes[0]],
            'rotations': self.poses[item][keyframes[0]],
            'velocity': self.g_positions[item][keyframes[0]][0] - self.g_positions[item][keyframes[0] - 1][0],
        })
        rot_offset = first_frame['rot_offset']
        trans_offset = first_frame['root_trans'][0]
        states = []
        for i, f in enumerate(keyframes):
            rotation = gt['rotations'][f]
            position = gt['g_positions'][f][0]
            g_position = gt['g_positions'][f]
            g_position[:, ::2] -= trans_offset[::2]
            root_mat = data_utils.matrix6D_to_9D(rotation[0])
            root_mat = np.matmul(rot_offset, root_mat)
            rotation[0] = data_utils.matrix9D_to_6D(root_mat)
            g_position = np.matmul(rot_offset, g_position[..., None])[..., 0]
            position = g_position[0]
            interval = f - keyframes[i - 1] if i > 0 else 0
            interval = np.array(interval).astype(np.float32)[None]
            if self.data_repr == 'gpos':
                state = np.concatenate([g_position.reshape(-1)])[None]
            elif self.data_repr == 'rot6d_pos':
                state = np.concatenate([rotation.reshape(-1), position])[None]
            states.append(state)
        if len(keyframes) < self.max_frames:
            states += [np.zeros_like(states[0]) for _ in range(self.max_frames - len(keyframes))]
        states = np.concatenate(states, axis=0)
        return states, np.array(self.get_action(item), dtype=np.float32)[None], len(keyframes)

    def calc_stats(self):
        aug = self.augment
        self.augment = False
        xxx = lambda states, action, length: states[:length]
        all_states = [ xxx(*self[i]) for i in range(len(self))]
        all_states = np.concatenate(all_states, axis=0)
        self.mean = all_states.mean(axis=0)[None]
        self.std = all_states.std(axis=0)[None]
        # self.std[np.where(self.std < 1e-3)] = 1e-3
        self.augment = aug
        return self.mean, self.std


class MotionDataset(ActionDataset):
    def __init__(self, base_cfg, split='train', data_repr='gpos', augment=False):
        self.data_repr = data_repr
        self.augment = augment
        self.tot_length = 0
        self.window = 219
        self.offset = 70
        self.parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
        self.seq_idx = []
        self.g_positions = []
        self.poses = []
        self.max_frames = 0
        super().__init__(**base_cfg, split = split)
        if isinstance(self.seq_idx, list):
            self.seq_idx = np.concatenate(self.seq_idx, axis=0)
            self.g_positions = np.concatenate(self.g_positions, axis=0).astype(np.float32)
            self.poses = np.concatenate(self.poses, axis=0).astype(np.float32)

    def _load_bvh_data(self, path, mirror = False):
        """

        """

        with open(path, 'r') as f:
            bvh = Bvh(f.read())

        root_trans = bvh.root_translation[:, None, :]
        n_frames = root_trans.shape[0]
        offset = [np.array(bvh.joint_offset(joint))[None, None, :].repeat(n_frames, axis=0) for joint in bvh.get_joints_names()[1:]]
        l_positions = np.concatenate([root_trans] + offset, axis=1)
        euler_rotations = bvh.joint_rotations.reshape(-1, 22, 3)[..., [2,1,0]]
        matrix_rotations = sRot.from_euler('xyz', euler_rotations.reshape(-1, 3), degrees=True).as_matrix().reshape(-1, 22, 3, 3)
        rotations = data_utils.matrix9D_to_6D(matrix_rotations)

        if mirror:
            rotations, g_positions = data_utils.swap_left_right(rotations, l_positions)
        else:
            _, g_positions = data_utils.fk(matrix_rotations, l_positions, self.parents)

        if self.augment:
            st_idx = np.arange(0, n_frames + 1 - self.window - self.offset, self.offset)
            ed_idx = np.arange(self.window + self.offset, n_frames + 1, self.offset)
            seq_idx = np.stack((st_idx, ed_idx), axis=-1) + self.tot_length # (len, 2)
            self.max_frames = self.window + self.offset if self.window + self.offset > self.max_frames else self.max_frames
        else:
            st_idx = np.arange(0, n_frames + 1 - self.window, self.offset)
            ed_idx = np.arange(self.window, n_frames + 1, self.offset)
            seq_idx = np.stack((st_idx, ed_idx), axis=-1) + self.tot_length # (len, 2)
            self.max_frames = self.window if self.window > self.max_frames else self.max_frames

        self.tot_length += n_frames

        self.seq_idx.append(seq_idx)

        self.g_positions.append(g_positions)
        self.poses.append(rotations)

        return seq_idx.shape[0]

    def _load_mib_data(self, path, mirror = False):
        """

        Args:
            path (Path): path to the npz file
            g_position (np.ndarray): (batch_size, n_frames, 22, 3)
            rotation (np.ndarray): (batch_size, n_frames, 22, 6)
        """
        data = np.load(path, allow_pickle=True)
        g_positions = data['g_positions'] 
        rotations = data['rotations']
        action = data['action']
        # keyframes = data['keyframes']
        batch_size = g_positions.shape[0]
        length = g_positions.shape[1]
        self.max_frames = length if length > self.max_frames else self.max_frames

        st = np.arange(0, batch_size) * length
        ed = np.arange(1, batch_size+1) * length
        seq_idx = np.stack((st, ed), axis=-1)

        self.seq_idx = seq_idx
        self.g_positions = g_positions.reshape(-1, 22, 3)
        self.poses = rotations.reshape(-1, 22, 6)
        self._clip_label = [self.action_to_label[action[i][0]] for i in range(batch_size)]
        return 0

    def _load_npz_data(self, path, mirror = False):
        """

        Args:
            path (Path): path to the npz file
            g_position (np.ndarray): (n_frames, 22, 3)
            rotation (np.ndarray): (n_frames, 22, 6)
        """
        data = np.load(path, allow_pickle=True)
        length = data['g_positions'].shape[0]
        if length < 219:
            return 0
        g_position = data['g_positions']
        rotation = data['rotations']

        seq_idx = np.array([[0, length]]) + self.tot_length # (1, 2)
        self.max_frames = length if length > self.max_frames else self.max_frames
        self.tot_length += length

        self.seq_idx.append(seq_idx)

        self.g_positions.append(g_position)
        self.poses.append(rotation)
        return seq_idx.shape[0]


    def _load_data(self, path, mirror = False):
        if 'npz' in path.suffix:
            if 'eval_motion' in path.stem:
                return self._load_mib_data(path, mirror)
            else:
                return self._load_npz_data(path, mirror)
        elif 'bvh' in path.suffix:
            return self._load_bvh_data(path, mirror)
        else:
            tqdm.write(f'Unknown file type: {path.name}')

    def __getitem__(self, item):
        """

        Returns:
            states (): (max_frames, state_dim)
        """
        start_idx, end_idx = self.seq_idx[item]
        if self.augment and end_idx - start_idx > self.window:
            start_idx = np.random.randint(start_idx, end_idx - self.window)
            end_idx = start_idx + self.window

        interval = end_idx - start_idx
        g_positions = self.g_positions[start_idx : end_idx].copy()
        rotations = self.poses[start_idx : end_idx].copy()

        # pos 移到中心, 并转为正面
        trans_offset = g_positions[0, 0, ::2]
        g_positions[..., ::2] -= trans_offset

        g_positions, rotations = data_utils.rotate_start_to_v2(g_positions, rotations, frame=0)

        if self.data_repr == 'gpos':
            states = g_positions
            states = states.reshape(interval, -1)
        elif self.data_repr == 'rot6d_pos':
            states = rotations
            states = states.reshape(interval, -1)
            states = np.concatenate([g_positions[:, 0], states], axis=1).astype(np.float32)
        elif self.data_repr == 'rot6d':
            states = rotations
            states = states.reshape(interval, -1)

        zero_padding = np.zeros((self.max_frames - interval, states.shape[1]), dtype=np.float32)
        states = np.concatenate([states, zero_padding], axis=0)

        return states, np.array(self.get_action(item), dtype=np.float32)[None], interval

    def calc_stats(self):
        aug = self.augment
        self.augment = False
        xxx = lambda states, action, length: states[:length]
        all_states = [ xxx(*self[i]) for i in range(len(self))]
        all_states = np.concatenate(all_states, axis=0)
        self.mean = all_states.mean(axis=0)[None]
        self.std = all_states.std(axis=0)[None]
        # self.std[np.where(self.std < 1e-3)] = 1e-3
        self.augment = aug
        return self.mean, self.std

class EvaluateWrapper(Dataset):
    def __init__(self, l_position = None):
        super().__init__()
        self.l_position = l_position
        self.max_frames = 0
        self.states = []
        self.rotations = []
        self.g_positions = []
        self.keyframes = []
        self.actions = []
        self.traj = []
        self.hint = []
        self.hint_mask = []
        self.metric_name = None

    def append_data(self, rotations, root_trans, g_positions, keyframes, action ,traj, hint, hint_mask):
        intervals = [keyframes[i] - keyframes[i - 1] for i in range(1, len(keyframes))]
        intervals.append(0)
        intervals.sort()
        intervals = np.array(intervals).astype(np.float32)[:, None]
        self.max_frames = len(keyframes) if len(keyframes) > self.max_frames else self.max_frames
        self.rotations.append(rotations)
        self.g_positions.append(g_positions)
        self.keyframes.append(keyframes)
        self.actions.append(action)
        self.traj.append(traj[None])
        self.hint.append(hint[None])
        self.hint_mask.append(hint_mask[None])

    def on_append_data_end(self):
        self.traj = np.concatenate(self.traj, axis=0)
        self.hint = np.concatenate(self.hint, axis=0)
        self.hint_mask = np.concatenate(self.hint_mask, axis=0)

        if not cfg.get('run_mib', False):
            return

        full_rotations_list = np.zeros((len(self), 219, 22, 6), dtype=np.float32)
        full_positions_list = np.zeros((len(self), 219, 22, 3), dtype=np.float32)
        full_g_positions_list = np.zeros((len(self), 219, 22, 3), dtype=np.float32)
        full_positions_list[:, :] = self.l_position
        full_positions_list[:, :, 0] = self.traj

        keyframes_list = [ np.concatenate([ks, np.zeros((self.max_frames - ks.shape[0]), dtype=np.int32)], axis=-1).astype(np.int32)[None] for ks in self.keyframes]
        keyframes_list = np.concatenate(keyframes_list, axis=0)
        action_list = np.concatenate(self.actions, axis=0)[:, None]

        for i, ks in enumerate(self.keyframes):
            full_rotations_list[i, ks] = self.rotations[i]
            full_positions_list[i, ks, 0] = self.g_positions[i][:, 0]
            full_g_positions_list[i, ks] = self.g_positions[i][:]
            full_rotations_list[i, :ks[0]] = self.rotations[i][0]
            full_positions_list[i, :ks[0], 0] = self.g_positions[i][0, 0]
            full_g_positions_list[i, :ks[0]] = self.g_positions[i][0]

        mib_eval_path = os.path.join(cfg.output_dir, f"eval_keyframe.npz")
        np.savez(mib_eval_path, rotations=full_rotations_list, l_positions=full_positions_list, keyframes=keyframes_list,
                 action=action_list, g_positions=full_g_positions_list, gt_hint=self.hint)

        cmd = f"python3 {os.path.join(cfg.mib_path, 'eval.py')}"  # Requires external MIB model (not yet open-sourced)
        cmd += f" --data_path {mib_eval_path}"
        cmd += f" --output_path {cfg.output_dir}"
        cmd += f" --output_name {cfg.eval_motion_name}"
        # cmd += f" --eval_edit True"
        cmd += f" --gpus {' '.join([str(gpu) for gpu in cfg.gpus])}"
        run_cmd(cmd)

        eval_path = os.path.join(cfg.output_dir, cfg.eval_motion_name)
        self._load_mib_data(eval_path)

    def _load_mib_data(self, path, mirror = False):
        """

        Args:
            path (Path): path to the npz file
            g_position (np.ndarray): (batch_size, n_frames, 22, 3)
            rotation (np.ndarray): (batch_size, n_frames, 22, 6)
        """
        data = np.load(path, allow_pickle=True)
        g_positions = data['g_positions'] 
        rotations = data['rotations']
        action = data['action']
        if 'traj' in data:
            self.traj = data['traj']
        # keyframes = data['keyframes']
        batch_size = g_positions.shape[0]
        length = g_positions.shape[1]
        self.max_frames = length if length > self.max_frames else self.max_frames

        # st = np.arange(0, batch_size) * length
        # ed = np.arange(1, batch_size+1) * length
        # seq_idx = np.stack((st, ed), axis=-1)

        # self.keyframes = seq_idx
        self.g_positions = g_positions.reshape(-1, length, 22, 3)
        self.rotations = rotations.reshape(-1, length, 22, 6)
        self.actions = action
        # self._clip_label = [self.action_to_label[action[i][0]] for i in range(batch_size)]
        return 0

    def clear(self):
        self.max_frames = 0
        self.states.clear()
        self.rotations.clear()
        self.g_positions.clear()
        self.keyframes.clear()
        self.actions.clear()
        self.traj.clear()
        self.hint_mask.clear()
        self.hint.clear()

    def __getitem__(self, idx):
        if cfg.get('run_mib', False):
            # start_idx, end_idx = self.keyframes[idx]
            # interval = end_idx - start_idx
            length = 219
            g_positions = self.g_positions[idx].copy()
            rotations = self.rotations[idx].copy()
            traj = self.traj[idx].copy()
            hint = self.hint[idx].copy()
            hint_mask = self.hint_mask[idx].copy()
        else:
            keyframes = self.keyframes[idx]
            length = len(keyframes)
            g_positions = self.g_positions[idx].copy()
            rotations = self.rotations[idx].copy()
            traj = self.traj[idx][keyframes].copy()
            hint = self.hint[idx][keyframes].copy()

        # pos 移到中心, 并转为正面
        trans_offset = g_positions[0, 0, ::2].copy() #HACK 0 并非起始帧, 这里对是因为起始帧 padding 到了 0, 如果修改 padding 这里也要修改
        g_positions[..., ::2] -= trans_offset
        traj[..., ::2] -= trans_offset
        hint[..., ::2] -= trans_offset

        g_positions, rotations, offset_matrix = data_utils.rotate_start_to_v2(g_positions, rotations, frame=0, return_offset=True)
        traj = np.matmul(offset_matrix, traj[..., None])[..., 0]
        hint = np.matmul(offset_matrix, hint[..., None])[..., 0]

        # if self.data_repr == 'gpos':
        states = np.concatenate([g_positions.reshape(length, -1)])
        # elif self.data_repr == 'rot6d_pos':
        #     states = rotations
        #     states = states.reshape(length, -1)
        #     states = np.concatenate([g_positions[:, 0], states], axis=1, dtype=np.float32)

        if self.metric_name == 'FID' or self.metric_name == 'MiB':
            zero_padding = np.zeros((self.max_frames - length, states.shape[1]), dtype=np.float32)
            states = np.concatenate([states, zero_padding], axis=0)
            return states.astype(np.float32), self.actions[idx].astype(np.float32), length
        elif self.metric_name == 'penetration' or self.metric_name == 'foot skate':
            if length < self.max_frames:
                padding = [np.zeros_like(g_positions[0][None]) for _ in range(self.max_frames - length)]
                g_positions = np.concatenate([g_positions] + padding, axis=0)
            return g_positions.astype(np.float32), length
        elif self.metric_name == 'traj':
            return traj, g_positions[:, 0].astype(np.float32)
        elif self.metric_name == 'pos':
            return hint, g_positions.astype(np.float32), hint_mask


    def __len__(self):
        return len(self.g_positions)

    def calc_stats(self):
        for i in range(len(self)):
            states, _, length = self[i]
            if i == 0:
                all_states = states[:length]
            else:
                all_states = np.concatenate([all_states, states[:length]], axis=0)
        self.mean = all_states.mean(axis=0)[None]
        self.std = all_states.std(axis=0)[None]
        # self.std[np.where(self.std < 1e-3)] = 1e-3
        return self.mean, self.std


class EditEvalDataset(Dataset):
    def __init__(self):
        self.g_positions = []
        self.gt_hint = []
        self.hint_mask = []
        self.rotations = []
        self.keyframes = []
        self.actions = []
        self.l_positions = []
        self.max_frames = 0

    def load_mib_data(self, path):
        parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
        data = np.load(path, allow_pickle=True)
        skeleton = SkeletonMotion(None, parents, data['l_positions'])
        g_positions = data['g_positions']
        l_positions = data['l_positions']
        g_positions[:, 0] = l_positions[:, 0]
        rotations = data['rotations']
        action = np.array([8])
        hint = data['hint']
        hint_mask = data['hint_mask']
        skeleton.fk_apply_pose(l_positions[:, 0], rotations)

        self.keyframes.append(data['keyframes'])
        self.hint_mask.append(hint_mask)
        self.g_positions.append(skeleton.joints_global_positions)
        self.rotations.append(rotations.reshape(-1, 22, 6))
        self.actions.append(action)
        self.gt_hint.append(hint)

    def __len__(self):
        return len(self.g_positions)

    def __getitem__(self, item):
        keyframes = self.keyframes[item]
        keyframes.sort()
        frame_list = []
        gt_hint = self.gt_hint[item]
        hint_mask = self.hint_mask[item]
        l_frame = None
        for i in range(len(keyframes)):
            f = keyframes[i]
            interval = f - keyframes[i - 1] if i > 0 else f
            frame = data_utils.preprocess_frame({
                'g_position': self.g_positions[item][f],
                'rotations': self.rotations[item][f],
                'velocity': self.g_positions[item][f, 0, :] - self.g_positions[item][f - 1, 0, :],
            })
            frame['interval'] = np.array(interval).astype(np.float32)[None]
            frame['hint'] = gt_hint[f].astype(np.float32)
            frame['hint_mask'] = hint_mask[f]
            frame['index'] = f
            frame_list.append(frame)
        return gt_hint, hint_mask, frame_list, self.actions[item].astype(np.float32)

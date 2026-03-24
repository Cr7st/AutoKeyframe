from abc import abstractmethod
import random
from torch.utils.data.dataset import Dataset
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from utils import data_utils
from utils.bvh import Bvh
from omegaconf.listconfig import ListConfig
# from memory_profiler import profile


class ActionDataset(Dataset):
    def __init__(self, dataset_dir, action_dict, action_repr='scalar', split='all', mirror=False):
        super().__init__()

        if isinstance(dataset_dir, list) or isinstance(dataset_dir, ListConfig):
            self.data_paths = []
            for data_path in dataset_dir:
                if os.path.isdir(data_path):
                    continue
                self.data_paths.append(Path(data_path))
        elif os.path.isdir(dataset_dir):
            data_paths = list(Path(dataset_dir).rglob("*.npz"))
            data_paths += list(Path(dataset_dir).rglob("*.bvh"))
            if split == 'test':
                data_paths = [path for path in data_paths if 'subject5' in path.__str__()]
            elif split == 'train':
                data_paths = [path for path in data_paths if 'subject5' not in path.__str__()]
            elif split == 'all':
                data_paths = data_paths

            self.data_paths = data_paths
        else:
            self.data_paths = [Path(dataset_dir)]

        self.action_to_label = {action: label for action, label in action_dict.items()}
        self.label_to_action = {label: action for action, label in action_dict.items()}
        self.action_num = len(action_dict)
        self._clip_label = []
        self.action_repr = action_repr

        for path in tqdm(self.data_paths):
            label = action_adjust_v1(path.stem)
            if label is None or label not in self.label_to_action.keys():
                tqdm.write(f'{path.stem} is not a valid action category, skipping...')
                continue
            n_loaded = self._load_data(path)
            if mirror:
                n_loaded += self._load_data(path, mirror=True)
            self._clip_label.extend([label] * n_loaded)

    def __len__(self):
        return len(self._clip_label)

    def get_action_label(self, idx):
        return self._clip_label[idx]

    def get_action(self, idx):
        if self.action_repr == 'scalar':
            return self.label_to_action[self._clip_label[idx]]
        elif self.action_repr == 'one_hot':
            return self.make_one_hot(self.label_to_action[self._clip_label[idx]])

    def make_one_hot(self, action):
        one_hot = np.zeros([self.action_num], dtype=np.float32)
        one_hot[action - 1] = 1
        return one_hot

    @staticmethod
    def _get_action_label_from_filename(filename):
        label = filename.split('_')[0][:-1]
        return label

    @abstractmethod
    def _load_data(self, path):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class KeyframeDataset(ActionDataset):
    def __init__(self, base_cfg ,use_prev_rot=True, use_global_rot=False, split='all'):
        self.intervals_from_last_frame = []
        self.current_frames = []
        self.last_frames = []
        self.use_prev_rot = use_prev_rot
        self.use_global_rot = use_global_rot
        super().__init__(**base_cfg, split=split)
        self.mean = None
        self.std = None

    def __len__(self):
        return len(self.current_frames)

    def __getitem__(self, idx):
        return self.current_frames[idx], self.last_frames[idx], \
                np.array(self.intervals_from_last_frame[idx], dtype=np.float32)[None], \
                np.array(self.get_action(idx), dtype=np.float32)[None]

    def _load_data(self, path, mirror=False):
        data = np.load(path, allow_pickle=True)
        keyframes = data['keyframes']
        l_frame = None
        for i, f in enumerate(keyframes):
            c_frame = data_utils.preprocess_frame({
                'g_position': data['g_positions'][f],
                'rotations': data['rotations'][f],
                'velocity': data['g_positions'][f, 0, :] - data['g_positions'][f - 1, 0, :],
            }, l_frame, 
            meta={
                'l_position': data['l_positions'][f].astype(np.float32),
            },
            use_prev_rot=self.use_prev_rot, use_global_rot = self.use_global_rot,mirror=mirror)

            if l_frame is not None:
                self.intervals_from_last_frame.append(f - keyframes[i - 1])
                self.last_frames.append(l_frame)
                self.current_frames.append(c_frame)

            l_frame = c_frame.copy()
            l_frame['rotations'] = l_frame['original_rotations']
        return len(keyframes) - 1

    def calc_stats(self, data_lambda, path = None):
        state_list = []
        for i in range(len(self)):
            current_frame = self.current_frames[i]
            state = data_lambda(current_frame)
            state_list.append(state[None, ...])
        states = np.concatenate(state_list, axis=0)
        self.mean = states.mean(axis=0)
        self.std = states.std(axis=0)
        if path is not None:
            self.save_stats(path)
        return self.mean, self.std
    
    def save_stats(self, path):
        path = Path(path)
        mean_path = path / 'mean.npy'
        std_path = path / 'std.npy'
        np.save(mean_path, self.mean)
        np.save(std_path, self.std)


class PairFrameDataset(KeyframeDataset):
    def __init__(self, base_cfg, frame_range = [10, 45, 5], num = 120, split='all', use_prev_rot=True, use_global_rot=False, real_random = False):
        self.frame_range = frame_range # [min, max) step
        self.num = num
        self.real_random = real_random
        self.target_frame_idx = []
        self.parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
        super().__init__(base_cfg = base_cfg, split = split, use_prev_rot=use_prev_rot, use_global_rot=use_global_rot)

    def __len__(self):
        return len(self.target_frame_idx)

    def __getitem__(self, idx):
        if not self.real_random:
            target_idx = self.target_frame_idx[idx]
        else:
            target_idx = random.randint(self.target_frame_idx[idx][0], self.target_frame_idx[idx][1]-1)

        current_frame = self.current_frames[target_idx].copy()

        current_idx = current_frame['index']

        prev_idx = current_idx - random.randint(self.frame_range[0], self.frame_range[1])
        if prev_idx < 0:
            prev_idx = 0
        prev_idx += target_idx - current_idx
        prev_frame = self.current_frames[prev_idx]
        if current_idx == 9: # first keyframe
            prev_frame['rotations'] = np.zeros_like(prev_frame['rotations'])
            prev_frame['rot_offset'] = np.eye(3)[None]

        delta_res = data_utils.preprocess_relative_info(current_frame, prev_frame, use_prev_rot=self.use_prev_rot, use_global_rot=self.use_global_rot)
        current_frame.update(delta_res)

        interval = current_frame['index'] - prev_frame['index']

        return current_frame, prev_frame, np.array(interval, dtype=np.float32)[None], np.array(self.get_action(idx), dtype=np.float32)[None]

    def _load_npz_data(self, path):
        data = np.load(path, allow_pickle=True)
        rotations = data['rotations']
        g_positions = data['g_positions']
        keyframes = data['keyframes']
        l_positions = data['l_positions']
        return rotations, g_positions, keyframes, l_positions
    
    def _load_bvh_data(self, path):
        with open(path, 'r') as f:
            bvh = Bvh(f.read())

        root_trans = bvh.root_translation[:, None, :]
        n_frames = root_trans.shape[0]
        offset = [np.array(bvh.joint_offset(joint))[None, None, :].repeat(n_frames, axis=0) for joint in bvh.get_joints_names()[1:]]
        l_positions = np.concatenate([root_trans] + offset, axis=1)
        euler_rotations = bvh.joint_rotations.reshape(-1, 22, 3)[..., [2,1,0]]
        matrix_rotations = sRot.from_euler('xyz', euler_rotations.reshape(-1, 3), degrees=True).as_matrix().reshape(-1, 22, 3, 3)
        rotations = data_utils.matrix9D_to_6D(matrix_rotations)

        _, g_positions = data_utils.fk(matrix_rotations, l_positions, self.parents)

        return rotations, g_positions, l_positions

    def _load_data(self, path, mirror = False):
        if 'npz' in path.suffix:
            rotations, g_positions, keyframes, l_positions = self._load_npz_data(path)
        elif 'bvh' in path.suffix:
            rotations, g_positions, l_positions =  self._load_bvh_data(path)
        else:
            tqdm.write(f'Unknown file type: {path.name}')

        length = g_positions.shape[0]

        new_frames = [
            data_utils.preprocess_frame({
                'g_position': g_positions[i],
                'rotations': rotations[i],
                'velocity': g_positions[i, 0] - g_positions[i - 1, 0],
            }, meta={
                'index': i,
                'l_position': l_positions[i]
            }, mirror=mirror) for i in range(length)
        ]

        new_start_idx = len(self.current_frames)
        new_finish_idx = new_start_idx + len(new_frames)  # [new_start_idx, new_finish_idx)

        if not self.real_random:
            new_target_frame_idx = [
                new_start_idx + i
                # for i in range(30, length)
                for i in keyframes
            ]
        else:
            new_target_frame_idx = [
                [new_start_idx, new_finish_idx] 
            ] * 300

        self.current_frames.extend(new_frames)
        self.target_frame_idx.extend(new_target_frame_idx)
        return len(new_target_frame_idx)


lafan1_action_dict = {
    1: 'aiming',
    2: 'dance',
    3: 'fallAndGetUp',
    4: 'fight',
    5: 'ground',
    6: 'jumps',
    7: 'multipleActions',
    8: 'obstacles',
    9: 'push',
    10: 'pushAndFall',
    11: 'pushAndStumble',
    12: 'run',
    13: 'sprint',
    14: 'walk',
    15: 'fightAndSports'
}

lafan1_action_dict_12 = {
    1: 'aiming',
    2: 'dance',
    3: 'fallAndGetUp',
    4: 'fight',
    5: 'ground',
    6: 'jumps',
    7: 'multipleActions',
    8: 'obstacles',
    9: 'push',
    10: 'run',
    11: 'sprint',
    12: 'walk',
}

lafan1_action_dict_9 = {
    1: 'aiming',
    2: 'dance',
    3: 'fallAndGetUp',
    4: 'fight',
    5: 'ground',
    6: 'jumps',
    8: 'obstacles',
    12: 'run',
    14: 'walk',
}

def action_adjust_v1(file_name):
    label = file_name.split('_')[0][:-1]
    if 'push' in label or 'multipleActions' in label:
        return None
    elif 'fight' in label:
        label = 'fight'
    elif 'sprint' in label:
        label = 'run'
    elif label in 'eval_mib':
        label = 'walk'

    return label

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = KeyframeDataset('./datasets/lafan1_keyframes/', lafan1_action_dict)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(batch)

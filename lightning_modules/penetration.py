from .FID import Evaluator
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch

class PenetrationEvaluator(Evaluator):
    def __init__(self, threshold=-3):
        super().__init__()
        self.threshold = threshold
        self.name = 'penetration'
        return

    def evaluate(self, dataloader):
        dataloader.dataset.metric_name = self.name
        total_entry = 0
        total_metric = 0
        for idx, data in enumerate(dataloader):
            g_positions, lengths = data[0], data[1]
            y_value = g_positions[..., 1]
            mask = y_value < self.threshold
            y_value[~mask] = 0

            total_entry += g_positions.shape[0]
            total_metric += (y_value.sum(dim=(1, 2)) / lengths).sum()

        result = -total_metric / total_entry
        return {
            'penetration': result.item()
        }

class FootSkateEvaluator(Evaluator):
    def __init__(self, threshold_height=2.5, threshold_vel=30):
        super().__init__()
        self.threshold_height = threshold_height
        self.threshold_vel = threshold_vel
        self.name = 'foot skate'
        return

    def evaluate(self, dataloader):
        dataloader.dataset.metric_name = self.name
        skating_ratio_sum = 0
        all_size = 0
        for idx, data in enumerate(dataloader):
            g_positions = data[0]

            batch_size = g_positions.shape[0]
            all_size += batch_size
            # 8 left, 4 right foot. XZ plane, y up
            # motions [bs, 22, 3, max_len]
            motions = g_positions.permute(0, 2, 3, 1)
            verts_feet = motions[:, [8, 4], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
            verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],
                                                  axis=2)  # [bs, 2, max_len-1]
            vel_avg = verts_feet_plane_vel

            verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
            # If feet touch ground in agjecent frames
            feet_contact = np.clip(2 - 2 ** (verts_feet_height[..., 1:] / self.threshold_height), 0,
                                   1)  # [bs, 2, max_len - 1]
            c = np.sum(feet_contact > 0)
            s = vel_avg * feet_contact
            c = max(c, 1)
            m = np.sum(s) / c

        return {
            'foot skate': m
        }


class TrajErrorEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.name = 'traj'
        return

    def evaluate(self, dataloader):
        dataloader.dataset.metric_name = self.name
        all_size = 0
        traj_fail_02_sum = 0
        traj_fail_05_sum = 0
        traj_error_sum = 0
        for idx, data in enumerate(dataloader):
            gt_traj, traj = data[0], data[1]
            bs = gt_traj.shape[0]
            all_size += traj.shape[0]
            gt_traj = gt_traj.detach().cpu().numpy()
            traj = traj.detach().cpu().numpy()
            dist_error = np.linalg.norm(gt_traj - traj, axis=-1)
            traj_fail_02 = bs - (dist_error <= 20).all(axis=-1).sum()
            traj_fail_05 = bs - (dist_error <= 50).all(axis=-1).sum()
            traj_fail_02_sum += traj_fail_02
            traj_fail_05_sum += traj_fail_05
            traj_error_sum += dist_error.mean(axis=-1).sum()
        return {
            'traj fail 0.2': traj_fail_02_sum / all_size,
            'traj fail 0.5': traj_fail_05_sum / all_size,
            'traj error': traj_error_sum / all_size
        }

class PosErrorEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.name = 'pos'
        return

    def evaluate(self, dataloader):
        dataloader.dataset.metric_name = self.name
        total_num = 0
        total_error = 0
        for idx, data in enumerate(dataloader):
            gt_g_pos, g_pos, mask = data[0], data[1], data[2]
            diff = torch.norm((gt_g_pos - g_pos), dim=-1)
            diff = diff * mask[..., 0]
            num_entries = mask.sum()
            total_num += num_entries
            total_error += diff.sum()
        return {
            'pos error': (total_error / total_num).item()
        }




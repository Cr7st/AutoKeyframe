import random
from .diffusion_module import DiffusionModule
from .loss import get_loss_funcs
from utils.torch_utils import randn_tensor
import torch
import numpy as np
from utils.skeleton_torch import SkeletonMotionTorch
from utils import torch_utils, heuristic
from data.test_gen_dataset import EvaluateWrapper
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
import os


class KeyframeModule(DiffusionModule):
    def __init__(self, diffusion_cfg, train_cfg=None, loss_cfg=None, evaluators=None, result_dir=None, use_prev_rot=True,
                 use_global_rot=False, eval_with_random_hint=False, enable_grad_guide=False, compress_pose = False):
        super().__init__(train_cfg = train_cfg, loss_cfg = loss_cfg, **diffusion_cfg)
        if loss_cfg is not None:
            self.loss_funcs = get_loss_funcs(loss_cfg)
            self.loss_funcs['diffusion_loss'] = torch.nn.MSELoss()
            self.loss_weight = {key: loss_cfg[key]['weight'] for key in loss_cfg}
        else:
            self.loss_funcs = None
        self.evaluators = evaluators
        self.result_dir = result_dir
        self.l_position = np.load(os.path.join(os.path.dirname(os.path.dirname(result_dir)), 'l_position.npy'))
        self.metrics = {}
        self.eval_with_random_hint = eval_with_random_hint
        self.enable_grad_guide = enable_grad_guide

    def on_train_start(self):
        assert self.loss_funcs is not None, 'Loss functions must be defined for training!'
        data_lambda = lambda data: data['rotations']
        self.mean, self.std = self.trainer.train_dataloader.dataset.calc_stats(data_lambda, self.result_dir)

    def training_step(self, batch, batch_idx):
        data, prev_data, interval, cond = batch
        l_position = data['l_position'][:, None]
        root_trans = data['root_trans']
        x_0 = data['rotations']

        for i in range(x_0.shape[0]):
            if prev_data['rotations'][i].sum() == 0:
                prev_data['rotations'][i] = self.zscore_denormalize(torch.zeros_like(prev_data['rotations'][i]))
        prev_frame = self.zscore_normalize(prev_data['rotations'])
        x_0 = self.zscore_normalize(x_0)
        if self.hparams.compress_pose:
            x_0 = x_0.reshape(-1, 1, 22 * 6)
            prev_frame = prev_frame.reshape(-1, 1, 22 * 6)

        conditions = self.mask_conditions({
            'prev_frame': prev_frame,
            'interval': interval,
            'action': cond,
            'velocity': data['velocity'],
            'position': data['position'],
            'height': data['height'],
        })
        loss_dict = {}
        model_args = {}
        scheduler_args = {}
        loss, sample = self._diffusion_train(x_0, conditions, model_args=model_args, loss_args={}, scheduler_args=scheduler_args)
        if self.hparams.compress_pose:
            x_0 = x_0.reshape(-1, 22, 6)
            sample = sample.reshape(-1, 22, 6)
        x_0 = self.zscore_denormalize(x_0)
        sample = self.zscore_denormalize(sample)
        for key in self.loss_funcs:
            if key in self.loss_weight:
                if key == 'diffusion_loss':
                    loss_dict[key] = loss
                else:
                    loss_dict[key] = self.loss_funcs[key](sample.unsqueeze(1), x_0.unsqueeze(1),
                                                          root_trans=root_trans, gt_root_trans=root_trans,
                                                          offsets=l_position)
        loss = sum([loss_dict[key] * self.loss_weight[key] for key in loss_dict])

        self.log('total_loss', loss, prog_bar=True)
        self.log_dict(loss_dict, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    @staticmethod
    def save_frame(frames, file_name, idx):
        np.savez(file_name, rotations=frames['rotations'][idx].cpu().numpy(),
                 trans_offset=frames['trans_offset'][idx].cpu().numpy(),
                 rot_offset=frames['rot_offset'][idx].cpu().numpy())

    def forward(self, conditions, **kwargs):
        """

        Args:
            conditions (dict):
                'prev_frame' (tensor): (batch_size, joint_num, 6)
                'interval' (tensor): (batch_size, 1)
                'action' (tensor): (batch_size, 1)
                'velocity' (tensor): (batch_size, 1, 3)
                'position' (tensor): (batch_size, 1, 3)
                'height' (tensor): (batch_size, 1, 1)

        """
        assert self.mean is not None and self.std is not None, 'Data stats not loaded or calculated!'

        if 'hint_mask' in kwargs['model_args'] and self.enable_grad_guide:
            scheduler_args = {'guider': lambda m, v, t:
                              self.guide_diffusion(m, v, t, .5, kwargs['model_args']['hint'], kwargs['model_args']['hint_mask'], self.training)}
        else:
            scheduler_args = {}
        conditions['prev_frame'] = self.zscore_normalize(conditions['prev_frame'])
        if self.hparams.compress_pose:
            conditions['prev_frame'] = conditions['prev_frame'].reshape(-1, 1, 22 * 6)
        if conditions['velocity'] is None:
            conditions['velocity'] = conditions['position'] / conditions['interval']
        noise = randn_tensor(conditions['prev_frame'].shape, device=conditions['prev_frame'].device)
        sample = self.diffuse(noise, conditions, model_args=kwargs['model_args'], scheduler_args=scheduler_args,
                              uncond=kwargs.get('uncond', False))
        if self.hparams.compress_pose:
            sample = sample.reshape(-1, 22, 6)
        sample = self.zscore_denormalize(sample)
        return sample

    def on_validation_start(self) -> None:
        self.eval_wrapper = EvaluateWrapper(self.l_position)

    def validation_step(self, batch, batch_idx):
        # _, frame_list, action = batch[0], batch[1], batch[2]
        # rotation_list, root_trans_list, keyframes = self.generate_keyframes_sequence(frame_list, action)
        traj, hint, hint_mask, action, first_frame = batch[0], batch[1], batch[2], batch[3], batch[4]
        rotation_list, root_trans_list, keyframes = self.generate_from_traj_and_hints(traj, hint, hint_mask, action, None)
        skeleton = SkeletonMotionTorch()
        for i, k in enumerate(keyframes):
            rotations = rotation_list[i][:k.shape[0]]
            root_trans = root_trans_list[i][:k.shape[0]]
            skeleton.from_parent_array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20],
                                       torch.tensor(np.repeat(self.l_position[None], rotations.shape[0], axis=0)).unsqueeze(0))
            skeleton.apply_pose(root_trans, rotations.unsqueeze(1))
            g_positions = skeleton.joints_global_positions
            self.eval_wrapper.append_data(rotations.cpu().numpy(), root_trans.cpu().numpy(),
                                         g_positions[:, 0].cpu().numpy(), k, action[i].cpu().numpy(), traj[i].cpu().numpy(), hint[i].cpu().numpy(), hint_mask[i].cpu().numpy())

    def on_validation_epoch_end(self):
        self.eval_wrapper.on_append_data_end()
        dataloader = DataLoader(self.eval_wrapper, batch_size=64, shuffle=False)
        for evaluator in self.evaluators:
            metric = evaluator.evaluate(dataloader)
            self.log_dict(metric, prog_bar=True)
    
    def generate_from_traj_and_hints(self, traj, hint, hint_mask, action, first_frame=None, keyframes_list=None):
        bs = traj.shape[0]
        most_keyframes = 0
        if keyframes_list is None:
            keyframes_list = []
            for i in range(bs):
                keyframes, _ = heuristic.keyframe_jerk(traj[i].unsqueeze(1), 30, 30, smooth_window=3,
                                                       random_infill=False, nms_threshold=0.85)
                keyframes = keyframes.tolist()
                keyframes = [i for i in keyframes if traj.shape[1] - 1 > i > 9]
                if hint_mask is not None:
                    frames_with_hint = torch.any(hint_mask[i, ..., 0], dim=1)
                    frames_with_hint = torch.where(frames_with_hint)[0].tolist()
                    keyframes = [k for k in keyframes if not
                        any([control_k + 10 >= k >= control_k - 10 for control_k in frames_with_hint])]
                    keyframes += frames_with_hint
                keyframes += [9, traj.shape[1] - 1]
                keyframes = list(set(keyframes))
                keyframes.sort()
                # num = len(keyframes)
                # keyframes = [i for i in range(9, 219, 30)] + [218]
                most_keyframes = max(most_keyframes, len(keyframes))
                keyframes_list.append(keyframes)
        else:
            for i in range(bs):
                most_keyframes = max(most_keyframes, len(keyframes_list[i]))
        rotation_list = torch.zeros((bs, most_keyframes, 22, 6), device=action.device)
        root_trans_list = torch.zeros((bs, most_keyframes, 1, 3), device=action.device)
        keyframes_tensor = torch.ones((bs, most_keyframes), dtype=torch.long, device=action.device)
        for i, keyframes in enumerate(keyframes_list):
            keyframes_tensor[i, :len(keyframes)] = torch.tensor(keyframes, dtype=torch.long)
        skeleton = SkeletonMotionTorch()
        skeleton.from_parent_array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20], torch.tensor(np.repeat(self.l_position[None], bs, axis=0,), device=action.device).unsqueeze(0))
        for i in range(most_keyframes):
            root_trans = torch.gather(traj, 1, keyframes_tensor[:, i, None, None].expand(-1, -1, 3))
            root_trans_list[:, i] = root_trans
            cur_hint_mask = torch.gather(hint_mask, 1,
                                         keyframes_tensor[:, i, None, None, None].expand(-1, -1, hint.shape[-2],
                                                                                         3)).squeeze(1)
            if i == 0 and first_frame is not None:
                prev_frame = first_frame
                sample = first_frame['rotations']
            else:
                if i == 0:
                    velocity = root_trans - torch.gather(traj, 1, (keyframes_tensor[:, i, None, None] - 1).expand(-1, -1, 3)) # relative to last frame
                    gather = keyframes_tensor[:, i, None, None].expand(-1, -1, 3).clone()
                    gather[:, :, ...] = 0
                    position = root_trans - torch.gather(traj, 1, gather) # relative to last keyframe
                    prev_frame = {
                        'rot_offset': torch.eye(3, device=self.device)[None].repeat(bs, 1, 1, 1),
                        'rotations': self.zscore_denormalize(torch.zeros([bs, hint.shape[-2], 6], device=self.device))
                    }
                    velocity = torch.matmul(prev_frame['rot_offset'], velocity.unsqueeze(-1)).squeeze(-1)
                    position = torch.matmul(prev_frame['rot_offset'], position.unsqueeze(-1)).squeeze(-1)
                    uncond = True
                    interval = keyframes_tensor[:, i].to(torch.float32)[:, None]
                else:
                    velocity = root_trans - torch.gather(traj, 1, (keyframes_tensor[:, i, None, None] - 1).expand(-1, -1, 3)) # relative to last frame
                    position = root_trans - torch.gather(traj, 1, (keyframes_tensor[:, i - 1, None, None]).expand(-1, -1, 3)) # relative to last keyframe
                    velocity = torch.matmul(prev_frame['rot_offset'], velocity.unsqueeze(-1)).squeeze(-1)
                    position = torch.matmul(prev_frame['rot_offset'], position.unsqueeze(-1)).squeeze(-1)
                    uncond = False
                    interval = (keyframes_tensor[:, i] - keyframes_tensor[:, i - 1]).to(torch.float32)[:, None]

                conditions = {
                    'prev_frame': prev_frame['rotations'],
                    'interval': interval,
                    'action': action,
                    'velocity': velocity,
                    'position': position,
                    'height': root_trans[..., 1],
                }
                if cur_hint_mask.sum() > 0 or self.eval_with_random_hint:
                    cur_hint = torch.gather(hint, 1, keyframes_tensor[:, i, None, None, None].expand(-1, -1, hint.shape[-2], 3)).squeeze(1)
                    cur_hint -= root_trans
                    cur_hint = torch.matmul(prev_frame['rot_offset'], cur_hint.unsqueeze(-1)).squeeze(-1)
                    if cur_hint_mask.sum() == 0:
                        control_idx = random.choice(self.controllable_joints)
                        cur_hint_mask[:, control_idx] = 1
                    model_args = {'hint': cur_hint, 'hint_mask': cur_hint_mask}
                else:
                    model_args = {}
                sample = self.forward(conditions, model_args=model_args, uncond=uncond)

            if not self.hparams.use_global_rot and self.hparams.use_prev_rot:
                # rotate the root to global 
                root_matrix = torch_utils.matrix6D_to_9D_torch(sample[:, 0])
                root_matrix = torch.matmul(prev_frame['rot_offset'][:, 0].transpose(-1, -2), root_matrix)
                sample[:, 0] = torch_utils.matrix9D_to_6D_torch(root_matrix)
                rotation_list[:, i] = sample

                # rotate the root to face the x-axis
                skeleton.apply_pose(root_trans, sample.unsqueeze(1))
                positions = skeleton.joints_global_positions
                if i > 0 and cur_hint_mask.sum() > 0:
                    target = torch.matmul(prev_frame['rot_offset'].transpose(-1, -2), cur_hint.unsqueeze(-1)).squeeze(-1)
                    target += root_trans
                    diff = positions - target
                    distance = torch.norm(diff * cur_hint_mask, dim=-1).max()
                sample2x , cur_rot_offset = torch_utils.rotate_start_to_v2_1_torch(positions[:, 0],  sample, return_offset=True)
            else:
                raise NotImplementedError

            prev_frame = {
                'rotations': sample2x,
                'rot_offset': cur_rot_offset,
                'root_trans': root_trans
            }
        return rotation_list, root_trans_list, [ks[ks >= 9].cpu().numpy().astype(np.int32) for ks in keyframes_tensor]

    def edit_keyframes_sequence(self, frame_list, action, edit_tag=None):
        if self.l_position is None:
            raise ValueError('l_position is not loaded')

        bs = action.shape[0]

        rotation_list = torch.zeros((bs, len(frame_list), 22, 6), device=action.device)
        root_trans_list = torch.zeros((bs, len(frame_list), 1, 3), device=action.device)
        keyframes = torch.zeros((bs, len(frame_list)), device=action.device)

        skeleton = SkeletonMotionTorch()
        skeleton.from_parent_array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20],
                                   torch.tensor(np.repeat(self.l_position[None], bs, axis=0, ),
                                                device=action.device).unsqueeze(0))
        for i, frame in enumerate(frame_list):
            hint_mask = frame['hint_mask']
            if edit_tag[i] == 0 and hint_mask.sum() == 0:
                prev_frame = frame
                sample = frame['rotations']
                keyframes[:, i] = frame['index']
            else:
                if i == 0:
                    prev_frame = {
                        'rot_offset': torch.eye(3, device=self.device)[None].repeat(bs, 1, 1, 1),
                        'rotations': self.zscore_denormalize(
                            torch.zeros([bs, 22, 6], device=self.device))
                    }
                else:
                    delta_res = torch_utils.preprocess_relative_info_torch(frame, prev_frame, self.hparams.use_prev_rot,
                                                                           self.hparams.use_global_rot)
                    frame.update(delta_res)
                root_trans = frame['root_trans']

                if hint_mask.sum() > 0:
                    hint = frame['hint']
                    hint -= root_trans
                    hint = torch.matmul(prev_frame['rot_offset'], hint.unsqueeze(-1)).squeeze(-1)
                    model_args = {'hint': hint, 'hint_mask': hint_mask}
                else:
                    model_args = {}
                conditions = {
                    'prev_frame': prev_frame['rotations'],
                    'interval': frame['interval'],
                    'action': action,
                    'velocity': frame['velocity'],
                    'position': frame['position'],
                    'height': frame['height'],
                }
                sample = self.forward(conditions, model_args=model_args)
                if sample.requires_grad:
                    sample = sample.detach()
                keyframes[:, i] = frame['index']

            root_trans = frame['root_trans']
            root_trans_list[:, i] = root_trans

            if not self.hparams.use_global_rot and self.hparams.use_prev_rot:
                # rotate the root to global
                root_matrix = torch_utils.matrix6D_to_9D_torch(sample[:, 0])
                root_matrix = torch.matmul(prev_frame['rot_offset'][:, 0].transpose(2, 1), root_matrix)
                sample[:, 0] = torch_utils.matrix9D_to_6D_torch(root_matrix)
                rotation_list[:, i] = sample

                # rotate the root to face the x-axis
                skeleton.apply_pose(root_trans, sample.unsqueeze(1))
                positions = skeleton.joints_global_positions
                sample2x, cur_rot_offset = torch_utils.rotate_start_to_v2_1_torch(positions[:, 0], sample,
                                                                                  return_offset=True)

            prev_frame = {
                'rotations': sample2x,
                'rot_offset': cur_rot_offset,
                'root_trans': frame['root_trans']
            }

        return rotation_list, root_trans_list, [ks[ks <= 218].cpu().numpy().astype(np.int32) for ks in keyframes]

    def guide_diffusion(self, mean, variance, t, guide_scale, hint, hint_mask, train=False):
        if train:
            n_guide_steps = 20
        else:
            if t < 50:
                n_guide_steps = 100
            else:
                n_guide_steps = 10
            variance = max(variance, 0.001)

        bs = mean.shape[0]
        skeleton = SkeletonMotionTorch()
        skeleton.from_parent_array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20],
                                   torch.tensor(np.repeat(self.l_position[None], bs, axis=0, ),
                                                device=mean.device).unsqueeze(0))

        for _ in range(n_guide_steps):
            root_trans = torch.zeros_like(hint[:, 0][:, None])
            with torch.enable_grad():
                if t > 700:
                    x_ = self.zscore_denormalize(mean)[:, None]
                    theta = torch.zeros([bs, ], device=x_.device)
                    theta.requires_grad_(True)
                    root_rot_y_mat = torch.eye(3, device=x_.device)[None].repeat(bs, 1, 1)
                    root_rot_y_mat[:, 0, 0] = torch.cos(theta)
                    root_rot_y_mat[:, 0, 2] = torch.sin(theta)
                    root_rot_y_mat[:, 2, 0] = -torch.sin(theta)
                    root_rot_y_mat[:, 2, 2] = torch.cos(theta)
                    root_rot = torch_utils.matrix6D_to_9D_torch(x_[:, :, 0])
                    root_rot = torch.matmul(root_rot_y_mat, root_rot)
                    x_[:, :, 0] = torch_utils.matrix9D_to_6D_torch(root_rot)
                    skeleton.apply_pose(root_trans, x_)
                    joint_pos = skeleton.joints_global_positions.squeeze(1)
                    loss = torch.norm((joint_pos - hint) * hint_mask, dim=-1)
                    grad = torch.autograd.grad([loss.sum()], [theta])[0]
                    theta = theta - guide_scale * grad * variance
                    root_rot_y_mat[:, 0, 0] = torch.cos(theta)
                    root_rot_y_mat[:, 0, 2] = torch.sin(theta)
                    root_rot_y_mat[:, 2, 0] = -torch.sin(theta)
                    root_rot_y_mat[:, 2, 2] = torch.cos(theta)
                    root_rot = torch_utils.matrix6D_to_9D_torch(x_[:, :, 0])
                    root_rot = torch.matmul(root_rot_y_mat, root_rot)
                    x_[:, :, 0] = torch_utils.matrix9D_to_6D_torch(root_rot)
                    mean = self.zscore_normalize(x_[:, 0])
                else:
                    mean.requires_grad_(True)
                    x_ = self.zscore_denormalize(mean)[:, None]
                    skeleton.apply_pose(root_trans, x_)
                    joint_pos = skeleton.joints_global_positions.squeeze()
                    loss = torch.norm((joint_pos - hint) * hint_mask, dim=-1)
                    grad = torch.autograd.grad([loss.sum()], [mean])[0]
                    mask = torch.zeros_like(grad)
                    if t <= 600:
                        mask[:, [0, 9, 10, 11, 18], :] = 0
                    if t <= 400:
                        mask = torch.ones_like(grad)
                    mean.detach()
                    if train:
                        grad = variance[:, None, None] * grad
                    else:
                        grad = variance * grad * mask
                    mean = mean - guide_scale * grad
        # print(loss.sum().item())
        return mean

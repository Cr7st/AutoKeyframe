import torch
from utils.skeleton_torch import SkeletonMotionTorch
from utils import torch_utils


def get_loss_funcs(loss_config):
    parent_tree = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    loss_funcs = {}
    for loss_name, loss_params in loss_config.items():
        if loss_name == 'joint_pos_loss':
            loss_funcs[loss_name] = JointPositionLoss(parent_tree, **loss_params)
        elif loss_name == 'global_rot_loss':
            loss_funcs[loss_name] = GlobalRotationLoss(parent_tree, **loss_params)
        elif loss_name == 'penetration_loss':
            loss_funcs[loss_name] = PenetrationLoss(parent_tree, **loss_params)
        elif loss_name == 'obs_control_loss':
            loss_funcs[loss_name] = JointPositionLoss(parent_tree, **loss_params)
        else:
            loss_funcs[loss_name] = Loss(**loss_params)
    return loss_funcs


class Loss:
    def __init__(self, loss_type='L2', reduction=None, weight=None, **kwargs):
        self.type = loss_type
        if loss_type == 'L2':
            self._loss = torch.nn.MSELoss(reduction=reduction)
        elif loss_type == 'L1':
            self._loss = torch.nn.L1Loss(reduction=reduction)
        elif loss_type == 'NLL':
            self._loss = torch.nn.NLLLoss(reduction=reduction, weight=torch.tensor([1, 1, 1.5, 1.2, 1.5, 2, 0.5, 1, 0.5], device='cuda', dtype=torch.float32))
        elif loss_type == 'Masked L2':
            self._loss = masked_mse_loss
        elif loss_type == 'Masked L1':
            self._loss = masked_l1_loss
        elif loss_type == 'CrossEntropy':
            self._loss = torch.nn.CrossEntropyLoss(reduction=reduction, weight=torch.tensor([1, 1, 1.5, 1.2, 1.5, 2, 0.5, 1, 0.5], device='cuda', dtype=torch.float32))

        self.loss_type = loss_type
        self.reduction = reduction

    def __call__(self, y_pred, y_true, **kwargs):
        return self._loss(y_pred, y_true, **kwargs)


class JointPositionLoss(Loss):
    def __init__(self, parent_tree, loss_type='L2', reduction='mean', **kwargs):
        super().__init__(loss_type, reduction, **kwargs)
        self.skeleton = SkeletonMotionTorch()
        self.parent_tree = parent_tree

    def __call__(self, y_pred, y_true, **kwargs):
        """
            y is a tensor of shape (batch_size, seq_len, num_joints, 6),
            which represents the 6D joint positions.
            Should perform forward kinematics to get the global positions
        """
        self.skeleton.from_parent_array(self.parent_tree, kwargs['offsets'])
        self.skeleton.apply_pose(kwargs['root_trans'], y_pred)
        g_position = self.skeleton.joints_global_positions
        self.skeleton.apply_pose(kwargs['gt_root_trans'], y_true)
        gt_g_position = self.skeleton.joints_global_positions

        if 'Masked' in self.loss_type:
            return self._loss(g_position, gt_g_position, kwargs['mask'])
        else:
            return self._loss(g_position, gt_g_position)


class GlobalRotationLoss(Loss):
    def __init__(self, parent_tree, loss_type='L2', reduction='mean', **kwargs):
        super().__init__(loss_type, reduction, **kwargs)
        self.skeleton = SkeletonMotionTorch()
        self.parent_tree = parent_tree

    def __call__(self, y_pred, y_true, **kwargs):
        """
            y is a tensor of shape (batch_size, seq_len, num_joints, 6),
            which represents the 6D joint positions.
            Should perform forward kinematics to get the global rotations
        """
        self.skeleton.from_parent_array(self.parent_tree, kwargs['offsets'])
        self.skeleton.apply_pose(kwargs['root_trans'], y_pred)
        g_rotation = torch_utils.matrix9D_to_6D_torch(self.skeleton.joints_global_rotations)
        self.skeleton.apply_pose(kwargs['gt_root_trans'], y_true)
        gt_g_rotation = torch_utils.matrix9D_to_6D_torch(self.skeleton.joints_global_rotations)

        if 'Masked' in self.loss_type:
            return self._loss(g_rotation, gt_g_rotation, kwargs['mask'])
        else:
            return self._loss(g_rotation, gt_g_rotation)


class PenetrationLoss(Loss):
    def __init__(self, parent_tree, loss_type='L2', reduction='mean', **kwargs):
        super().__init__(loss_type, reduction, **kwargs)
        self.skeleton = SkeletonMotionTorch()
        self.parent_tree = parent_tree

    def __call__(self, y_pred, y_true, **kwargs):
        self.skeleton.from_parent_array(self.parent_tree, kwargs['offsets'])
        self.skeleton.apply_pose(kwargs['root_trans'], y_pred)
        g_position = self.skeleton.joints_global_positions
        y_value = g_position[..., 1]
        mask = y_value < -2.0
        y_value[~mask] = 0
        zeros = torch.zeros_like(y_value)
        return self._loss(y_value, zeros)


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def masked_mse_loss(pred, target, mask):
    """
    :param pred: (bs, seq_len, ...)
    :param target: same shape as pred
    :param mask:
    :return:
    """
    reshaped_mask = mask.clone()
    while len(reshaped_mask.shape) < len(pred.shape):
        reshaped_mask = reshaped_mask.unsqueeze(-1)
    loss = (pred - target) ** 2 * reshaped_mask
    loss = sum_flat(loss)
    n_entries = 1
    for i in range(2, len(pred.shape)):
        n_entries *= pred.shape[i]
    non_zero_elements = sum_flat(reshaped_mask) * n_entries
    loss = loss / non_zero_elements
    return loss.mean()

def masked_l1_loss(pred, target, mask):
    reshaped_mask = mask.clone()
    while len(reshaped_mask.shape) < len(pred.shape):
        reshaped_mask = reshaped_mask.unsqueeze(-1)
    loss = (pred - target).abs() * reshaped_mask
    loss = sum_flat(loss)
    n_entries = 1
    for i in range(2, len(pred.shape)):
        n_entries *= pred.shape[i]
    non_zero_elements = sum_flat(reshaped_mask) * n_entries
    loss = loss / non_zero_elements
    return loss.mean()

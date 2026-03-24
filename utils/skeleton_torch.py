import copy

import torch
from utils import torch_utils

# This is a pytorch version of the skeleton.py file, and support batch operations
# All rotations should be in form of 6D rotations

class JointTorch:
    def __init__(self, name, parent=None, offset=None):
        self.name = name
        self.parent = parent
        self.rotation = torch.zeros((1, 1, 6), dtype=torch.float32)
        self.offset = offset
        self.children = []
        self._global_transform = None

    def set_rotation(self, rotation):
        self.rotation = rotation
        if self.parent is not None:
            self.offset = self.offset[:, :1, :].expand(-1, self.rotation.shape[1], 3)
        self._global_transform = None

    @property
    def local_transform_matrix(self):
        rotation_matrix = torch_utils.matrix6D_to_9D_torch(self.rotation)
        if len(self.offset.shape) == 2:
            translation = self.offset.unsqueeze(1).expand(self.rotation.shape[0], -1, -1)
        else:
            translation = self.offset
        transform_matrix = (torch.eye(4, device=self.rotation.device, dtype=self.rotation.dtype).unsqueeze(0).unsqueeze(1)
                            .expand(self.rotation.shape[0], self.rotation.shape[1], -1, -1))
        transform_matrix = transform_matrix.clone()
        transform_matrix[..., :3, :3] = rotation_matrix
        transform_matrix[..., :3, 3] = translation
        return transform_matrix

    @property
    def global_transform_matrix(self):
        if self._global_transform is None:
            if self.parent is None:
                self._global_transform = self.local_transform_matrix
            else:
                self._global_transform = torch.matmul(self.parent.global_transform_matrix, self.local_transform_matrix)

        return self._global_transform

    @property
    def global_position(self):
        return self.global_transform_matrix[..., :3, 3]

    def __repr__(self):
        return f'JOINT {self.name} {self.offset}'



class SkeletonTreeTorch:
    def __init__(self, skeleton_config=None):
        if skeleton_config:
            self.joints_dict = {}
        else:
            self.joints_dict = {}

    @property
    def joints(self):
        return list(self.joints_dict.values())

    @property
    def root_joint(self):
        return self.joints[0]

    def __getitem__(self, item):
        return self.joints[item]

    def forward_kinematics(self):
        for joint in self.joints_dict.values():
            if joint.parent:
                joint.parent.children.append(joint)

    def from_parent_array(self, parents, l_positions=None):
        self.joints_dict = {}
        for i, p in enumerate(parents):
            name = f'joint_{i}'
            parent = self.joints_dict.get(f'joint_{p}', None)
            joint = JointTorch(name, parent, l_positions[..., i, :].clone() if l_positions is not None else None)
            self.joints_dict[name] = joint
        self.forward_kinematics()

    @property
    def joints_global_positions(self):
        positions = [joint.global_position.unsqueeze(-2) for joint in self.joints]
        return torch.cat(positions, dim=-2)

    @property
    def joints_global_rotations(self):
        rotations = [joint.global_transform_matrix[..., :3, :3].unsqueeze(-2) for joint in self.joints]
        return torch.cat(rotations, dim=-3)


class SkeletonMotionTorch(SkeletonTreeTorch):
    def __init__(self, root_translations=None, joints_rotations=None):
        super().__init__()
        self.root_translations = root_translations
        self.joints_rotations = joints_rotations
        if root_translations is not None and joints_rotations is not None:
            self.apply_pose(root_translations, joints_rotations)

    def from_motion_data(self, translations, rotations):
        self.root_translations = translations
        self.joints_rotations = rotations
        self.apply_pose(self.root_translations, self.joints_rotations)

    def __getitem__(self, item):
        """
        Indexing SkeletonMotion with slice will create a new motion, use with care!
        Indexing SkeletonMotion with integer will create a static Skeleton
        """
        if isinstance(item, slice):
            res = copy.deepcopy(self)
            res.joints_rotations = self.joints_rotations[item, :]
            res.root_translations = self.root_translations[item, :]
            res.apply_pose(res.root_translations, res.joints_rotations)
            return res
        elif isinstance(item, int):
            res = SkeletonTreeTorch()
            res.joints_dict = self.joints_dict
            return res

    def apply_pose(self, translation, rotation):
        if len(self.joints_dict) == 0:
            raise RuntimeError('Skeleton is empty')
        self.root_translations = translation
        self.joints_rotations = rotation
        self.root_joint.offset = translation
        for i, joint in enumerate(self.joints):
            joint.set_rotation(rotation[..., i, :])

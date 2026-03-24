import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from .bvh import Bvh
import json
from utils import data_utils


class Joint:
    def __init__(self, name, parent=None, offset=None):
        self.name=name
        self.parent = parent
        self.rotation = None
        self.offset = offset # (3,)
        self.children = []
        self._global_transform = None

    def set_rotation(self, rotation):
        self.rotation = rotation
        self._global_transform = None

    def set_global_transform(self, rotation, position):
        """

        Args:
            rotation (ndarray): (nframes, 6) global rotations
            position (ndarray): (nframes, 3) global positions

        """
        transform_matrix = np.eye(4)[None, ...].repeat(rotation.shape[0], axis=0)
        rotation_matrix = data_utils.matrix6D_to_9D(rotation)
        transform_matrix[:, :3, :3] = rotation_matrix
        transform_matrix[:, :3, 3] = position
        self._global_transform = transform_matrix
        self.rotation = None

    @property
    def local_rotation(self):
        if self.rotation is None:
            if self._global_transform is None:
                raise ValueError('global transform is not set')
            if self.parent is None:
                self.rotation = data_utils.matrix9D_to_6D(self.global_transform_matrix[:, :3, :3])
            else:
                self.rotation = data_utils.matrix9D_to_6D(self.parent.global_transform_matrix[:, :3, :3].transpose(0, 2, 1) @ self.global_transform_matrix[:, :3, :3])
        return self.rotation

    @property
    def local_transform_matrix(self):
        """

        Returns:
            (ndarray): (nframes, 4, 4) local transformation
        """
        if self.rotation is None:
            raise ValueError('rotation is not set')
        rotation_matrix = data_utils.matrix6D_to_9D(self.rotation)
        if len(self.offset.shape) == 1:
            translation = np.repeat(self.offset[None, ...], self.rotation.shape[0], axis=0)
        elif len(self.offset.shape) == 2:
            translation = self.offset
        else:
            raise ValueError('offset shape should be (3,) or (nframes, 3)')
        transform_matrix = np.eye(4)[None, ...].repeat(self.rotation.shape[0], axis=0)
        transform_matrix[:, :3, :3] = rotation_matrix
        transform_matrix[:, :3, 3] = translation
        return transform_matrix

    @property
    def global_transform_matrix(self):
        if self._global_transform is None:
            if self.parent is None:
                self._global_transform = self.local_transform_matrix
            else:
                self._global_transform = self.parent.global_transform_matrix @ self.local_transform_matrix

        return self._global_transform

    @property
    def global_position(self):
        return self.global_transform_matrix[:, :3, 3]
    
    @property
    def global_rotation(self):
        return data_utils.matrix9D_to_6D(self.global_transform_matrix[:, :3, :3])

    def __repr__(self):
        return f'JOINT {self.name} {self.offset}'



class Skeleton:
    def __init__(self, skeleton_config = None):
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
        """

        Args:
            parents
            l_positions

        """
        self.joints_dict = {}
        for i, p in enumerate(parents):
            name = f'joint_{i}'
            parent = self.joints_dict.get(f'joint_{p}', None)
            joint = Joint(name, parent, l_positions[..., i, :].copy() if l_positions is not None else None)
            self.joints_dict[name] = joint
        self.forward_kinematics()

    def from_bvh(self, bvh_file):
        """
        construct T-pose skeleton from bvh file(rotation all zero, offset from bvh file)
        :param bvh_file:
        :return:
        """
        self.joints_dict = {}
        with open(bvh_file, 'r') as f:
            bvh = Bvh(f.read())
        root = bvh.get_joints_names()[0]
        root = Joint(root, offset=np.array(bvh.joint_offset(root)))
        self.joints_dict[root.name] = root
        for node in bvh.get_joints()[1:]:
            parent_joint = self.joints_dict.get(bvh.joint_parent(node.name).name, None)
            joint = Joint(node.name, parent_joint, np.array(bvh.joint_offset(node.name)))
            self.joints_dict[node.name] = joint
        self.forward_kinematics()

    def from_json(self, json_file):
        self.joints_dict = {}
        with open(json_file, 'r') as f:
            data = json.load(f)
        for name, info in data.items():
            parent = self.joints_dict.get(info['parent'], None)
            joint = Joint(name, parent, np.array(info['offset']))
            self.joints_dict[name] = joint
        self.forward_kinematics()

    def to_json(self, file_name):
        with open(file_name, 'w') as f:
            json.dump({joint.name: {
                'offset': joint.offset.tolist(),
                'parent': joint.parent.name if joint.parent else None} for joint in self.joints}, f, indent=4)

    @property
    def joints_global_positions(self):
        positions = [joint.global_position for joint in self.joints]
        return np.stack(positions, axis=1)
    
    @property
    def joints_global_rotations(self):
        rotations = [joint.global_rotation for joint in self.joints]
        return np.stack(rotations, axis=1)
    
    @property
    def joints_local_rotations(self):
        rotations = [joint.local_rotation for joint in self.joints]
        return np.stack(rotations, axis=1)


class SkeletonMotion(Skeleton):
    def __init__(self, bvh_file=None, parents = None, l_positions = None):
        super().__init__()
        self.root_translations = None
        self.joints_rotations = None
        self.fps = None
        if not bvh_file is None:
            self.from_bvh(bvh_file)
        elif not parents is None:
            self.from_parent_array(parents, l_positions)

    def from_bvh(self, bvh_file):
        super().from_bvh(bvh_file)
        with open(bvh_file, 'r') as f:
            bvh = Bvh(f.read())
        self.root_translations = bvh.root_translation
        self.joints_rotations = bvh.joint_rotations.reshape(bvh.nframes, -1, 3)
        bvh_channels = bvh.joint_channels(self.joints[1].name)
        order = [['Xrotation', 'Yrotation', 'Zrotation'].index(channel) for channel in bvh_channels]
        self.joints_rotations = self.joints_rotations[..., order]
        self.fps = 1 / bvh.frame_time
        self.apply_pose(self.root_translations, self.joints_rotations)

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
            res = Skeleton()
            res.joints_dict = self.joints_dict
            return res

    def fk_apply_pose(self, translation, rotation):
        """
        foward kinematics initialization

        Args:
            translation (ndarray): (nframes, 3) root translation
            rotation (ndarray): (nframes, njoints, 6) local rotations

        Returns:
            None
        """
        self.root_translations = translation
        self.joints_rotations = rotation
        self.root_joint.offset = translation
        for i, joint in enumerate(self.joints):
            joint.set_rotation(rotation[:, i])

    def ik_apply_pose(self, global_positions, global_rotations):
        """

        Args:
            global_positions (ndarray): (nframes, njoints, 3)
            global_rotations (ndarray): (nframes, njoints, 6)

        Returns:
            None
        """
        self.root_translations = global_positions[:, 0]
        self.root_joint.offset = global_positions[:, 0]
        self.joints_rotations = None
        for i, joint in enumerate(self.joints):
            joint.set_global_transform(global_rotations[:, i], global_positions[:, i])


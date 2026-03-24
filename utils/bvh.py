import re
import numpy as np


class BvhNode:

    def __init__(self, value=[], parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1:]
        raise IndexError('key {} not found'.format(key))

    def __repr__(self):
        return str(' '.join(self.value))

    def __str__(self):
        return str(' '.join(self.value))

    @property
    def name(self):
        return self.value[1]


class Bvh:
    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }
    channelmap = {
        'Xrotation': 'x',
        'Yrotation': 'y',
        'Zrotation': 'z'
    }
    ordermap = {
        'x': 0,
        'y': 1,
        'z': 2,
    }
    def __init__(self, data=None):
        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.frames_data = None
        if self.data is not None:
            self.tokenize()

    def tokenize(self):
        first_round = []
        accumulator = ''
        for char in self.data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                first_round.append(re.split('\\s+', accumulator.strip()))
                accumulator = ''
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                for i, data in enumerate(item):
                    item[i] = float(data)
                self.frames.append(item)
                continue
            key = item[0]
            if key == '{':
                node_stack.append(node)
            elif key == '}':
                node_stack.pop()
            else:
                node = BvhNode(item)
                node_stack[-1].add_child(node)
            if item[0] == 'Frame' and item[1] == 'Time:':
                frame_time_found = True
        self.frames_data = np.array(self.frames)

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)

        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def get_joints_names(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter('JOINT'):
                iterate_joints(child)

        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter('JOINT')]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search('ROOT', name)
        if not found:
            found = self.search('JOINT', name)
        if found:
            return found[0]
        raise LookupError('joint not found')

    def joint_offset(self, name):
        joint = self.get_joint(name)
        offset = joint['OFFSET']
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name):
        joint = self.get_joint(name)
        return joint['CHANNELS'][1:]

    def get_joint_channels_index(self, joint_name):
        index = 0
        for joint in self.get_joints():
            if joint.value[1] == joint_name:
                return index
            index += int(joint['CHANNELS'][0])
        raise LookupError('joint not found')

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                    float(
                        self.frames[frame_index][joint_index + channel_index]
                    )
                )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                        float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def joint_parent(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    def save(self, filename):
        with open(filename, 'w') as f:
            if self.data is not None:
                f.write(self.data)
                return

            stack = self.root.children
            stack.reverse()
            t = ''
            while len(stack) > 0:
                node = stack.pop()

                if node == '{':
                    f.write(f'{t}{node}\n')
                    t += '\t'
                elif node == '}':
                    t = t[:-1]
                    f.write(f'{t}{node}\n')
                elif len(node.children) > 0:
                    f.write(f'{t}{node}\n')
                    stack.append('}')
                    stack += reversed(node.children)
                    stack.append('{')
                else:
                    f.write(f'{t}{node}\n')

            for frame in self.frames:
                f.write(' '.join([str(j) for j in frame]) + '\n')

    def load_from_data(self, l_positions, rotations, parents, fps, names=None, order='zyx'):
        def make_joint_node(offset, name, parent, order, end=False):
            if parent is None:
                joint_node = BvhNode(['ROOT', name])
                joint_node.add_child(BvhNode(['OFFSET'] + [str(offset[i]) for i in range(3)]))
                joint_node.add_child(BvhNode(['CHANNELS', '6', 'Xposition', 'Yposition', 'Zposition'] +
                                             [self.channelmap_inv[order[i]] for i in range(3)]))
            else:
                joint_node = BvhNode(['JOINT', name])
                joint_node.add_child(BvhNode(['OFFSET'] + [str(offset[i]) for i in range(3)]))
                joint_node.add_child(BvhNode(['CHANNELS', '3'] + [self.channelmap_inv[order[i]] for i in range(3)]))
                parent.add_child(joint_node)
                if end:
                    end_node = BvhNode(['End', 'Site'])
                    end_node.add_child(BvhNode(['OFFSET', '0.00', '0.00', '0.00']))
                    joint_node.add_child(end_node)

            return joint_node

        offsets = l_positions[0]
        n_joints = len(parents)
        n_frames = l_positions.shape[0]
        names = [f'joint_{i}' for i in range(n_joints)] if names is None else names
        self.root.add_child(BvhNode(['HIERARCHY']))
        self.root.add_child(make_joint_node(offsets[0], names[0], None, order))
        self.root.add_child(BvhNode(['MOTION']))
        self.root.add_child(BvhNode(['Frames:', str(n_frames)]))
        self.root.add_child(BvhNode(['Frame', 'Time:', str(1. / fps)]))

        for i in range(1, n_joints):
            make_joint_node(offsets[i], names[i], self.get_joint(names[parents[i]]), order, i not in parents)

        root_trans = l_positions[:, 0, :]
        self.frames = []
        for i in range(n_frames):
            frame = [root_trans[i, j] for j in range(3)]
            for j in range(n_joints):
                frame += [rotations[i, j, self.ordermap[k]] for k in order]
            self.frames.append(frame)
        self.frames_data = np.array(self.frames)

    @property
    def nframes(self):
        try:
            return int(next(self.root.filter('Frames:')).value[1])
        except StopIteration:
            raise LookupError('number of frames not found')

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter('Frame')).value[2])
        except StopIteration:
            raise LookupError('frame time not found')

    @property
    def root_translation(self):
        return self.frames_data[:, :3]

    @property
    def joint_rotations(self):
        return self.frames_data[:, 3:]


if __name__ == '__main__':
    with open('datasets/lafan1/aiming1_subject1.bvh', 'r') as f:
        bvh = Bvh(f.read())
    root_trans = bvh.root_translation[:, None, :]
    n_frames = root_trans.shape[0]
    offset = [np.array(bvh.joint_offset(joint))[None, None, :].repeat(n_frames, axis=0) for joint in bvh.get_joints_names()[1:]]
    l_position = np.concatenate([root_trans] + offset, axis=1)
    rotations = bvh.joint_rotations.reshape(-1, 22, 3)[..., [2,1,0]]
    names = bvh.get_joints_names()
    parents = [bvh.joint_parent_index(name) for name in names]
    test = Bvh()
    test.load_from_data(l_position, rotations, parents, 30, names)
    test.save('test.bvh')
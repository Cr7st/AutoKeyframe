import numpy as np
from scipy.spatial.transform import Rotation as sRot
from utils.skeleton import SkeletonMotion
from configs import cfg
import os
from pathlib import Path


def move_start_to_origin(root_trans, frame=0, return_offset=False):
    """
    Move the starting position to origin, while remaining y (height) unchanged.

    :param: root_trans (ndarray): (seq_len, 3)
    :return: ndarray: (seq_len, 3)
    """
    res = root_trans.copy()
    offset = root_trans[None, frame, ::2]
    res[:, ::2] = root_trans[:, ::2] - offset
    if return_offset:
        return res, root_trans[None, frame, ::2]
    else:
        return res


def rotate_start_to(root_trans, rotations, frame=0, forward_axis='x', return_offset=False):
    """
    Rotate the starting direction to face the forward direction we want.

    :param: root_trans (ndarray): (seq_len, 3)
    :param: rotations (ndarray): (seq_len, joint_num, 3), euler representation
    :param: forward_axis (str): 'x', 'z'
    :return:
    """
    rotations_res = rotations.copy()
    root_trans_res = root_trans.copy()
    root_rot = rotations[:, 0, :]
    root_rot = sRot.from_euler('xyz', root_rot, degrees=True)

    # in lafan1 dataset, zero-pose faces y-axis with x-axis as the up direction
    # We want to rotate it to y-up and faces the "forward_axis"
    y = np.array([[0.0, 1.0, 0.0]])
    y_rotated = np.matmul(root_rot[frame].as_matrix(), y[..., None])[..., 0]

    # project y_rotated to xz-plane
    # We don't want root joint to face forward strictly,
    # since it may be doing pose like leaning and lying
    y_rotated[..., 1] = 0
    y_rotated = y_rotated / np.linalg.norm(y_rotated, axis=-1, keepdims=True)

    if forward_axis == 'x':
        forward = np.array([[1.0, 0.0, 0.0]])
    elif forward_axis == 'z':
        forward = np.array([[0.0, 0.0, 1.0]])
    else:
        raise ValueError("forward_axis expect value 'x' or 'z', "
                         "got '{}'.".format(forward_axis))

    dot = np.dot(y_rotated, forward.T)
    cross = np.cross(y_rotated, forward)
    angle = np.arctan2(np.dot(cross, y.T), dot)[0, 0]

    euler_angle = np.array([[0, angle, 0]])
    matrix = sRot.from_euler('xyz', euler_angle, degrees=False).as_matrix()

    root_rot = sRot.from_matrix(np.matmul(matrix, root_rot.as_matrix())).as_euler('xyz', degrees=True)
    rotations_res[:, 0, :] = root_rot
    root_trans_res = np.matmul(matrix, root_trans[..., None])[..., 0]

    if return_offset:
        return root_trans_res, rotations_res, matrix
    else:
        return root_trans_res, rotations_res

def calc_rot_offset(joints_positions, forward_axis='x'):
    """
    Calculate the rotation matrix to rotate the root joint to face the forward direction we want.

    Args:
        joints_positions (ndarray): (joint_num, 3) global positions
        forward_axis (str): 'x', 'z'
    
    Returns:
        ndarray: (1, 3, 3) rotation matrix 
    """
    # 1 for LeftUpLeg, 5 for RightUpLeg
    hips_left = joints_positions[1, :] - joints_positions[5, :]
    hips_left /= np.linalg.norm(hips_left)

    # 14 for LeftShoulder, 18 for RightShoulder
    shoulder_left = joints_positions[14, :] - joints_positions[18, :]
    shoulder_left /= np.linalg.norm(shoulder_left)

    left = (hips_left + shoulder_left) / 2
    left = (left / np.linalg.norm(left))[None, ...]
    left[..., 1] = 0

    up = np.array([[0.0, 1.0, 0.0]])

    face = np.cross(left, up)
    face /= np.linalg.norm(face)

    if forward_axis == 'x':
        forward = np.array([[1.0, 0.0, 0.0]])
    elif forward_axis == 'z':
        forward = np.array([[0.0, 0.0, 1.0]])
    else:
        raise ValueError("forward_axis expect value 'x' or 'z', "
                         "got '{}'.".format(forward_axis))

    cos = np.dot(face, forward.T) # \cos\tehta
    sin = np.dot(np.cross(face, forward), up.T) # \sin\theta
    angle = np.arctan2(sin, cos)[0, 0] # \tan\theta = \frac{\sin\theta}{\cos\theta}

    euler_angle = np.array([[0, angle, 0]])
    matrix = sRot.from_euler('xyz', euler_angle, degrees=False).as_matrix()
    matrix = matrix.astype(np.float32)

    return matrix

def rotate_start_to_v2(joints_positions, rotations, frame=0, forward_axis='x', return_offset=False):
    """

    Args:
        joints_positions (ndarray): (seq, joint_num, 3) global positions
        rotations (ndarray): (seq, joint_num, 3/6) local rotations, euler or 6d
        frame (int): frame index
        forward_axis (str): 'x', 'z'
        return_offset (bool): whether to return the rotation matrix

    Returns:
        tuple: (g_pos_res, rotations_res, matrix):
            g_pos_res (ndarray): (seq, joint_num, 3) global positions
            rotations_res (ndarray): (seq, joint_num, 3/6) local rotations, euler or 6d
            matrix (ndarray, optional): (1, 3, 3) rotation matrix
    """
    rotations_res = rotations.copy()
    g_pos_res = joints_positions.copy()

    matrix = calc_rot_offset(joints_positions[frame], forward_axis)

    root_rot = rotations[:, 0, :]
    if root_rot.shape[-1] == 3: # euler representation
        root_rot = sRot.from_euler('xyz', root_rot, degrees=True)
        root_rot = sRot.from_matrix(np.matmul(matrix, root_rot.as_matrix())).as_euler('xyz', degrees=True)
    elif root_rot.shape[-1] == 6: # 6d representation
        root_rot = matrix6D_to_9D(root_rot)
        root_rot = np.matmul(matrix, root_rot)
        root_rot = matrix9D_to_6D(root_rot)
    rotations_res[:, 0, :] = root_rot
    g_pos_res = np.matmul(matrix, joints_positions[..., None])[..., 0]

    if return_offset: 
        return g_pos_res, rotations_res, matrix
    else:
        return g_pos_res, rotations_res

def rotate_start_to_v2_1(joints_positions, rotations, forward_axis='x', return_offset=False):
    """
    Rotate the starting direction to face the forward direction we want.

    Args:
        joints_positions (ndarray): (joint_num, 3) global positions
        rotations (ndarray): (joint_num, 6) local rotations

    Returns:
        rotations_res (ndarray): (joint_num, 6) local rotations
        matrix (ndarray, optional): (1, 3, 3) rotation
    """
    matrix = calc_rot_offset(joints_positions, forward_axis)

    rotations_res = rotations.copy()
    root_rot = rotations[0]
    root_rot = matrix6D_to_9D(root_rot)
    root_rot = np.matmul(matrix, root_rot)
    rotations_res[0] = matrix9D_to_6D(root_rot)

    if return_offset:
        return rotations_res, matrix
    else:
        return rotations_res


def normalize(array, axis=-1, eps=1e-5):
    """
    Normalize ndarray along given axis.

    Args:
        array (ndarray) N-dimensional array.
        axis (int, optional): Axis. Defaults to -1.
        eps (float, optional): Small value to avoid division by zero.
            Defaults to 1e-5.

    Returns:
        ndarray: Normalized N-dimensional array.
    """
    magnitude = np.linalg.norm(array, axis=axis, keepdims=True)
    magnitude[magnitude < eps] = np.inf
    return array / magnitude

def matrix6D_to_9D(mat):
    return rotation_6d_to_matrix(mat) 

def matrix9D_to_6D(mat):
    return matrix_to_rotation_6d(mat)

def rotation_6d_to_matrix(d6: np.float32) -> np.float32:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    d6 = d6.copy()
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1, axis=-1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2, axis=-1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.float32) -> np.float32:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))


def matrix6D_to_9D_old(mat):
    """
    Convert 6D rotation representation to 3x3 rotation matrix using
    Gram-Schmidt orthogonalization.

    Args:
        mat (ndarray): Input 6D rotation representation. Shape: (..., 6)

    Raises:
        ValueError: Last dimension of mat is not 6.

    Returns:
        ndarray: Output rotation matrix. Shape: (..., 3, 3)
    """
    if mat.shape[-1] != 6:
        raise ValueError(
            "Last two dimension should be 6, got {0}.".format(mat.shape[-1]))

    mat = mat.copy().reshape(*mat.shape[:-1], 3, -1)

    # normalize column 0
    mat[..., 0] = normalize(mat[..., 0], axis=-1)

    # calculate column 1
    dot_prod = np.matmul(mat[..., 0][..., None, :], mat[..., 1][..., None]).squeeze(-1)
    mat[..., 1] -= dot_prod * mat[..., 0]
    mat[..., 1] = normalize(mat[..., 1], axis=-1)

    # calculate last column using cross product
    last_col = np.cross(mat[..., 0:1], mat[..., 1:2],
                        axisa=-2, axisb=-2, axisc=-2)

    return np.concatenate([mat, last_col], axis=-1)


def matrix9D_to_6D_old(mat):
    return mat[..., :-1].reshape((*mat.shape[:-2], 6))


# reference to https://github.com/orangeduck/Motion-Matching/blob/fe1e6e6ec274b04e412a08098848337836860a4c/resources/quat.py#L169
def fk(lrot, lpos, parents):
    """

    Args:
        lrot (ndarray): (nframes, joint_num, 3, 3) local rotations
        lpos (ndarray): (nframes, joint_num, 3) local positions
        parents (list): parent index of each joint

    Returns:
        ndarray: (nframes, joint_num, 3, 3) global rotations
        ndarray: (nframes, joint_num, 3) global positions
    """
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:,:]]
    for i in range(1, len(parents)):
        gp.append(np.matmul(gr[parents[i]], lpos[...,i:i+1,:, None])[..., 0] + gp[parents[i]])
        gr.append(np.matmul(gr[parents[i]], lrot[...,i:i+1,:,:]))
        
    return np.concatenate(gr, axis=-3), np.concatenate(gp, axis=-2)

# reference to https://github.com/orangeduck/Motion-Matching/blob/fe1e6e6ec274b04e412a08098848337836860a4c/resources/quat.py#L178
def ik(grot, gpos, parents):
    """

    Args:
        grot (ndarray): (nframes, joint_num, 3, 3) global rotations
        gpos (ndarray): (nframes, joint_num, 3) global positions
        parents (list): parent index of each joint

    Returns:
        ndarray: (nframes, joint_num, 3, 3) local rotations
    """

    return np.concatenate([
        grot[...,:1,:,:],
        np.matmul(grot[...,parents[1:],:,:].transpose(0, 1, 3, 2), grot[...,1:,:,:]),
    ], axis=-3)


# reference to https://github.com/orangeduck/Motion-Matching/blob/fe1e6e6ec274b04e412a08098848337836860a4c/resources/generate_database.py#L15
def swap_left_right(lrot, lpos, origin_gpos = None):
    """

    Args:
        lrot (ndarray): (nframes, joint_num, 6) local rotations
        lpos (ndarray): (joint_num, 3) local positions
        origin_gpos (ndarray): (nframes, joint_num, 3) global position

    Returns:
        ndarray: (nframes, joint_num, 6) mirror local rotations
        ndarray: (nframes, joint_num, 3) mirror global positions
    """

    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    joints_mirror = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 18, 19, 20, 21, 14, 15, 16, 17]
    # right_chain = [5, 6, 7, 8, 18, 19, 20, 21]
    # left_chain = [1, 2 ,3, 4, 14, 15, 16, 17]
    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    # skeleton = SkeletonMotion(parents = parents, l_positions = lpos)
    # root_trans = origin_gpos[:, 0]
    # skeleton.fk_apply_pose(root_trans, lrot)
    # gpos = skeleton.joints_global_positions
    # grot = skeleton.joints_global_rotations

    if origin_gpos is not None:
        root_trans = origin_gpos[:, 0]
        lpos = np.repeat(lpos[None, ...], lrot.shape[0], axis=0)
        lpos[:, 0] = root_trans
    grot, gpos = fk(matrix6D_to_9D(lrot), lpos, parents)

    # assert np.allclose(gpos, origin_gpos, atol=1e-4), f"atol is {np.max(np.abs(gpos - origin_gpos))}"

    gpos_mirror = mirror_pos * gpos[:, joints_mirror]
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    # gpos_mirror = gpos

    # skeleton.ik_apply_pose(gpos_mirror, grot_mirror)
    # lrot_mirror = skeleton.joints_local_rotations

    lrot_mirror = ik(grot_mirror, gpos_mirror, parents)
    lrot_mirror = matrix9D_to_6D(lrot_mirror)

    # assert np.allclose(lrot, lrot_mirror, atol=1e-4), f"atol is {np.max(np.abs(gpos - origin_gpos))}"

    return lrot_mirror, gpos_mirror

def preprocess_relative_info(frame, prev_frame = None, use_prev_rot = True, use_global_rot = False):
    """
    Preprocess relative information between two frames.

    Args:
        prev_frame (dict): Previous frame preprocessed data:
            'root_trans' (ndarray): (1, 3) global position of root joint
            'rot_offset' (ndarray): (1, 3, 3) rotation matrix to rotate the root joint to face the x-axis
        frame (dict): Frame data, should contain following keys:
            'root_trans' (ndarray): (1, 3) global position of root joint
            'rotations' (ndarray): (joint_num, 6) local rotations. rotate the root joint to face the x-axis using rot_offset
            'rot_offset' (ndarray): (1, 3, 3) rotation matrix to rotate the root joint to face the x-axis
            'velocity' (ndarray): (1, 3) velocity of root joint, rotated to face the x-axis
        use_prev_rot (bool, optional): Whether to use the previous frame's rot_offset to rotate the current frame's rotation.
        use_global_rot (bool, optional): Whether to use global rotation for the root rotation. Cover the use_prev_rot option.
    Returns:
        dict: Preprocessed relative information:
            'rotations' (ndarray): (joint_num, 6) local rotations. rotate the root joint using the rot_offset of the previous frame
            'velocity' (ndarray): (1, 3) velocity of root joint, rotated using the rot_offset of the previous frame
            'position' (ndarray): (1, 3) displacement of root joint from the previous frame
    """
    if prev_frame is None:
        return {
            'position': np.zeros((1, 3)).astype(np.float32),
        }

    trans_diff = frame['root_trans'] - prev_frame['root_trans']
    if not use_global_rot:
        position = np.matmul(prev_frame['rot_offset'], trans_diff[0]).astype(np.float32)
    else:
        position = trans_diff.astype(np.float32)

    res = {
        'position': position,
    }

    if not use_global_rot and use_prev_rot:
        prev_rot = np.matmul(prev_frame['rot_offset'], frame['rot_offset'].transpose(0, 2, 1)) # R_{prev} R_{cur}^\top

        relative_rotations = matrix6D_to_9D(frame['rotations'])
        relative_rotations[0] = np.matmul(prev_rot[0], relative_rotations[0])
        relative_rotations = matrix9D_to_6D(relative_rotations).astype(np.float32)
        relative_velocity = np.matmul(prev_rot, frame['velocity'][0]).astype(np.float32)

        res.update({
            'rotations': relative_rotations,
            'velocity': relative_velocity,
        })

    return res

def preprocess_frame(frame, prev_frame = None, meta = None, use_prev_rot = True, use_global_rot = False, mirror = False):
    """
    Preprocess a single frame of data.

    Args:
        prev_frame (dict, optional): Previous frame preprocessed data
        frame (dict): Frame data with following keys:
            'g_position' (ndarray): (joint_num, 3) global positions
            'rotations' (ndarray): (joint_num, 6) local rotations
            'velocity' (ndarray): (3, ) velocity of root joint
        meta (dict, optional): Additional meta information to be included in the output. Defaults to None.
    
    Returns:
        dict: Preprocessed data:
            'root_trans': (1, 3) global position of root joint
            'rotations': (joint_num, 6) local rotations. rotate the root joint to face the x-axis using root_trans
            'rot_offset': (1, 3, 3) rotation matrix to rotate the root joint to face the x-axis
            'velocity': (1, 3) velocity of root joint, rotated to face the x-axis
            'position': (1, 3) displacement of root joint from the previous frame
            'height': (1, 1) height of the root joint
            and additional meta information if provided
    """
    original_rotations = frame['rotations']
    original_positions = frame['g_position']

    if mirror and 'l_position' in meta:
        original_rotations, original_positions = swap_left_right(original_rotations[None], meta['l_position'], original_positions[None])
        original_rotations = original_rotations[0]
        original_positions = original_positions[0]

    root_trans = original_positions[None, 0]
    height = root_trans[..., 1:2].astype(np.float32)

    if not use_global_rot:
        rotations, rot_offset = rotate_start_to_v2_1(original_positions, original_rotations, return_offset=True)
        velocity = np.matmul(rot_offset, frame['velocity']).astype(np.float32)
    else:
        rot_offset = np.eye(3, dtype=np.float32)[None]
        rotations = original_rotations
        velocity = frame['velocity'][None].astype(np.float32)

    res = {
        'root_trans': root_trans.astype(np.float32),
        'rotations': rotations.astype(np.float32),
        'original_rotations': rotations.copy().astype(np.float32),
        'rot_offset': rot_offset.astype(np.float32),
        'velocity': velocity,
        'height': height,
    }

    delta_res = preprocess_relative_info(res, prev_frame, use_prev_rot, use_global_rot)
    res.update(delta_res)

    if meta is not None:
        res.update(meta)

    return res


def save_results(res_dataloader, with_gt=False, res_cfg = None,l_pos=None,  reset=False):
    res_dataloader.dataset.metric_name = 'test'
    if res_cfg is None:
        from configs import cfg
        res_cfg = cfg

    if l_pos is None:
        l_pos = np.load(f'{res_cfg.original_result_dir}/l_position.npy')

    for i, data in enumerate(res_dataloader):
        g_positions, rotations, keyframes, actions = data
        name = Path(res_cfg.test_path[i]).stem
        res_npz_path = os.path.join(res_cfg.output_dir, f"{name}.npz")
        res_bvh_path = os.path.join(res_cfg.output_dir, f"{name}.bvh")

        g_positions = g_positions[0].cpu().numpy()
        rotations = rotations[0].cpu().numpy()
        keyframes = keyframes[0].cpu().numpy()
        actions = actions[0].cpu().numpy()
        if reset:
            offset_matrix = offset_matrix[0, 0].cpu().numpy()

        keyframes = keyframes[keyframes>=9]

        matrix_rot = matrix6D_to_9D(rotations)
        euler_rot = sRot.from_matrix(matrix_rot.reshape(-1 , 3, 3)).as_euler('xyz', degrees=True).reshape(-1, 22, 3)
        x = np.repeat(l_pos[None], rotations.shape[0], axis=0)
        x[:, 0] = g_positions[:, 0]

        np.savez(res_npz_path, rotations=rotations, l_positions=x, g_positions=g_positions, keyframes=keyframes, action=actions)

        bvh = Bvh()
        bvh.load_from_data(x, euler_rot, res_cfg.parents, 30, res_cfg.joint_names)
        bvh.save(res_bvh_path)

        if with_gt:
            gt_data = np.load(res_cfg.test_path[i], allow_pickle=True)
            gt_rotations = gt_data['rotations']
            gt_g_positions = gt_data['g_positions']
            if reset:  #FIXME reset has bug, the trajectory is not correct
                trans_offset = gt_g_positions[keyframes[0], 0, ::2].copy()
                gt_g_positions[:, :, ::2] -= trans_offset
                gt_g_positions = np.matmul(offset_matrix, gt_g_positions[..., None])[..., 0]

            gt_matrix_rot = matrix6D_to_9D(gt_rotations)
            if reset:
                gt_matrix_rot[:, 0] = np.matmul(offset_matrix, gt_matrix_rot[:, 0])

            gt_euler_rot = sRot.from_matrix(gt_matrix_rot.reshape(-1 , 3, 3)).as_euler('xyz', degrees=True).reshape(-1, 22, 3)
            gt_x = np.repeat(l_pos[None], gt_rotations.shape[0], axis=0)
            gt_x[:, 0] = gt_g_positions[:, 0]

            bvh = Bvh()
            bvh.load_from_data(gt_x, gt_euler_rot, res_cfg.parents, 30, res_cfg.joint_names)
            bvh.save(res_bvh_path.replace('.bvh', '_gt.bvh'))

import numpy as np
import torch
import matplotlib.pyplot as plt


def smooth(y, box_pts=3):
    box = torch.ones(box_pts, device=y.device) / box_pts
    y_smooth = torch.nn.functional.conv1d(y.view(1, 1, -1), box.view(1, 1, -1), padding=box_pts // 2).view(-1)
    return y_smooth


def approximate_jerk(frames, fps):
    jkp = ((frames[3:] - 3 * frames[2:-1] + 3 * frames[1:-2] - frames[:-3]) * (fps ** 3)).norm(dim=-1)
    return jkp

def approximate_accel(frames, fps):
    jkp = ((frames[2:] - 2 * frames[1:-1] + frames[:-2]) * (fps ** 2)).norm(dim=-1)
    return jkp

def approximate_velocity(frames, fps):
    jkp = ((frames[1:] - frames[:-1]) * fps).norm(dim=-1)
    return jkp

def soft_nms(jerks, hard_t, soft_t, threshold=0.8):
    # find supress the values that with closer index to max values
    if jerks.size(0) == 0:
        return torch.empty((0,), dtype=torch.long)
    # normalize the jerks
    jerks_normalized = jerks / jerks.max()
    # make sure that the first and last frames are selected as keyframes
    jerks_normalized[6] = jerks_normalized[-1] = 1.
    max_jerks_index = torch.argsort(jerks_normalized, descending=True)
    mean = jerks_normalized[max_jerks_index].mean()
    start_set = max_jerks_index[torch.where(jerks_normalized[max_jerks_index] > mean)[0]].tolist()
    start_set.reverse()
    keyframes = []
    while len(start_set) > 0:
        k = start_set.pop()
        jerk_val = jerks_normalized[k]
        start_set = [f for f in start_set if f < k - hard_t or f > k + hard_t]
        start_set = [f for f in start_set if abs(f - k) > soft_t or jerks_normalized[f] > threshold]
        keyframes.append(k)

    return torch.tensor(keyframes, device=jerks.device, dtype=torch.long)


def non_max_suppression_jerks(jerks, kf_rate, threshold=0.85):
    """
    jerks: torch.tensor, the jerk magnitude of each frame
    kf_rate: int, the average number of frames between two keyframes
    threshold: float, the threshold for non-maximum suppression

    return: torch.tensor, the index of keyframes
    """
    # find supress the values that with closer index to max values
    if jerks.size(0) == 0:
        return torch.empty((0,), dtype=torch.long)
    # normalize the jerks
    jerks_normalized = jerks / jerks.max()
    # make sure that the first and last frames are selected as keyframes
    jerks_normalized[6] = jerks_normalized[-1] = 1.
    max_jerks_index = torch.argsort(jerks_normalized, descending=True)
    mean = jerks_normalized[max_jerks_index].mean()
    max_jerks_index = max_jerks_index[torch.where(jerks_normalized[max_jerks_index] > mean)[0]]

    ind = 0
    while ind < len(max_jerks_index):
        i = max_jerks_index[ind]
        close_index = max_jerks_index[torch.where(torch.abs(max_jerks_index - i) < (kf_rate // 2))[0]]
        corresponding_values = jerks_normalized[[close_index]]
        max_index = torch.argmax(corresponding_values)
        diff_jerk = torch.abs(corresponding_values - corresponding_values[max_index])

        # remove every one within the threshold, which is very close to the max value
        combine = (diff_jerk < threshold).bool() & (diff_jerk > 0).bool()
        combine_index = close_index[combine]
        rm_index = torch.isin(max_jerks_index, combine_index)

        if rm_index.shape[0] > 0:
            max_jerks_index = max_jerks_index[~rm_index]

        ind += 1

    return max_jerks_index


def keyframe_jerk(joint_positions, fps, keyframe_rate, smooth_window=3, nms_threshold=0.85, min_block_size=40,
                  min_gap=5, random_infill=True):
    if type(joint_positions) != torch.Tensor:
        joint_positions = torch.tensor(joint_positions)
    jerk_magnitude = approximate_jerk(joint_positions[:, 0, :], fps)
    jerk_magnitude = smooth(jerk_magnitude, smooth_window)

    keyframe_index_array = soft_nms(jerk_magnitude, 7, 15, threshold=nms_threshold)
    # keyframe_index_array = non_max_suppression_jerks(jerk_magnitude, keyframe_rate, threshold=nms_threshold)
    # fill large gaps
    # find whether the gap between two keyframes is larger than min_block_size
    keyframe_index_array = keyframe_index_array.sort()[0]
    gaps = keyframe_index_array[1:] - keyframe_index_array[:-1]
    if random_infill:
        # random fill the gap with the frames
        # for i in range(len(gaps)):
        i = 0
        while i < len(gaps):
            if gaps[i] >= min_block_size:
                start = keyframe_index_array[i]
                end = keyframe_index_array[i + 1]
                # +5, -5, avoid selected frame to be too close to the start and end
                randinfill = torch.randint(start + 10, end - 10, (torch.floor(gaps[i] / min_block_size).int(),), device=keyframe_index_array.device)

                keyframe_index_array = torch.cat(
                    (keyframe_index_array[:i + 1], randinfill, keyframe_index_array[i + 1:]))
                keyframe_index_array = keyframe_index_array.sort()[0]
                gaps = keyframe_index_array[1:] - keyframe_index_array[:-1]
            i += 1
    else:
        i = 0
        while i < len(gaps):
            if gaps[i] >= min_block_size:
                start = keyframe_index_array[i]
                end = keyframe_index_array[i + 1]
                jerk_magnitude_block = jerk_magnitude[start + min_gap:end - min_gap]

                if jerk_magnitude_block.numel() > 0:
                    max_index = torch.argmax(jerk_magnitude_block)
                    keyframe_index_array = torch.cat((keyframe_index_array[:i + 1],
                                                      torch.tensor([(start + end) // 2], device=keyframe_index_array.device),
                                                      keyframe_index_array[i + 1:]))
                    keyframe_index_array, _ = keyframe_index_array.sort()
            i += 1
            gaps = keyframe_index_array[1:] - keyframe_index_array[:-1]
            if (gaps <= min_block_size).all():
                break
            if i == len(gaps):
                i = 0

    # +3 make up for the 3 frames that were removed in the jerk calculation
    return keyframe_index_array + 3, jerk_magnitude


if __name__ == '__main__':
    dataset_root = r"./exps/guidance_on_action"
    dataset_root_case1 = r"{}/aiming1_subject1_730_948".format(dataset_root)

    for i, data_root in enumerate([dataset_root_case1]):
        data = np.load(data_root + r".npz")

        fps = 30
        key_frames_gt = data['keyframes'] - 3
        framerate = len(data['g_positions']) // len(key_frames_gt)

        keyframe_jerk_nms, jerk_magnitude = keyframe_jerk(data['g_positions'], fps, framerate, smooth_window=3,
                                                          nms=True, nms_threshold=0.85)
        keyframe_jerk_block, jerk_magnitude = keyframe_jerk(data['g_positions'], fps, framerate, smooth_window=3,
                                                            nms=False, nms_threshold=0.85)

    print("Done")

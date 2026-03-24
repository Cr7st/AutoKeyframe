import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import cfg

def run_single_test():
    import torch
    import numpy as np
    from utils import data_utils
    from scipy.spatial.transform import Rotation as sRot
    from utils.skeleton_torch import SkeletonMotionTorch
    from tqdm import tqdm
    from lightning_modules import load_lightning
    from utils.debug_util import run_cmd

    model = load_lightning(cfg.lightning_module, cfg.test_checkpoint, cfg.lightning_cfg)
    import lightning as L
    # L.seed_everything(7)

    model.eval()
    model.l_position = np.load('exps/l_position.npy')
    model.mean = np.load(f"{cfg.result_dir}/mean.npy")
    model.std = np.load(f"{cfg.result_dir}/std.npy")
    model = model.to('cuda')

    data = np.load('./sample_data/fight1_subject3_4720_4938.npz', allow_pickle=True)
    traj = (data['g_positions'][:, 0])
    if 'hint' in data:
        hint = data['hint']
        hint_mask = data['hint_mask']
    else:
        hint = data['g_positions']
        hint_mask = np.zeros_like(hint)
    full_positions = np.tile(model.l_position, (1, traj.shape[0], 1, 1))
    action = torch.tensor([4], dtype=torch.float32, device=model.device).unsqueeze(0)
    # fisrt_frame = data_utils.preprocess_frame({
    #             'g_position': hint[9],
    #             'rotations': data['rotations'][9],
    #             'velocity': traj[9] - traj[8],
    #         })
    # for key, value in fisrt_frame.items():
    #     fisrt_frame[key] = torch.tensor(value, dtype=torch.float32, device=model.device).unsqueeze(0)
    traj = torch.tensor(traj, dtype=torch.float32, device=model.device).unsqueeze(0)
    hint = torch.tensor(hint, dtype=torch.float32, device=model.device).unsqueeze(0)
    hint_mask = torch.tensor(hint_mask, dtype=torch.float32, device=model.device).unsqueeze(0)
    hint_mask[:, 94, 13, :] = 1
    hint_mask[:, 94, 21, :] = 1
    hint_mask[:, 94, 7, :] = 1
    hint_mask[:, 94, 3, :] = 1
    hint_mask[:, 94, 17, :] = 1
    hint_mask[:, 135, [17, 21], :] = 1
    model.enable_grad_guide = True
    with torch.no_grad():
        rotation_list, root_trans_list, keyframes = \
            model.generate_from_traj_and_hints(traj, hint, hint_mask, action, None,
                                               keyframes_list=[[9, 38, 70, 83, 94, 101, 135, 144, 169, 194, 218]])
    
    skeleton = SkeletonMotionTorch()
    full_rotations = torch.zeros((1, traj.shape[1], 22, 6))
    full_gpos = torch.zeros((1, traj.shape[1], 22, 3))

    for e, ks in enumerate(keyframes): 
        rotations = rotation_list[e][:ks.shape[0]].cpu()
        root_trans = root_trans_list[e][:ks.shape[0]].cpu()

        full_rotations[e][ks] = rotations
        full_rotations[e][:ks[0]] = rotations[0]

        skeleton.from_parent_array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20], torch.tensor(np.repeat(model.l_position[None], rotations.shape[0], axis=0)).unsqueeze(0))
        skeleton.apply_pose(root_trans, rotations.unsqueeze(1))
        g_positions = skeleton.joints_global_positions

        full_gpos[e][ks] = g_positions[:, 0]
        full_gpos[e, :, 0] = traj[e, :]
        full_positions[e, :, 0] = traj[e, :].cpu().numpy()
        full_positions[e, ks, 0] = g_positions[:, 0, 0]
        full_positions[e, :ks[0], 0] = g_positions[0, 0, 0]

        save_path = os.path.join(cfg.output_dir, f"single_test_{e}")

        np.savez(save_path, rotations=full_rotations[e].numpy(), l_positions=full_positions[e],
                 g_positions=full_gpos[e].numpy(), keyframes=ks, action=action[e].cpu().numpy(),
                 gt_hint=hint[e].cpu().numpy(), hint_mask=hint_mask[e].cpu().numpy())

def run_edit():
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as sRot
    from torch.utils.data import DataLoader
    from utils.skeleton_torch import SkeletonMotionTorch
    from tqdm import tqdm
    from data import make_data_loader
    from lightning_modules import load_lightning, make_trainer
    from utils.debug_util import run_cmd
    from utils.data_utils import save_results
    from data.test_gen_dataset import EvaluateWrapper, EditEvalDataset

    dataset = EditEvalDataset()
    dataset.load_mib_data('edit.npz')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_lightning(cfg.lightning_module, cfg.test_checkpoint, cfg.lightning_cfg)
    # import lightning as L
    # L.seed_everything(7)

    model.eval()
    model.l_position = np.load('exps/l_position.npy')
    model.mean = np.load(f"{cfg.result_dir}/mean.npy")
    model.std = np.load(f"{cfg.result_dir}/std.npy")
    model = model.to('cuda')
    model.enable_grad_guide = True

    for batch in dataloader:
        gt_hint, hint_mask, frame_list, actions = batch[0], batch[1], batch[2], batch[3]
        for i in range(len(frame_list)):
            for key, value in frame_list[i].items():
                frame_list[i][key] = value.to('cuda')

        actions = actions.to('cuda')

        edit_tag = np.zeros((len(frame_list)), dtype=np.int32)
        for i, f in enumerate(frame_list):
            if f['hint_mask'].sum() > 0:
                edit_tag[i] = 1

        rotation_list, root_trans_list, keyframes = model.edit_keyframes_sequence(frame_list, actions, edit_tag)

    skeleton = SkeletonMotionTorch()
    full_rotations = torch.zeros((1, gt_hint.shape[1], 22, 6))
    full_gpos = torch.zeros((1, gt_hint.shape[1], 22, 3))
    full_positions = np.tile(model.l_position, (1, gt_hint.shape[1], 1, 1))

    for e, ks in enumerate(keyframes):
        rotations = rotation_list[e][:ks.shape[0]].cpu()
        root_trans = root_trans_list[e][:ks.shape[0]].cpu()

        full_rotations[e][ks] = rotations
        full_rotations[e][:ks[0]] = rotations[0]

        skeleton.from_parent_array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20],
                                   torch.tensor(
                                       np.repeat(model.l_position[None], rotations.shape[0], axis=0)).unsqueeze(0))
        skeleton.apply_pose(root_trans, rotations.unsqueeze(1))
        g_positions = skeleton.joints_global_positions

        full_gpos[e][ks] = g_positions[:, 0]
        # full_gpos[e, :, 0] = traj[e, :]
        # full_positions[e, :, 0] = traj[e, :].cpu().numpy()
        full_positions[e, ks, 0] = g_positions[:, 0, 0]
        full_positions[e, :ks[0], 0] = g_positions[0, 0, 0]

        mib_path = os.path.join(cfg.output_dir, f"mib_single_test_{e}")

        np.savez(mib_path, rotations=full_rotations[e].numpy(), l_positions=full_positions[e],
                 g_positions=full_gpos[e].numpy(), keyframes=ks, action=actions[e].cpu().numpy())


def run_test():
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as sRot
    from utils.skeleton_torch import SkeletonMotionTorch
    from tqdm import tqdm
    from data import make_data_loader
    from lightning_modules import load_lightning
    from utils.debug_util import run_cmd
    from scripts.reprocess import reprocess_main

    cfg.test_dataset.base_cfg.dataset_dir = cfg.test_path

    dataloader = make_data_loader(cfg.test_dataset, cfg.test, cfg.test_dataset_module)

    model = load_lightning(cfg.lightning_module, cfg.test_checkpoint, cfg.lightning_cfg)

    model.eval()
    model.l_position = np.load('exps/l_position.npy')
    model.mean = np.load(f"{cfg.result_dir}/mean.npy")
    model.std = np.load(f"{cfg.result_dir}/std.npy")
    model = model.to('cuda')

    tot_num = 0
    skeleton = SkeletonMotionTorch()
    with (torch.no_grad()):
        for i , batch in enumerate(tqdm(dataloader)):
            traj, hint, hint_mask, action, first_frame, keyframes = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            traj = traj.cuda()
            hint = hint.cuda()
            hint_mask = hint_mask.cuda()
            action = action.cuda()
            # action[0, 0] = 2.
            for key, value in first_frame.items():
                first_frame[key] = value.to('cuda')

            rotation_list, root_trans_list, keyframes =\
                model.generate_from_traj_and_hints(traj, hint, hint_mask, action, first_frame, keyframes_list=keyframes)
            full_rotations = torch.zeros((traj.shape[0], 219, 22, 6))
            full_gpos = torch.zeros((traj.shape[0], 219, 22, 3))
            full_positions = np.tile(model.l_position, (traj.shape[0], traj.shape[1], 1, 1))
            for e, ks in enumerate(keyframes): 
                rotations = rotation_list[e][:ks.shape[0]].cpu()
                root_trans = root_trans_list[e][:ks.shape[0]].cpu()

                full_rotations[e][ks] = rotations
                full_rotations[e][:ks[0]] = full_rotations[e][ks[0]]

                skeleton.from_parent_array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20], torch.tensor(np.repeat(model.l_position[None], rotations.shape[0], axis=0)).unsqueeze(0))
                skeleton.apply_pose(root_trans, rotations.unsqueeze(1))
                g_positions = skeleton.joints_global_positions

                full_gpos[e][ks] = g_positions[:, 0]
                full_gpos[e, :, 0] = traj[e, :]
                full_positions[e, :, 0] = traj[e, :].cpu().numpy()
                full_positions[e, ks, 0] = g_positions[:, 0, 0]
                full_positions[e, :ks[0], 0] = g_positions[0, 0, 0]

                mib_path = os.path.join(cfg.output_dir, f"mib_{cfg.test_path[tot_num].split('/')[-1]}")

                np.savez(mib_path, rotations=full_rotations[e].numpy(), l_positions=full_positions[e],
                         g_positions=full_gpos[e].numpy(), keyframes=ks, action=action[e].cpu().numpy())

                tot_num += 1

    # reprocess_main(cfg)

def run_mib_eval():
    import torch
    import lightning as L
    import numpy as np
    from utils.debug_util import run_cmd
    from tqdm import tqdm
    from data import make_data_loader
    from lightning_modules import load_lightning

    use_gt = False

    dataloader = make_data_loader(cfg.test_dataset, cfg.test, cfg.test_dataset_module)

    model = load_lightning(cfg.lightning_module, cfg.test_checkpoint, cfg.lightning_cfg)

    model.eval()
    model.l_position = np.load(f'{cfg.original_result_dir}/l_position.npy')
    model.mean = np.load(f"{cfg.result_dir}/mean.npy")
    model.std = np.load(f"{cfg.result_dir}/std.npy")
    # model.mean, model.std = train_dataset.calc_stats(lambda data: data['rotations'])
    model = model.to('cuda')

    full_rotations_list = []
    full_positions_list = []
    keyframes_list = []
    max_keyframes = 0
    action_list = []
    with torch.no_grad():
        for i , batch in enumerate(tqdm(dataloader)):
            gt, frame_list, action = batch[0], batch[1], batch[2]
            full_rotations = gt['rotations']
            full_positions = gt['l_positions']

            if use_gt:
                bs = action.shape[0]
                keyframes = torch.zeros((bs, len(frame_list)), device=action.device)
                keyframes[:, 0] = 9
                for i, frame in enumerate(tqdm(frame_list, leave=False)):
                    keyframes[:, i] = keyframes[:, i - 1] + frame['interval'][:, 0]

                keyframes = [ks[ks <= 218].cpu().numpy().astype(np.int32) for ks in keyframes]
                #TODO 完成这里
            else:
                action = action.to('cuda')
                for i in range(len(frame_list)):
                    for key, value in frame_list[i].items():
                        frame_list[i][key] = value.to('cuda')

                rotation_list, root_trans_list, keyframes = model.generate_keyframes_sequence(frame_list, action)
                    
                for e, ks in enumerate(keyframes): 
                    rotations = rotation_list[e][:ks.shape[0]].cpu()
                    root_trans = root_trans_list[e][:ks.shape[0]].cpu()

                    full_rotations[e][ks] = rotations

                    max_keyframes = max(max_keyframes, ks.shape[0])
                    keyframes_list.append(ks[None])
                    action_list.append(action[e].cpu().numpy()[None])

            full_rotations_list.append(full_rotations.numpy())
            full_positions_list.append(full_positions.numpy())

    full_rotations_list = np.concatenate(full_rotations_list, axis=0)
    full_positions_list = np.concatenate(full_positions_list, axis=0)
    action_list = np.concatenate(action_list, axis=0)
    keyframes_list = [ np.concatenate([ks, np.zeros((1, max_keyframes - ks.shape[1]), dtype=np.int32)], axis=-1, dtype=np.int32) for ks in keyframes_list]
    keyframes_list = np.concatenate(keyframes_list, axis=0)

    mib_eval_path = os.path.join(cfg.output_dir, f"eval_keyframe.npz")

    np.savez(mib_eval_path, rotations=full_rotations_list, l_positions=full_positions_list, keyframes=keyframes_list, action=action_list)

    # MIB evaluation (requires external MIB model, not yet open-sourced)
    if not cfg.mib_path:
        print("MIB path not set. Skipping MIB evaluation. Set cfg.mib_path to enable.")
        return

    cmd = f"python {cfg.mib_path}/eval.py"
    cmd += f" --data_path {mib_eval_path}"
    cmd += f" --output_path {cfg.output_dir}"
    cmd += f" --output_name {cfg.eval_motion_name}"
    cmd += f" --gpus {' '.join([str(gpu) for gpu in cfg.gpus])}"
    run_cmd(cmd)

    cmd = f"python scripts/run.py --type eval_motion --cfg_file {cfg.cfg_file}"
    run_cmd(cmd)

def run_eval_motion():
    import torch
    import json
    import lightning as L
    import numpy as np
    from data import make_data_loader
    from torch.utils.data import DataLoader
    from lightning_modules import load_lightning, make_trainer, make_evaluators
    from data.test_gen_dataset import EvaluateWrapper
    from utils.debug_util import copy_file_with_increment

    trainer = make_trainer(cfg)

    evaluators = make_evaluators(cfg, trainer)
    l_position = np.load(f'{cfg.original_result_dir}/l_position.npy')

    eval_wrapper = EvaluateWrapper(l_position)
    eval_path = os.path.join(cfg.output_dir, cfg.eval_motion_name)
    eval_wrapper._load_mib_data(eval_path)
    dataloader = DataLoader(eval_wrapper, batch_size=64, shuffle=False)

    res = {}
    for evaluator in evaluators:
        res.update(evaluator.evaluate(dataloader))

    for key, val in res.items():
        res[key] = float(val)

    val_path = os.path.join(cfg.output_dir, f"eval.json")
    with open(val_path, 'w') as f:
        json.dump(res, f)

    copy_file_with_increment(val_path, val_path)

    for key, value in res.items():
        print(f"{key}: {value}")

def run_eval():
    import lightning as L
    import json
    import numpy as np
    from lightning_modules import load_lightning, make_evaluators, make_trainer
    from data import make_data_loader
    from utils.debug_util import copy_file_with_increment

    trainer = make_trainer(cfg)

    test_dataloader = make_data_loader(cfg.test_dataset, cfg.test, cfg.test_dataset_module)
    
    model = load_lightning(cfg.lightning_module, cfg.test_checkpoint, cfg.lightning_cfg)

    model.eval()
    if cfg.lightning_module == 'lightning_modules.keyframe_module.KeyframeModule':
        model.l_position = np.load(f'{cfg.original_result_dir}/l_position.npy')
        model.mean = np.load(f"{cfg.result_dir}/mean.npy")
        model.std = np.load(f"{cfg.result_dir}/std.npy")
        model.evaluators = make_evaluators(cfg, trainer)

    results = trainer.validate(model, test_dataloader)

    val_path = os.path.join(cfg.output_dir, f"val.json")
    with open(val_path, 'w') as f:
        json.dump(results, f)

    copy_file_with_increment(val_path, val_path)



def run_other(type):
    pass

if __name__ == '__main__':
    cfg.trainer.logger = None
    if 'run_' + cfg.type in globals():
        globals()['run_' + cfg.type]()
    else:
        run_other(cfg.type)

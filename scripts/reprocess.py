import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import shutil
from utils.debug_util import mkdir, run_cmd, copy_file_with_increment, copy_folder_with_increment

def invalid_file(file):
    return "eval" in file or ".npz" not in file

def reprocess(data_path, target_path, cfg):
    # data_path = Path('./exps/test/')
    # target_path = Path('./exps/target/')
    mkdir(target_path)

    list_dir = os.listdir(data_path)
    # print(list_dir)
    print(f"Processing {data_path}...")

    for file in list_dir:
        if invalid_file(file):
            continue
        file_path = data_path / file
        label = file.split('_')[1][:-1]
        full_label = file[4:-4]

        target_file_path = f"{target_path / full_label / f'{cfg.exp_name}_{cfg.version}' / label}.npz"  

        mkdir(target_path / full_label / f'{cfg.exp_name}_{cfg.version}')

        copy_file_with_increment(file_path, target_file_path)


def view(target_path, skip = False, full = True):
    # target_path = Path('./exps/target/')

    list_dir = os.listdir(target_path)
    # print(list_dir)

    for sub_dir in list_dir:
        sub_path = target_path / sub_dir
        list_cases = os.listdir(sub_path)

        # print(sub_dir)
        # print(sub_path)
        # print(list_cases)

        for case in list_cases:
            case_path = sub_path / case
            list_files = os.listdir(case_path)

            for file in list_files:
                if invalid_file(file):
                    continue
                file_path = case_path / file
                file_dir = file_path.with_suffix('')


                if skip and os.path.exists(file_dir): # Skip the case that have already been captured
                    continue

                cmd = '''python ./visualization/vis.py'''
                cmd += f' --data_path {file_path}'
                if full:
                    cmd_full = cmd + f' --num {-1}'
                    run_cmd(cmd_full, bg = False)
                else:
                    pass

                cmd += ' --capture'
            
                run_cmd(cmd, bg = False)

            # print(file)
            # print(file_path)

            # break

        # break


def mib(target_path, mib_path, cfg, skip = True):
    # MIB processing (requires external MIB model, not yet open-sourced)

    list_dir = os.listdir(target_path)
    # print(list_dir)

    for sub_dir in list_dir:
        sub_path = target_path / sub_dir
        list_cases = os.listdir(sub_path)

        # print(sub_dir)
        # print(sub_path)
        # print(list_cases)

        for case in list_cases:
            case_path = sub_path / case
            list_files = os.listdir(case_path)

            for file in list_files:
                if invalid_file(file):
                    continue
                file_path = case_path / file
                # file_dir = file_path.with_suffix('')
                file_name = file_path.stem

                if skip and os.path.exists(case_path / f"{file_name}.bvh"): # Skip the case that have already been processed
                    continue

                cmd = f"python3 {os.path.join(cfg.mib_path, 'inference.py')}"
                cmd += f' --data_path {file_path}'
                cmd += f' --output_path {case_path}'
                cmd += f' --output_name {file_name}'

                cmd += f" --gpus {' '.join([str(gpu) for gpu in cfg.gpus])}"
            
                run_cmd(cmd, bg = False)

            # break

        # break

def reprocess_main(cfg):
    data_path = Path(cfg.output_dir)
    target_path = Path(cfg.original_result_dir, 'target')
    if cfg.reprocess and os.path.exists(data_path):
        print("Reprocessing data...")
        reprocess(data_path, target_path, cfg)

        copy_folder_with_increment(data_path, data_path)
        shutil.rmtree(data_path)

    if cfg.view and os.path.exists(target_path):
        print("Viewing data...")
        view(target_path, False)
    
    if cfg.mib and os.path.exists(target_path):
        print("MIBing data...")
        mib(target_path, cfg.mib_path, cfg, True)

## Copyright (c) Meta Platforms, Inc. and affiliates
from memory_profiler import profile

import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader
from collections import defaultdict
import random
import glob
import gc
from equi_diffpo.dataset.readh5 import ReadH5Files
import time
import random
import copy
import torchvision.transforms as transforms
import cv2
from PIL import Image
import psutil
import os
from torch.utils.data import Sampler, DistributedSampler

# process = psutil.Process(os.getpid())
# print_memory_info(process, 1)
def print_memory_info(process, idx):
    used_bytes = process.memory_info().rss
    used_MB = used_bytes / (1024*1024)
    print(f"{idx}. used_MB: {used_MB}")

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, robot_infor, norm_stats, episode_ids, episode_len, chunk_size, rank=None, use_data_aug=False, act_norm_class='norm2', use_raw_lang=False, use_depth_image=False, exp_type='franka_3rgb', tg_mode='mode1'):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.use_depth_image = use_depth_image
        self.min_depth = 0.25
        self.max_depth = 1.25
        self.robot_infor = robot_infor
        self.exp_type = exp_type
        self.tg_mode = tg_mode

        if exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb']:
            self.qpos_arm_key = 'puppet'
            self.action_arm_key = 'puppet'
            self.ctl_elem_key = 'joint_position'
        elif exp_type in ['songling_3rgb']:
            self.qpos_arm_key = 'puppet'
            self.action_arm_key = 'master'
            self.ctl_elem_key = ['joint_position_left', 'joint_position_right']
        elif exp_type in ['simulation_4rgb']:
            self.qpos_arm_key = 'franka'
            self.action_arm_key = 'franka'
            self.ctl_elem_key = 'joint_position'
        elif exp_type in ['tiangong_1rgb']: # h5_new_tiangong
            if self.tg_mode == 'mode5':
                self.qpos_arm_key = 'puppet'
                self.action_arm_key = 'master'
                self.ctl_elem_key = 'joint_position'
            elif self.tg_mode == 'mode6':
                self.qpos_arm_key = 'master'
                self.action_arm_key = 'master'
                self.ctl_elem_key = 'joint_position'
            elif self.tg_mode == 'mode7':
                self.qpos_arm_key = 'puppet'
                self.action_arm_key = 'puppet'
                self.ctl_elem_key = 'joint_position'
            elif self.tg_mode == 'mode8':
                self.qpos_arm_key = 'master'
                self.action_arm_key = 'puppet'
                self.ctl_elem_key = 'joint_position'
            else: # h5_tiangong
                self.qpos_arm_key = 'puppet'
                self.action_arm_key = 'puppet'
                self.ctl_elem_key = ['end_effector', 'joint_position']

        self.read_h5files = ReadH5Files(self.robot_infor)

        self.path2string_dict = dict()
        for dataset_path in self.dataset_path_list:
            with h5py.File(dataset_path, 'r') as root:
                if 'language_raw' in root.keys():
                    lang_raw_utf = root['language_raw'][0].decode('utf-8')
                else:
                    lang_raw_utf = 'None'
                self.path2string_dict[dataset_path] = lang_raw_utf
            root.close()
        
        print(f"len episode_ids: {len(self.episode_ids)}")
        print(f"first 10 episode_ids: {self.episode_ids[:10]}")
        print(f"len episode_len: {len(self.episode_len)}")
        print(f"sum episode_len: {sum(self.episode_len)}")
        print(f"len dataset_path_list: {len(self.dataset_path_list)}")
        # print(f"total dataset cumulative_len: {self.cumulative_len}")

        self.max_episode_len = max(episode_len)
        self.resize_images = True
        print('Initializing resize transformations')

        # (w, h) for cv2.resize
        self.new_size = (640, 480)

        self.augment_images = use_data_aug
        self.aug_trans = None
        print(f"cur dataset augment_images: {self.augment_images}")
        # augmentation
        if self.augment_images is True:
            print('Initializing transformations')
            # (h, w) for transforms aug
            original_size = (480, 640)
            ratio = 0.95
            self.aug_trans = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize(original_size, antialias=True),
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
            ]

        self.rank = rank
        self.transformations = None
        self.act_norm_class = act_norm_class
        self.use_raw_lang = use_raw_lang

        self.tmp_cnt = 0

        self.__getitem__(0) # initialize self.is_sim and self.transformations
    
    def __len__(self):
        return 1000 * sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        # print(f"index: {index}")
        # print(f"self.cumulative_len: {self.cumulative_len}")
        # print(f"episode_id: {episode_id}")
        # print(f"start_ts: {start_ts}")
        return episode_id, start_ts

    def __getitem__(self, index):
        index = index % sum(self.episode_len)  # Ensure index wraps around
        episode_id, start_ts = self._locate_transition(index)

        dataset_path = self.dataset_path_list[episode_id]
        
        image_dict, control_dict, _, _, _ = self.read_h5files.execute(dataset_path, camera_frame=start_ts, use_depth_image=self.use_depth_image)

        if self.use_raw_lang:
            lang_raw = self.path2string_dict[dataset_path]

        if isinstance(self.ctl_elem_key, list):
            qpos_list = []
            action_list = []
            for ele in self.ctl_elem_key:
                # 12, 14
                # ['end_effector', 'joint_position']
                if self.exp_type in ['tiangong_1rgb'] and ele == 'end_effector':
                    # print(f"tg_mode: {self.tg_mode}")
                    if self.tg_mode == 'mode1':
                        cur_qpos = control_dict[self.qpos_arm_key][ele][:]
                        cur_action = control_dict[self.action_arm_key][ele][:]
                    elif self.tg_mode == 'mode2':
                        cur_qpos = control_dict[self.qpos_arm_key][ele][:]
                        cur_left_action = control_dict[self.action_arm_key][ele][:, [3,4]]
                        cur_right_action = control_dict[self.action_arm_key][ele][:, [9,10]]
                        # print(f"cur_qpos: {cur_qpos[1]}")
                        # print(f"cur_left_action: {cur_left_action[1]}")
                        # print(f"cur_right_action: {cur_right_action[1]}")
                        cur_action = np.concatenate((cur_left_action, cur_right_action), axis=1)
                        # print(f"cur_action: {cur_action[1]}")
                    elif self.tg_mode == 'mode3':
                        cur_left_qpos = control_dict[self.qpos_arm_key][ele][:, [3,4]]
                        cur_right_qpos = control_dict[self.qpos_arm_key][ele][:, [9,10]]
                        cur_left_action = control_dict[self.action_arm_key][ele][:, [3,4]]
                        cur_right_action = control_dict[self.action_arm_key][ele][:, [9,10]]
                        # print(f"cur_left_qpos: {cur_left_qpos[1]}")
                        # print(f"cur_right_qpos: {cur_right_qpos[1]}")
                        # print(f"cur_left_action: {cur_left_action[1]}")
                        # print(f"cur_right_action: {cur_right_action[1]}")
                        cur_qpos = np.concatenate((cur_left_qpos, cur_right_qpos), axis=1)
                        cur_action = np.concatenate((cur_left_action, cur_right_action), axis=1)
                        # print(f"cur_qpos: {cur_qpos[1]}")
                        # print(f"cur_action: {cur_action[1]}")
                else:
                    cur_qpos = control_dict[self.qpos_arm_key][ele][:]
                    cur_action = control_dict[self.action_arm_key][ele][:]
                qpos_list.append(cur_qpos)
                action_list.append(cur_action)
            qpos = np.concatenate(qpos_list, axis=-1)
            action = np.concatenate(action_list, axis=-1)
        else:
            qpos = control_dict[self.qpos_arm_key][self.ctl_elem_key][:]
            action = control_dict[self.action_arm_key][self.ctl_elem_key][:]

        original_action_shape = action.shape
        # print(f"0. original_action_shape: {original_action_shape}")
        episode_len = original_action_shape[0]
        # get observation at start_ts only
        # qpos = action[start_ts]
        
        cur_qpos = qpos[start_ts]
        prev_qpos = qpos[start_ts - 1] if start_ts > 0 else np.zeros_like(cur_qpos)
        combined_qpos = np.stack((cur_qpos, prev_qpos), axis=0)

        # for real-world exp
        action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
        action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned
        # print(f"0. action size: {action.shape}")

        if original_action_shape[0] < self.chunk_size:
            original_action_shape = list(original_action_shape)
            original_action_shape[0] = self.chunk_size
            original_action_shape = tuple(original_action_shape)
        
        if episode_len < self.chunk_size:
            episode_len = self.chunk_size

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        padded_action = padded_action[:self.chunk_size]
        # print(f"0. padded_action size: {padded_action.shape}")

        is_pad = is_pad[:self.chunk_size]
        # new axis for different cameras
        all_cam_images = []
        all_cam_depths = []

        # tmp_dir = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/difffusion_input_img'
        # img_dir = os.path.join(tmp_dir, f"{self.exp_type}")
        # os.makedirs(img_dir, exist_ok=True)

        for cam_name in self.robot_infor['camera_names']:
            
            cur_img = image_dict[self.robot_infor['camera_sensors'][0]][cam_name]['current']
            prev_img = image_dict[self.robot_infor['camera_sensors'][0]][cam_name]['previous']
            if prev_img is None:
                prev_img = np.zeros_like(cur_img)
            # print(f'1 cur_img size: {cur_img.shape}')
            # todo
            if self.resize_images:
                cur_img = cv2.resize(cur_img, dsize=self.new_size)
                prev_img = cv2.resize(prev_img, dsize=self.new_size)
                # 2 cur_img size: (480, 640, 3)
                # print(f'2 cur_img size: {cur_img.shape}')
            if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'simulation_4rgb']:
                cur_img = cur_img[:, :, ::-1]
                prev_img = prev_img[:, :, ::-1]

            combined_img = np.stack((cur_img, prev_img), axis=0)

            all_cam_images.append(combined_img) # rgb

            # print_memory_info(process, 6)
            if self.use_depth_image:
                depth_image = image_dict[self.robot_infor['camera_sensors'][1]][cam_name] / 1000 # m
                depth_image_filtered = np.zeros_like(depth_image)
                depth_image_filtered[(depth_image >= self.min_depth) & (depth_image <= self.max_depth)] = depth_image[
                    (depth_image >= self.min_depth) & (depth_image <= self.max_depth)]
                if self.resize_images:
                    depth_image_filtered = cv2.resize(depth_image_filtered, dsize=self.new_size)
                    # print(f'2 depth_image_filtered size: {depth_image_filtered.shape}')
                all_cam_depths.append(depth_image_filtered)
            else:
                dummy_depth = np.zeros((480, 640))
                all_cam_depths.append(dummy_depth)

        # print_memory_info(process, 7)

        all_cam_images = np.stack(all_cam_images, axis=0)
        all_cam_depths = np.stack(all_cam_depths, axis=0)
        all_cam_images = (all_cam_images / 255.0).astype(np.float32)
        all_cam_images = all_cam_images * 2 - 1
        image_data = torch.from_numpy(all_cam_images)
        

        all_cam_depths = all_cam_depths.astype(np.float32)
        #depth_data = torch.from_numpy(all_cam_depths)

        #qpos_data = torch.from_numpy(qpos).float()
        qpos_data = torch.from_numpy(combined_qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        image_data = torch.einsum('k t h w c -> k t c h w', image_data)
        # image_data size: torch.Size([2, 3, 480, 640])
        # depth_data size: torch.Size([2, 480, 640])
        #print(f"image_data size: {image_data.size()}")
        # print(f"depth_data size: {depth_data.size()}")

        # print_memory_info(process, 8)

        if self.augment_images:
            for transform in self.aug_trans:
                image_data = transform(image_data)
        
        # normalize image and change dtype to float
        # image_data = image_data / 255.0

        if self.act_norm_class == 'norm1':
            # normalize to [-1, 1]
            action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
        elif self.act_norm_class == 'norm2':
            # normalize to mean 0 std 1
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        qpos_data = ((qpos_data - self.norm_stats["qpos_min"]) / (self.norm_stats["qpos_max"] - self.norm_stats["qpos_min"])) * 2 - 1

        # lang_embed = control_dict['language_distilbert']
        #lang_embed = torch.zeros(1)

        # print_memory_info(process, 9)
        # print(f"=============+++++++++++")
        # image_data size: torch.Size([3, 3, 480, 640])
        # depth_data size: torch.Size([3, 480, 640])
        # qpos_data size: torch.Size([8])
        # action_data size: torch.Size([50, 8])
        # is_pad size: torch.Size([50])
        # print(f"image_data size: {image_data.size()}")
        # print(f"depth_data size: {depth_data.size()}")
        # print(f"qpos_data size: {qpos_data.size()}")
        # print(f"action_data size: {action_data.size()}")
        # print(f"is_pad size: {is_pad.size()}")
        # print(f"lang_embed size: {lang_embed.size()}")

        # if self.use_raw_lang:
        #     return image_data, depth_data, qpos_data, action_data, is_pad, lang_embed, lang_raw
        # return image_data, depth_data, qpos_data, action_data, is_pad, lang_embed
        torch_data = {
            'obs':{
                'rgb': image_data,
                'low_dim': qpos_data},
            'action': action_data
        }
        return torch_data
        # return image_data, depth_data, qpos_data, action_data, is_pad


def get_files(dataset_dir, robot_infor):
    read_h5files = ReadH5Files(robot_infor)
    # dataset_dir = os.path.join(dataset_dir, 'success_episodes')
    files = []
    for trajectory_id in sorted(os.listdir(dataset_dir)):
        trajectory_dir = os.path.join(dataset_dir, trajectory_id)
        file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
        try:
            _, control_dict, base_dict, _, is_compress = read_h5files.execute(file_path, camera_frame=2)
            files.append(file_path)
        except Exception as e:
            print(f"Open {file_path} fail!")
            print(e)
    return files

def get_norm_stats(train_dataset_path_list, val_dataset_path_list, robot_infor, exp_type, tg_mode):
    read_h5files = ReadH5Files(robot_infor)

    all_qpos_data = []
    all_action_data = []
    val_episode_len = []
    train_episode_len = []

    if exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb']:
        qpos_arm_key = 'puppet'
        action_arm_key = 'puppet'
        ctl_elem_key = 'joint_position'
    elif exp_type in ['songling_3rgb']:
        qpos_arm_key = 'puppet'
        action_arm_key = 'master'
        ctl_elem_key = ['joint_position_left', 'joint_position_right']
    elif exp_type in ['simulation_4rgb']:
        qpos_arm_key = 'franka'
        action_arm_key = 'franka'
        ctl_elem_key = 'joint_position'
    elif exp_type in ['tiangong_1rgb']:
        if tg_mode == 'mode5':
            qpos_arm_key = 'puppet'
            action_arm_key = 'master'
            ctl_elem_key = 'joint_position'
        elif tg_mode == 'mode6':
            qpos_arm_key = 'master'
            action_arm_key = 'master'
            ctl_elem_key = 'joint_position'
        elif tg_mode == 'mode7':
            qpos_arm_key = 'puppet'
            action_arm_key = 'puppet'
            ctl_elem_key = 'joint_position'
        elif tg_mode == 'mode8':
            qpos_arm_key = 'master'
            action_arm_key = 'puppet'
            ctl_elem_key = 'joint_position'
        else:
            qpos_arm_key = 'puppet'
            action_arm_key = 'puppet'
            ctl_elem_key = ['end_effector', 'joint_position']

    for list_id, cur_dataset_path_list in enumerate([train_dataset_path_list, val_dataset_path_list]):
        cur_episode_len = []
        for id, dataset_path in enumerate(cur_dataset_path_list):
            try:
                _, control_dict, base_dict, _, is_compress = read_h5files.execute(dataset_path, camera_frame=0)

                if isinstance(ctl_elem_key, list):
                    qpos_list = []
                    action_list = []
                    for ele in ctl_elem_key:
                        if exp_type in ['tiangong_1rgb'] and ele == 'end_effector':
                            # print(f"tg_mode: {tg_mode}")
                            if tg_mode == 'mode1':
                                cur_qpos = control_dict[qpos_arm_key][ele][:]
                                cur_action = control_dict[action_arm_key][ele][:]
                            elif tg_mode == 'mode2':
                                cur_qpos = control_dict[qpos_arm_key][ele][:]
                                cur_left_action = control_dict[action_arm_key][ele][:, [3,4]]
                                cur_right_action = control_dict[action_arm_key][ele][:, [9,10]]
                                # print(f"cur_qpos: {cur_qpos[1]}")
                                # print(f"cur_left_action: {cur_left_action[1]}")
                                # print(f"cur_right_action: {cur_right_action[1]}")
                                cur_action = np.concatenate((cur_left_action, cur_right_action), axis=1)
                                # print(f"cur_action: {cur_action[1]}")

                            elif tg_mode == 'mode3':
                                cur_left_qpos = control_dict[qpos_arm_key][ele][:, [3,4]]
                                cur_right_qpos = control_dict[qpos_arm_key][ele][:, [9,10]]
                                cur_left_action = control_dict[action_arm_key][ele][:, [3,4]]
                                cur_right_action = control_dict[action_arm_key][ele][:, [9,10]]
                                # print(f"cur_left_qpos: {cur_left_qpos[1]}")
                                # print(f"cur_right_qpos: {cur_right_qpos[1]}")
                                # print(f"cur_left_action: {cur_left_action[1]}")
                                # print(f"cur_right_action: {cur_right_action[1]}")
                                cur_qpos = np.concatenate((cur_left_qpos, cur_right_qpos), axis=1)
                                cur_action = np.concatenate((cur_left_action, cur_right_action), axis=1)
                                # print(f"cur_qpos: {cur_qpos[1]}")
                                # print(f"cur_action: {cur_action[1]}")
                        else:
                            cur_qpos = control_dict[qpos_arm_key][ele][:]
                            cur_action = control_dict[action_arm_key][ele][:]
                        qpos_list.append(cur_qpos)
                        action_list.append(cur_action)
                    qpos = np.concatenate(qpos_list, axis=-1)
                    action = np.concatenate(action_list, axis=-1)
                else:
                    qpos = control_dict[qpos_arm_key][ctl_elem_key][:]
                    action = control_dict[action_arm_key][ctl_elem_key][:]

                # puppet_joint_position = control_dict['puppet']['joint_position'][:]
                
                # action = puppet_joint_position

                # for frame_id in range(action.shape[0]):
                #     trial_indices.append((id, frame_id))
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))
                cur_episode_len.append(len(action))
            except Exception as e:
                print(e)
                print('filename:',dataset_path)
        if list_id == 0:
            train_episode_len = cur_episode_len
        else:
            val_episode_len = cur_episode_len

    num_episodes = len(all_action_data)

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)  # 该任务所有的轨迹帧按行合并[每条轨迹长度*轨迹数，x]

    # normalize action data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    qpos_min = all_qpos_data.min(dim=0).values.float()
    qpos_max = all_qpos_data.max(dim=0).values.float()

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(), "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps, "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(), "qpos_min": qpos_min.numpy() - eps, "qpos_max": qpos_max.numpy() + eps}

    return stats, train_episode_len, val_episode_len

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    # print(f"BatchSampler episode_len_l: {episode_len_l}")
    # print(f"BatchSampler len episode_len_l: {len(episode_len_l)}")
    # print(f"BatchSampler sum_dataset_len_l: {sum_dataset_len_l}")

    def batch_generator():
        while True:
            batch = []
            for _ in range(batch_size):
                task_idx = np.random.choice(len(episode_len_l), p=sample_probs)
                step_idx = np.random.randint(sum_dataset_len_l[task_idx], sum_dataset_len_l[task_idx + 1])
                batch.append(step_idx)
            yield batch

    return batch_generator()


class DistributedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, episode_len_l, sample_weights, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.episode_len_l = episode_len_l
        self.sample_weights = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        self.sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
        self.sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])

        self.dist_sampler = DistributedSampler(self.dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle)

    def __iter__(self):
        batch = []
        for idx in self.dist_sampler:
            # Assume episode index selection and within-episode index calculation is done here
            # For simplicity, we're just incrementing idx to mimic selecting indices
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch  # Yield the last batch if it's not empty
    

    def __len__(self):
        return len(self.dist_sampler) // self.batch_size
    

def load_data(dataset_dir_l, robot_infor, batch_size_train, batch_size_val, chunk_size, use_depth_image=False, sample_weights=None, rank=None, use_data_aug=False, act_norm_class='norm2', use_raw_lang=False, name_filter=None, exp_type='franka_3rgb', logger=None, tg_mode='mode1'):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]

    # [[task1_epi1, task1_epi2], [task2_epi1, task2_epi2]]
    train_dataset_path_list_list = []
    val_dataset_path_list_list = []
    # logger.info(f"load_data is called!")
    for dataset_dir in dataset_dir_l:
        # hdf5_files_list = find_all_hdf5(dataset_dir) 
        # obtain train test split
        succ_dataset_dir = os.path.join(dataset_dir, 'success_episodes')
        train_dir = os.path.join(succ_dataset_dir, 'train')
        val_dir = os.path.join(succ_dataset_dir, 'val')
        hdf5_train_files_list = get_files(train_dir, robot_infor)
        hdf5_val_files_list = get_files(val_dir, robot_infor)
        train_dataset_path_list_list.append(hdf5_train_files_list)
        val_dataset_path_list_list.append(hdf5_val_files_list)

        # hdf5_files_list = get_files(dataset_dir, robot_infor)
        # dataset_path_list_list.append(hdf5_files_list)

    train_num_episodes_l = [len(cur_dataset_path_list) for cur_dataset_path_list in train_dataset_path_list_list]
    train_num_episodes_cumsum = np.cumsum(train_num_episodes_l)
    val_num_episodes_l = [len(cur_dataset_path_list) for cur_dataset_path_list in val_dataset_path_list_list]
    val_num_episodes_cumsum = np.cumsum(val_num_episodes_l)
    if rank == 0 or rank is None:
        print(f"train_num_episodes_cumsum: {train_num_episodes_cumsum}")
        print(f"val_num_episodes_cumsum: {val_num_episodes_cumsum}")
        # logger.info(f"train_num_episodes_cumsum: {train_num_episodes_cumsum}")
        # logger.info(f"val_num_episodes_cumsum: {val_num_episodes_cumsum}")
    name_filter = lambda n: True
    train_dataset_path_fla_list = flatten_list(train_dataset_path_list_list)
    train_dataset_path_fla_list = [n for n in train_dataset_path_fla_list if name_filter(n)]
    val_dataset_path_fla_list = flatten_list(val_dataset_path_list_list)
    val_dataset_path_fla_list = [n for n in val_dataset_path_fla_list if name_filter(n)]

    train_episode_ids_l = []
    cur_num_episodes = 0
    for task_idx, _ in enumerate(train_dataset_path_list_list):
        # num episodes of task idx
        num_episodes_i = len(train_dataset_path_list_list[task_idx])
        if rank == 0 or rank is None:
            print(f"Train: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
            # logger.info(f"Train: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
        shuffled_episode_ids_i = np.random.permutation(num_episodes_i)
        train_episode_ids_l.append(shuffled_episode_ids_i)
        for i in range(num_episodes_i):
            shuffled_episode_ids_i[i] += cur_num_episodes
        cur_num_episodes += num_episodes_i

    val_episode_ids_l = []
    cur_num_episodes = 0
    for task_idx, _ in enumerate(val_dataset_path_list_list):
        # num episodes of task idx
        num_episodes_i = len(val_dataset_path_list_list[task_idx])
        if rank == 0 or rank is None:
            print(f"Val: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
            # logger.info(f"Val: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
        shuffled_episode_ids_i = np.random.permutation(num_episodes_i)
        val_episode_ids_l.append(shuffled_episode_ids_i)
        for i in range(num_episodes_i):
            shuffled_episode_ids_i[i] += cur_num_episodes
        cur_num_episodes += num_episodes_i
    
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    if rank == 0 or rank is None:
        # train_episode_ids: list []
        # val_episode_ids: list []
        print(f"train_episode_ids: {train_episode_ids}")
        print(f"val_episode_ids: {val_episode_ids}")
        print(f"len train_episode_ids: {len(train_episode_ids)}")
        print(f"len val_episode_ids: {len(val_episode_ids)}")
        # logger.info(f"train_episode_ids: {train_episode_ids}")
        # logger.info(f"val_episode_ids: {val_episode_ids}")
        # logger.info(f"len train_episode_ids: {len(train_episode_ids)}")
        # logger.info(f"len val_episode_ids: {len(val_episode_ids)}")


    if rank == 0 or rank is None:
        print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
        # logger.info(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    
    norm_stats, train_episode_len, val_episode_len = get_norm_stats(train_dataset_path_fla_list, val_dataset_path_fla_list, robot_infor, exp_type, tg_mode)
    if rank == 0 or rank is None:
        print(f"norm_stats: {norm_stats}")
        # all_episode_len: list []
        # print(f"len all_episode_len: {len(all_episode_len)}")
        print(f"len train_episode_len: {len(train_episode_len)}")
        print(f"len val_episode_len: {len(val_episode_len)}")
        # logger.info(f"norm_stats: {norm_stats}")
        # logger.info(f"len train_episode_len: {len(train_episode_len)}")
        # logger.info(f"len val_episode_len: {len(val_episode_len)}")

    train_episode_len_l = []
    for cur_task_train_episode_ids in train_episode_ids_l:
        cur_task_train_episode_len_l = []
        for i in cur_task_train_episode_ids:
            cur_task_train_episode_len_l.append(train_episode_len[i])
        train_episode_len_l.append(cur_task_train_episode_len_l)
    # if rank == 0 or rank is None:
    #     # train_episode_len_l: list of list [[task1_ep1, task1_ep2], [task2_ep1, task2_ep2]]
    #     print(f"train_episode_len_l: {train_episode_len_l}")
    
    val_episode_len_l = []
    for cur_task_val_episode_ids in val_episode_ids_l:
        cur_task_val_episode_len_l = []
        for i in cur_task_val_episode_ids:
            cur_task_val_episode_len_l.append(val_episode_len[i])
        val_episode_len_l.append(cur_task_val_episode_len_l)
    # if rank == 0 or rank is None:
    #     # val_episode_len_l: list of list [[task1_ep1, task1_ep2], [task2_ep1, task2_ep2]]
    #     print(f"val_episode_len_l: {val_episode_len_l}")

    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if rank == 0 or rank is None:
        # train_episode_len: list []
        # val_episode_len: list []
        # print(f"train_episode_len: {train_episode_len}")
        # print(f"val_episode_len: {val_episode_len}")
        print(f'Norm stats from: {dataset_dir_l}')
        # logger.info(f'Norm stats from: {dataset_dir_l}')
    
    
    if torch.cuda.device_count() == 1:
        #batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
        #batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)
        batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
        batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)
        # construct dataset and dataloader
        train_dataset = EpisodicDataset(train_dataset_path_fla_list, robot_infor, norm_stats, train_episode_ids, train_episode_len, chunk_size, rank=rank, use_data_aug=use_data_aug, act_norm_class=act_norm_class, use_raw_lang=use_raw_lang, use_depth_image=use_depth_image, exp_type=exp_type, tg_mode=tg_mode)
        val_dataset = EpisodicDataset(val_dataset_path_fla_list, robot_infor, norm_stats, val_episode_ids, val_episode_len, chunk_size, rank=rank, use_data_aug=use_data_aug, act_norm_class=act_norm_class, use_raw_lang=use_raw_lang, use_depth_image=use_depth_image, exp_type=exp_type, tg_mode=tg_mode)

        # train_num_workers = 0
        train_num_workers = 4 #4 #4 #16
        # val_num_workers = 0
        val_num_workers = 8 #4 #4 #16
        print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
        # logger.info(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
        # pin_memory=True
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
        val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)
    else:
        train_dataset = EpisodicDataset(train_dataset_path_fla_list, robot_infor, norm_stats, train_episode_ids, train_episode_len, chunk_size, rank=rank, use_data_aug=use_data_aug, act_norm_class=act_norm_class, use_raw_lang=use_raw_lang, use_depth_image=use_depth_image,exp_type=exp_type)
        val_dataset = EpisodicDataset(val_dataset_path_fla_list, robot_infor, norm_stats, val_episode_ids, val_episode_len, chunk_size, rank=rank, use_data_aug=use_data_aug, act_norm_class=act_norm_class, use_raw_lang=use_raw_lang, use_depth_image=use_depth_image,exp_type=exp_type)
        num_replicas = torch.cuda.device_count()
        # val_num_workers = 8 if train_dataset.augment_images else 2
        train_num_workers = 8 #0 #4 #0 #for local #8 #0 #2
        val_num_workers = 8 #0 #4 #0 #for local#8 #0 #2

        if rank == 0 or rank is None:
            logger.info(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')

        batch_sampler_train = DistributedBatchSampler(train_dataset, batch_size_train, train_episode_len, sample_weights, num_replicas=num_replicas, rank=rank)
        batch_sampler_val = DistributedBatchSampler(val_dataset, batch_size_val, val_episode_len, None, num_replicas=num_replicas, rank=rank)

        # prefetch_factor=2
        train_dataloader = DataLoader(
            train_dataset, 
            batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers)
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers)
        
    return train_dataloader, val_dataloader, norm_stats


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    # print('base_action1:',base_action)
    base_action = smooth_base_action(base_action)
    return base_action
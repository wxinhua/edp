import hydra
import os
import torch
import dill
import sys
from omegaconf import OmegaConf
import pathlib
import argparse
import threading
import cv2
import pickle
import numpy as np
from pynput import keyboard
from equi_diffpo.workspace.base_workspace import BaseWorkspace
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply

class InferVLAIL():
    def __init__(self, args=None):
        self.args = args
        model_dir = self.args['model_dir']
        if not os.path.isdir(self.args['model_dir']):
            print(f"ckpt_dir {ckpt_dir} does not exist!!")
        ckpt_dir = os.path.join(model_dir, self.args['ckpt_name'])
        
        # 打印检查点路径
        print(f"Loading checkpoint from: {ckpt_dir}")
        
        try:
            payload = torch.load(open(ckpt_dir, 'rb'), pickle_module=dill)
            print("Payload loaded successfully.")
        except Exception as e:
            print(f"Error loading payload: {e}")
            return
        
        # 打印payload的内容
        print("Payload keys:", payload.keys())
        
        self.cfg = payload['cfg']
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        
        # try:
        #     workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        # except Exception as e:
        #     print(f"Error loading workspace payload: {e}")
        #     return
        
        
        self.model: BaseImagePolicy = workspace.model
        if self.cfg.training.use_ema:
            self.model = workspace.ema_model

        try:
            self.model.load_state_dict(payload['state_dicts'], strict=False)
        except RuntimeError as e:
            print(f"Error loading model state dict: {e}")


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(self.device)
        self.exp_type = self.cfg.exp_type

        stats_path = os.path.join(model_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.dataset_stats = pickle.load(f)

        print("loaded model")

    def on_press(self, key):
        global preparing
        try:
            if key == keyboard.Key.enter:
                preparing = False

        except AttributeError:
            pass

    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def print_color(self, *args, color=None, attrs=(), **kwargs):
        import termcolor

        if len(args) > 0:
            args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
        print(*args, **kwargs)

    def get_image(self, obs, show_img=False):
        # (w, h) for cv2.resize
        img_new_size = (640, 480) #(480, 640)
        all_cam_images = []

        for cam_name in self.cfg['robot_infor']['camera_names']:
            if self.exp_type == 'tiangong_1rgb':
                curr_image = obs['images'][cam_name]
            else:
                # for franka_3rgb, ur_1rgb
                curr_image = obs['images'][cam_name]
            print(f'{cam_name} curr_image:',curr_image.shape)
            # rgb_image_encode = cv2.imencode(".jpg", curr_image)[1]
            rgb_image_encode = curr_image
            if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'simulation_4rgb', 'tiangong_1rgb']:
                curr_image = cv2.imdecode(rgb_image_encode, cv2.IMREAD_COLOR)
            else:
                curr_image = rgb_image_encode

            if show_img:
                rgb_image = curr_image[:, :, ::-1]
                cv2.imshow(f"{cam_name} image", rgb_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'simulation_4rgb']:
                curr_image = curr_image[:, :, ::-1]

            # img_dir = '/home/ps/wk/tmp'
            # if self.tmp_cnt < 10:
            #     img = Image.fromarray((curr_image).astype(np.uint8))
            #     img.save(os.path.join(img_dir, str(self.tmp_cnt)+'_rgb.png'))
            # self.tmp_cnt += 1            

            # curr_image = rearrange(curr_image, 'h w c -> c h w')
            # curr_image: (3, 480, 640)
            # print('curr_image:',curr_image.shape)
            all_cam_images.append(curr_image)
        all_cam_images = np.stack(all_cam_images, axis=0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        all_cam_images = (all_cam_images / 255.0).astype(np.float32)
        all_cam_images = all_cam_images * 2 - 1

        return all_cam_images
    
    def get_qpos(self, obs):
        # incros/incros/robot_env/franka_env.py
        # incros/incros/robot_env/ur_env.py
        if self.exp_type == 'tiangong_1rgb':
            # [self.left_jpos, self.left_hand, self.right_jpos, self.right_hand]
            qpos = obs['qpos']
            left_jpos = qpos[:7]
            left_hand_jpos = qpos[7:13]
            right_jpos = qpos[13:20]
            right_hand_jpos = qpos[20:26]
            # mode1: input 26, output 26;
            # mode2: input 26, output 14+4=18: Index Finger食指, Thumb拇指
            # mode3: input 14+4=18, output 14+4=18: Index Finger食指, Thumb拇指
            # mode4: input 14, output 14: only contron arm
            if self.args['tg_mode'] == 'mode1':
                qpos = np.concatenate((left_hand_jpos, right_hand_jpos, left_jpos, right_jpos))
            elif self.args['tg_mode'] == 'mode2':
                qpos = np.concatenate((left_hand_jpos, right_hand_jpos, left_jpos, right_jpos))
            elif self.args['tg_mode'] == 'mode3':
                qpos = np.concatenate((left_hand_jpos[[3,4]], right_hand_jpos[3:4], left_jpos, right_jpos))
            elif self.args['tg_mode'] == 'mode4':
                qpos = np.concatenate((left_jpos, right_jpos))
            elif self.args['tg_mode'] in ['mode5', 'mode6', 'mode7', 'mode8']:
                ### 16 dim joint_position
                qpos = obs['qpos']
        else:
            # for franka_3rgb, ur_1rgb
            qpos = obs['qpos']
        return qpos
    
    def qpos_pre_process(self, robot_state_value, stats):
        tmp = robot_state_value
        tmp = ((tmp - stats["qpos_min"]) / (stats["qpos_max"] - stats["qpos_min"])) * 2 - 1
        return tmp
    
    def action_post_process(self, action, stats):
        action = ((action + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        return action

    def get_model_input(self, timestep, obs_stack):
        if timestep == 0:
            cur_image = self.get_image(obs_stack[1])
            prev_image = np.zeros_like(cur_image)
            input_image = np.stack((cur_image, prev_image), axis=0)
            cur_qpos = self.get_qpos(obs_stack[1])
            cur_qpos = self.qpos_pre_process(cur_qpos, self.dataset_stats)
            prev_qpos = np.zeros_like(cur_qpos)
            input_qpos = np.stack((cur_qpos, prev_qpos), axis=0)
        else:
            cur_image = self.get_image(obs_stack[1])
            prev_image = self.get_image(obs_stack[0])
            input_image = np.stack((cur_image, prev_image), axis=0)
            cur_qpos = self.get_qpos(obs_stack[1])
            cur_qpos = self.qpos_pre_process(cur_qpos, self.dataset_stats)
            prev_qpos = self.get_qpos(obs_stack[0])
            prev_qpos = self.qpos_pre_process(prev_qpos, self.dataset_stats)
            input_qpos = np.stack((cur_qpos, prev_qpos), axis=0)

        qpos_data = torch.from_numpy(input_qpos).float()
        image_data = torch.from_numpy(input_image)
        # k is the number of camera
        image_data = torch.einsum('k t h w c -> k t c h w', image_data)
        # qpos_data = qpos_data.unsqueeze(0)
        # image_data = image_data.unsqueeze(0)
        qpos_data = qpos_data.cuda()
        image_data = image_data.cuda()
        obs_dict = {'obs':{'rgb': image_data,'low_dim': qpos_data}}

        return obs_dict

    def execute(self):
        ###
        listener_thread = threading.Thread(target=self.start_keyboard_listener, daemon=True)
        listener_thread.start()
        ###
        print("Going to start position")

        self.print_color("\nReady for Start 🚀🚀🚀", color="green", attrs=("bold",))
        os.system("espeak start")

        if self.exp_type == 'franka_3rgb':
            sys.path.append('/home/ps/Dev/inrocs/')
            from robot_env.franka_env import robot_env
        elif self.exp_type == 'ur_1rgb':
            sys.path.append("/home/ps/work_sapce_hqr/inrocs")
            from robot_env.ur_env import robot_env
        elif self.exp_type == 'tiangong_1rgb':
            # sys.path.append('/home/ps/code_lei/inrocs')
            # # os.environ["ROS_MASTER_URI"] = "http://192.168.41.1:11311"
            # os.environ["ROS_MASTER_URI"] = "http://192.168.41.13:11311"
            # os.environ["ROS_IP"] = "192.168.41.55"
            # from robot_env.tiangong_env_5hz_wk import  TiangongEnv
            # robot_env = TiangongEnv(
            #     camera_topic="/camera/color/image_raw",
            #     left_arm_ctrl_topic="/human_arm_ctrl_left",
            #     right_arm_ctrl_topic="/human_arm_ctrl_right",
            #     left_arm_state_topic="/human_arm_state_left",
            #     right_arm_state_topic="/human_arm_state_right",
            #     left_hand_topic="/inspire_hand/ctrl/left_hand",
            #     right_hand_topic="/inspire_hand/ctrl/right_hand"
            # )
            from inrocs.robot_env.tianyi_env import tianyi_env as robot_env
            # qpos = tianyi_env.get_obs_full()['qpos']

            # print(qpos)

            # qpos[7] = 1.0
            # tianyi_env.step_full(qpos)

            # # tianyi_env.reset_to_home()
            if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                robot_env.reset_to_parepre()
            elif self.args['tg_mode'] in ['mode5', 'mode6', 'mode7', 'mode8']:
                robot_env.reset_to_parepre_left()

            # traj_list = ['/home/ps/wk/benchmark_results/tiangong_1122_traj.hdf5',
            # '/home/ps/wk/benchmark_results/tiangong_1122_traj_2.hdf5', 
            # '/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5' ]

            # h5_file = '/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5'
            # h5_file = '/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5'
            # self.init_tiangong(robot_env, h5_file)

        # warm up
        import time
        time.sleep(2)
        if self.exp_type == 'tiangong_1rgb':
            if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                obs = robot_env.get_obs_full()
            else:
                obs = robot_env.get_obs()
        else:
            obs = robot_env.get_obs()
        print('***obs***:', obs)

        ###
        print("enter enter to go")
        global preparing
        while preparing:
            # ...
            # obs = robot_env.get_obs()
            if self.exp_type == 'tiangong_1rgb':
                if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                    obs = robot_env.get_obs_full()
                else:
                    obs = robot_env.get_obs()
            else:
                obs = robot_env.get_obs()
            input_image = self.get_image(obs, show_img=True)
            # print('***obs***:', obs)

        preparing = True
        ###
        for i in range(2):
            if self.exp_type == 'tiangong_1rgb':
                if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                    obs = robot_env.get_obs_full()
                else:
                    obs = robot_env.get_obs()
            else:
                obs = robot_env.get_obs()

        max_timesteps = self.args['episode_len']
        
        # 推理
        with torch.inference_mode():
            obs_stack = []
            obs_stack.insert(0, None)
            obs_stack.append(obs)
            for t in range(max_timesteps):
                obs_dict = self.get_model_input(t, obs_stack)
                obs_dict = dict_apply(obs_dict, 
                            lambda x: x.to(self.device, non_blocking=True))
                result = self.model.predict_action(obs_dict)
                raw_action = result['action'][0].detach().to('cpu').numpy()
                print(f"raw action shape:{raw_action.shape}")
                pred_action = self.action_post_process(raw_action, self.dataset_stats)
                obs = robot_env.step(pred_action)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', required=True)
    parser.add_argument('--tg_mode', type=str, default='mode1', required=False)
    parser.add_argument('--episode_len', action='store', type=int, help='episode_len', default=10000, required=False)
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    args_input = vars(args)
    infer_model = InferVLAIL(args_input)

    infer_model.execute()

if __name__ == '__main__':
    main()






        
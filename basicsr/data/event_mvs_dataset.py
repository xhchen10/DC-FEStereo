import os.path
from pathlib import Path
import weakref
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import hdf5plugin
import yaml
from easydict import EasyDict as edict

from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.dsec_utils import VoxelGrid
from basicsr.data.event_dataset import EventSlicer


OFFSET = 1
SEQ_LEN = 3

class Sequence(Dataset):
    def __init__(self, root_path: Path, seq_name: str, pose_dict: np.array, mode='train'):
        assert root_path.is_dir()

        self.mode = mode
        self.seq = seq_name

        # Save output dimensions
        self.height, self.width = 480, 640
        self.num_bins = 15
        delta_t_ms = 50
        self.delta_t_us = delta_t_ms * 1000

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        disp_dir = root_path / '{}_disparity'.format(mode) / seq_name / 'disparity'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

        # load disparity paths
        img_disp_dir = root_path / '{}_my_disparity_fromHR'.format(mode) / seq_name / 'disparity' / 'image'
        assert img_disp_dir.is_dir()
        disp_gt_pathstrings = list()
        for entry in img_disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.img_disp_gt_pathstrings = disp_gt_pathstrings

        assert int(Path(self.img_disp_gt_pathstrings[0]).stem) == 0
        assert len(self.img_disp_gt_pathstrings) == self.timestamps.shape[0]

        # load left image of IE
        self.img_pathstrings = dict()
        self.rectify_img_maps = dict()
        for location in ['left']:
            # load image exposure timestamps
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location
            assert img_dir.is_dir()
            # load image paths
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location / 'rectified'
            assert img_dir.is_dir()
            img_pathstrings = list()
            for entry in img_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_pathstrings.append(str(entry))
            img_pathstrings.sort()

            assert int(Path(img_pathstrings[0]).stem) == 0
            self.img_pathstrings[location] = img_pathstrings

            img_rect_file = root_path / '{}_my_rectify_IE'.format(mode) / seq_name / 'rectify_img_map.h5'
            with h5py.File(str(img_rect_file), 'r') as h5_rect:
                self.rectify_img_maps[location] = []
                self.rectify_img_maps[location].append(h5_rect['rectify_map_0'][()])
                self.rectify_img_maps[location].append(h5_rect['rectify_map_1'][()])

        # align the frame rate between disparity and frame
        for location in ['left']:
            self.img_pathstrings[location] = self.img_pathstrings[location][::2]
            assert len(self.img_pathstrings[location]) == self.timestamps.shape[0]

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        self.img_disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]
        for location in self.img_pathstrings.keys():
            self.img_pathstrings[location].pop(0)

        # load event sequence
        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()
        for location in ['right']:
            ev_dir_location = root_path / '{}_events'.format(mode) / seq_name / 'events' / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = root_path / '{}_my_rectify_IE'.format(mode) / seq_name / 'rectify_ev_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # load camera intrinsic/extrinsic params
        confpath = root_path / '{}_my_rectify_IE'.format(mode) / seq_name / 'cam_to_cam.yaml'
        assert confpath.exists()
        with open(confpath, 'r') as f:
            conf = edict(yaml.load(f, Loader=yaml.CLoader))
        self.K = np.eye(3, dtype=np.float32)
        self.K[[0, 1, 0, 1], [0, 1, 2, 2]] = np.array(conf['intrinsics']['camMyRect1']['camera_matrix'])
        self.Q = np.array(conf['disparity_to_depth']['cams_13'], dtype=np.float32)
        self.T_myr3_myr1 = np.array(conf['extrinsics']['T_myrect3_myrect1'], np.float32)
        self.T_myr1_r1 = np.eye(4, dtype=np.float32)
        self.T_myr1_r1[:3, :3] = np.array(conf['extrinsics']['R_myrect1'])

        self.pose_dict = pose_dict

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    def get_frame(self, filepath: Path):
        assert filepath.is_file()
        image = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        return image

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        # if not (x.max() < self.width or y.max() < self.height):
        #     print(x.max(), y.max())
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def get_event(self, idx, location='right'):
        ts_end = self.timestamps[idx]
        ts_start = ts_end - self.delta_t_us
        event_data = self.event_slicers[location].get_events(ts_start, ts_end)
        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']
        xy_rect = self.rectify_events(x, y, location)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
        return event_representation

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def __len__(self):
        return len(self.timestamps) - OFFSET - (SEQ_LEN - 1)

    def get_poses(self, idx_0):
        disp_gt_path_0 = Path(self.img_disp_gt_pathstrings[idx_0])
        file_idx = '%s_%06d' % (self.seq, int(disp_gt_path_0.stem))

        # pose estimated with the depth of r1 using Deep-Global-Registration
        pose_fw_r1 = self.pose_dict[file_idx].astype(np.float32)
        pose_fw_myr1 = self.T_myr1_r1 @ pose_fw_r1 @ np.linalg.inv(self.T_myr1_r1)
        pose_bw = np.linalg.inv(pose_fw_myr1)
        pose_img0_img1 = pose_bw.copy()
        pose_ev_img = self.T_myr3_myr1  # pose transform from Left to Right
        pose_ev0_img1 = pose_ev_img @ pose_img0_img1

        return pose_img0_img1[:3, :], pose_ev_img[:3, :], pose_ev0_img1[:3, :]

    def get_image(self, index, location):
        image = self.get_frame(Path(self.img_pathstrings[location][index])).astype(np.float32) / 255.
        image = cv2.remap(image, self.rectify_img_maps[location][0], self.rectify_img_maps[location][1],
                          cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        image = np.clip(image, 0, 1)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image

    def __getitem__(self, index):
        index = index + OFFSET + (SEQ_LEN - 1)  # index of the final predicted timestamp

        intrinsic = self.K
        baseline_mul_focal_length = (1 / self.Q[3, 2]) * self.Q[2, 3]

        poses_img0_img1, poses_ev_img, poses_ev0_img1 = [], [], []
        for i in range(SEQ_LEN):
            pose_idx0 = (index - OFFSET) - (SEQ_LEN - i - 1)
            pose_img0_img1, pose_ev_img, pose_ev0_img1 = self.get_poses(pose_idx0)
            poses_img0_img1.append(pose_img0_img1)
            poses_ev_img.append(pose_img0_img1)
            poses_ev0_img1.append(pose_ev0_img1)

        disps_gt_1 = []
        for i in range(SEQ_LEN):
            gt_idx_1 = index - (SEQ_LEN - i - 1)
            disp_gt_path_1 = Path(self.img_disp_gt_pathstrings[gt_idx_1])
            disp_gt_1 = self.get_disparity_map(disp_gt_path_1)
            disp_gt_1 = torch.from_numpy(disp_gt_1)
            disps_gt_1.append(disp_gt_1)

        images, events = [], []
        for i in range(SEQ_LEN + 1):
            idx = index - (SEQ_LEN - i)
            images.append(self.get_image(idx, location='left'))
            events.append(self.get_event(idx, location='right'))

        output = {'images': images, 'events': events,
                  'intrinsic': intrinsic, 'baseline_mul_focal_length': baseline_mul_focal_length,
                  'poses_ev_img': poses_ev_img, 'poses_img0_img1': poses_img0_img1,
                  'poses_ev0_img1': poses_ev0_img1, 'disps_gt_1': disps_gt_1
                  }

        return output


@DATASET_REGISTRY.register()
class DSECMVSRecurDataset(Dataset):
    subfolder = {"train": ['zurich_city_04_c',
                         'zurich_city_02_e',
                         'zurich_city_11_a',
                         'zurich_city_10_a',
                         'zurich_city_01_b',
                         'zurich_city_01_d',
                         'zurich_city_01_e',
                         'zurich_city_04_e',
                         'zurich_city_02_b',
                         'zurich_city_07_a',
                         'zurich_city_05_b',
                         'zurich_city_04_f',
                         'zurich_city_01_c',
                         'zurich_city_11_b',
                         'zurich_city_09_a',
                         'zurich_city_05_a',
                         'zurich_city_11_c',
                         'zurich_city_04_b',
                         'zurich_city_03_a',
                         'zurich_city_09_d',
                         'thun_00_a',
                         'interlaken_00_f',
                         'zurich_city_00_b',
                         'zurich_city_00_a',
                         'interlaken_00_d',
                         'zurich_city_02_a',
                         'interlaken_00_e',
                         'interlaken_00_c',
                         'zurich_city_06_a',
                         'zurich_city_01_f',
                         'zurich_city_04_d',
                         'zurich_city_02_d',
                         'zurich_city_09_e',
                         'zurich_city_02_c',
                         'zurich_city_09_b'],
                 "val": ['zurich_city_04_a',
                         'zurich_city_01_a',
                         'zurich_city_09_c',
                         'zurich_city_10_b',
                         'zurich_city_08_a',
                         'interlaken_00_g']}
    def __init__(self, opt):
        self.opt = opt
        dataset_path = opt['dataset_path']
        pose_dict = np.load(str(os.path.join(dataset_path, 'pose_dict_%d.npy' % OFFSET)), allow_pickle=True).item()
        sequences = [Sequence(Path(dataset_path), scene, pose_dict) for scene in self.subfolder[opt['phase']]]
        self.dataset = ConcatDataset(sequences)

    def __getitem__(self, idx):
        if self.opt['phase'] == 'train':
            return self.dataset[idx]
        else:
            return self.dataset[idx*20]

    def __len__(self):
        if self.opt['phase'] == 'train':
            return len(self.dataset)
        else:
            return len(self.dataset) // 20

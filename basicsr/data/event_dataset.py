import math
from pathlib import Path
import weakref
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import imageio
from typing import Dict, Tuple
from numba import jit
import hdf5plugin
import random
from PIL import Image

from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.dsec_utils import VoxelGrid, load_exposure_timestamps, render


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


class Sequence(Dataset):
    """
    root_dir
    ├── ${mode}$_disparity
    │     ├── ${seq_name}$
    │     │     ├── disparity
    │     │     │     ├── event
    │     │     │     │     ├── 000000.png
    │     │     │     │     └── ...
    │     │     │     ├── image
    │     │     │     │     ├── 000000.png
    │     │     │     │     └── ...
    │     │     │     └── timestamps.txt
    ├── ${mode}$_events
    │     ├── ${seq_name}$
    │     │     │     ├── events
    │     │     │     │     ├── left
    │     │     │     │     │     ├── events.h5
    │     │     │     │     │     └── rectify_map.h5
    │     │     │     │     ├── right
    │     │     │     │     │     ├── events.h5
    │     │     │     │     │     └──rectify_map.h5
    ├── ${mode}$_images
    │     ├── ${seq_name}$
    │     │     │     ├── images
    │     │     │     │     ├── left
    │     │     │     │     │     ├── rectified
    │     │     │     │     │     │     ├── 000000.png
    │     │     │     │     │     │     └── ...
    │     │     │     │     │     └── exposure_timestamps.txt
    │     │     │     │     ├── right
    │     │     │     │     │     ├── rectified
    │     │     │     │     │     │     ├── 000000.png
    │     │     │     │     │     │     └── ...
    │     │     │     │     │     └── exposure_timestamps.txt
    │     │     │     │     └── timestamps.txt
    """
    def __init__(self, root_path: Path, seq_name: str, mode='train',
                 modality='EE', delta_t_ms=50, num_bins=15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert root_path.is_dir()

        self.mode = mode
        self.modality = modality

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.locations = ['left', 'right']

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000
        ## delta_t is replaced by the exposure time of frame

        # load disparity timestamps
        disp_dir = root_path / '{}_disparity'.format(mode) / seq_name / 'disparity'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

        # load disparity paths
        # TODO: With what timestamp to index
        ev_disp_dir = disp_dir / 'event'
        assert ev_disp_dir.is_dir()
        disp_gt_pathstrings = list()
        for entry in ev_disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.ev_disp_gt_pathstrings = disp_gt_pathstrings

        img_disp_dir = disp_dir / 'image'
        assert img_disp_dir.is_dir()
        disp_gt_pathstrings = list()
        for entry in img_disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.img_disp_gt_pathstrings = disp_gt_pathstrings

        assert int(Path(self.img_disp_gt_pathstrings[0]).stem) == 0
        assert int(Path(self.ev_disp_gt_pathstrings[0]).stem) == 0
        assert len(self.img_disp_gt_pathstrings) == self.timestamps.shape[0]
        assert len(self.ev_disp_gt_pathstrings) == self.timestamps.shape[0]

        # load image
        self.exposure_timestamps = dict()
        self.img_pathstrings = dict()
        for location in self.locations:
            # load image exposure timestamps
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location
            assert img_dir.is_dir()
            self.exposure_timestamps[location] = load_exposure_timestamps(img_dir / 'exposure_timestamps.txt')
            # load image paths
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location / 'rectified'
            assert img_dir.is_dir()
            img_pathstrings = list()
            for entry in img_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_pathstrings.append(str(entry))
            img_pathstrings.sort()

            assert int(Path(img_pathstrings[0]).stem) == 0
            assert len(img_pathstrings) == self.exposure_timestamps[location].shape[0]

            self.img_pathstrings[location] = img_pathstrings

        # align the frame rate between disparity and frame
        for location in self.exposure_timestamps.keys():
            self.exposure_timestamps[location] = self.exposure_timestamps[location][::2]
            self.img_pathstrings[location] = self.img_pathstrings[location][::2]

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        self.ev_disp_gt_pathstrings.pop(0)
        self.img_disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]
        for location in self.exposure_timestamps.keys():
            self.exposure_timestamps[location] = self.exposure_timestamps[location][1:]
            self.img_pathstrings[location].pop(0)

        # load event sequence
        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        for location in self.locations:
            ev_dir_location = root_path / '{}_events'.format(mode) / seq_name / 'events' / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

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

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def resize_disparity_map(disp, h, w):
        h_ori, w_ori = disp.shape[:2]
        scale_factor = h / h_ori
        disp = Image.fromarray(disp)
        disp = np.array(disp.resize((w, h), resample=Image.NEAREST), dtype=np.float32) * scale_factor
        return disp

    def get_frame(self, filepath: Path):
        assert filepath.is_file()
        image = cv2.imread(str(filepath))
        image = cv2.resize(image, (self.width, self.height))
        return image.astype('float32')/255

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        return len(self.ev_disp_gt_pathstrings)

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def getitem_EE(self, index):
        ts_end = self.timestamps[index]
        # ts_start should be fine (within the window as we removed the first disparity map)
        ts_start = ts_end - self.delta_t_us

        disp_gt_path = Path(self.ev_disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'disparity_gt': self.get_disparity_map(disp_gt_path),
            'file_index': file_index,
        }
        for location in self.locations:
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
            if 'representation' not in output:
                output['representation'] = dict()
            output['representation'][location] = event_representation

        return output

    def getitem_II(self, index):
        disp_gt_path = Path(self.img_disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'disparity_gt': self.resize_disparity_map(self.get_disparity_map(disp_gt_path),
                                                      self.height, self.width),
            'file_index': file_index,
        }
        for location in self.locations:
            image_path = Path(self.img_pathstrings[location][index])
            if 'frame' not in output:
                output['frame'] = dict()
            output['frame'][location] = np.transpose(self.get_frame(image_path), (2, 0, 1))
        return output

    def __getitem__(self, index):
        if self.modality == "EE":
            return self.getitem_EE(index)
        elif self.modality == "II":
            return self.getitem_II(index)
        else:
            raise NotImplementedError


class HybridSequence(Sequence):
    def __init__(self, root_path: Path, seq_name: str, mode='train',
                 modality='IE', delta_t_ms=50, num_bins=15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert root_path.is_dir()

        self.mode = mode
        self.modality = modality

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.locations = ['left', 'right']

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # load disparity timestamps
        disp_dir = root_path / '{}_disparity'.format(mode) / seq_name / 'disparity'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

        # load disparity paths
        ev_disp_dir = root_path / '{}_my_disparity_fromHR'.format(mode) / seq_name / 'disparity' / 'event'
        assert ev_disp_dir.is_dir()
        disp_gt_pathstrings = list()
        for entry in ev_disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.ev_disp_gt_pathstrings = disp_gt_pathstrings

        img_disp_dir = root_path / '{}_my_disparity_fromHR'.format(mode) / seq_name / 'disparity' / 'image'
        assert img_disp_dir.is_dir()
        disp_gt_pathstrings = list()
        for entry in img_disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.img_disp_gt_pathstrings = disp_gt_pathstrings

        assert int(Path(self.img_disp_gt_pathstrings[0]).stem) == 0
        assert int(Path(self.ev_disp_gt_pathstrings[0]).stem) == 0
        assert len(self.img_disp_gt_pathstrings) == self.timestamps.shape[0]
        assert len(self.ev_disp_gt_pathstrings) == self.timestamps.shape[0]

        # load image
        self.exposure_timestamps = dict()
        self.img_pathstrings = dict()
        self.rectify_img_maps = dict()
        for location in self.locations:
            # load image exposure timestamps
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location
            assert img_dir.is_dir()
            self.exposure_timestamps[location] = load_exposure_timestamps(img_dir / 'exposure_timestamps.txt')
            # load image paths
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location / 'rectified'
            assert img_dir.is_dir()
            img_pathstrings = list()
            for entry in img_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_pathstrings.append(str(entry))
            img_pathstrings.sort()

            assert int(Path(img_pathstrings[0]).stem) == 0
            assert len(img_pathstrings) == self.exposure_timestamps[location].shape[0]

            self.img_pathstrings[location] = img_pathstrings

            img_rect_file = root_path / '{}_my_rectify_{}'.format(mode, modality) / seq_name / 'rectify_img_map.h5'
            with h5py.File(str(img_rect_file), 'r') as h5_rect:
                self.rectify_img_maps[location] = []
                self.rectify_img_maps[location].append(h5_rect['rectify_map_0'][()])
                self.rectify_img_maps[location].append(h5_rect['rectify_map_1'][()])

        # align the frame rate between disparity and frame
        for location in self.exposure_timestamps.keys():
            self.exposure_timestamps[location] = self.exposure_timestamps[location][::2]
            self.img_pathstrings[location] = self.img_pathstrings[location][::2]

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        self.ev_disp_gt_pathstrings.pop(0)
        self.img_disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]
        for location in self.exposure_timestamps.keys():
            self.exposure_timestamps[location] = self.exposure_timestamps[location][1:]
            self.img_pathstrings[location].pop(0)

        # load event sequence
        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        for location in self.locations:
            ev_dir_location = root_path / '{}_events'.format(mode) / seq_name / 'events' / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = root_path / '{}_my_rectify_{}'.format(mode, modality) / seq_name / 'rectify_ev_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def getitem(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us

        if self.modality == 'EI':
            disp_gt_path = Path(self.ev_disp_gt_pathstrings[index])
        else:
            disp_gt_path = Path(self.img_disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'disparity_gt': self.get_disparity_map(disp_gt_path),
            'file_index': file_index,
        }

        for location in self.locations:
            if location == 'left' and self.modality == 'IE':
                continue
            if location == 'right' and self.modality == 'EI':
                continue
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
            if 'representation' not in output:
                output['representation'] = dict()
            output['representation'][location] = event_representation

        for location in self.locations:
            if location == 'right' and self.modality == 'IE':
                continue
            if location == 'left' and self.modality == 'EI':
                continue
            image_path = Path(self.img_pathstrings[location][index])
            if 'frame' not in output:
                output['frame'] = dict()
            img = self.get_frame(image_path)
            img = cv2.remap(img, self.rectify_img_maps[location][0], self.rectify_img_maps[location][1],
                            cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            img = np.clip(img, 0, 1)
            output['frame'][location] = np.transpose(img, (2, 0, 1))

        return output

    def __getitem__(self, index):
        return self.getitem(index)


class HybridEdEvSequence(HybridSequence):
    def getitem(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us

        assert self.modality == 'EDEV'
        disp_gt_path = Path(self.img_disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'file_index': file_index,
        }

        location = 'right'
        event_data = self.event_slicers[location].get_events(ts_start, ts_end)

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y, location)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
        if 'representation' not in output:
            output['representation'] = dict()
        output['representation'][location] = event_representation

        location = 'left'
        image_path = Path(self.img_pathstrings[location][index])
        if 'frame' not in output:
            output['frame'] = dict()
        frame = cv2.cvtColor(self.get_frame(image_path), cv2.COLOR_BGR2GRAY)
        frame = cv2.remap(frame, self.rectify_img_maps[location][0], self.rectify_img_maps[location][1],
                        cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255.).astype(np.uint8)
        edge = cv2.Laplacian(frame, cv2.CV_64F)
        edge_ori = edge.copy()
        edge[np.where(np.abs(edge) <= np.max(np.abs(edge / 50)))] = 0
        output['frame'][location] = np.expand_dims(edge, 0).astype(np.float32)
        img_gt = self.get_disparity_map(disp_gt_path)
        img_gt[np.where(np.abs(edge_ori) <= np.max(np.abs(edge_ori / 50)))] = 0
        output['disparity_gt'] = img_gt

        return output


class HybridSymSequence(Sequence):
    def __init__(self, root_path: Path, seq_name: str, mode='train',
                 modality='IEIE', delta_t_ms=50, num_bins=15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert root_path.is_dir()

        self.mode = mode
        self.modality = modality

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.locations = ['left', 'right']

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # load disparity timestamps
        disp_dir = root_path / '{}_disparity'.format(mode) / seq_name / 'disparity'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

        # load disparity paths
        ev_disp_dir = root_path / '{}_disparity'.format(mode) / seq_name / 'disparity' / 'event'
        assert ev_disp_dir.is_dir()
        disp_gt_pathstrings = list()
        for entry in ev_disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.ev_disp_gt_pathstrings = disp_gt_pathstrings

        assert int(Path(self.ev_disp_gt_pathstrings[0]).stem) == 0
        assert len(self.ev_disp_gt_pathstrings) == self.timestamps.shape[0]

        # load image
        self.exposure_timestamps = dict()
        self.img_pathstrings = dict()
        self.rectify_img_maps = dict()
        for location in self.locations:
            # load image exposure timestamps
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location
            assert img_dir.is_dir()
            self.exposure_timestamps[location] = load_exposure_timestamps(img_dir / 'exposure_timestamps.txt')
            # load image paths
            img_dir = root_path / '{}_images'.format(mode) / seq_name / 'images' / location / 'rectified'
            assert img_dir.is_dir()
            img_pathstrings = list()
            for entry in img_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_pathstrings.append(str(entry))
            img_pathstrings.sort()

            assert int(Path(img_pathstrings[0]).stem) == 0
            assert len(img_pathstrings) == self.exposure_timestamps[location].shape[0]

            self.img_pathstrings[location] = img_pathstrings

            img_rect_file = root_path / '{}_my_rectify_IEIE'.format(mode) / seq_name / 'rectify_{}_map.h5'.format(location)
            with h5py.File(str(img_rect_file), 'r') as h5_rect:
                self.rectify_img_maps[location] = []
                self.rectify_img_maps[location].append(h5_rect['rectify_map_x'][()])
                self.rectify_img_maps[location].append(h5_rect['rectify_map_y'][()])

        # align the frame rate between disparity and frame
        for location in self.exposure_timestamps.keys():
            self.exposure_timestamps[location] = self.exposure_timestamps[location][::2]
            self.img_pathstrings[location] = self.img_pathstrings[location][::2]

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        self.ev_disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]
        for location in self.exposure_timestamps.keys():
            self.exposure_timestamps[location] = self.exposure_timestamps[location][1:]
            self.img_pathstrings[location].pop(0)

        # load event sequence
        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        for location in self.locations:
            ev_dir_location = root_path / '{}_events'.format(mode) / seq_name / 'events' / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def getitem(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self.delta_t_us

        disp_gt_path = Path(self.ev_disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'disparity_gt': self.get_disparity_map(disp_gt_path),
            'file_index': file_index,
        }

        for location in self.locations:
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)

            image_path = Path(self.img_pathstrings[location][index])
            img = self.get_frame(image_path)
            img = cv2.remap(img, self.rectify_img_maps[location][0], self.rectify_img_maps[location][1],
                            cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (2, 0, 1))

            if 'cat_representation' not in output:
                output['cat_representation'] = dict()
            output['cat_representation'][location] = np.concatenate([event_representation, img], 0)

        return output

    def __getitem__(self, index):
        return self.getitem(index)

    def get_frame(self, filepath: Path):
        assert filepath.is_file()
        image = cv2.imread(str(filepath))
        return image.astype('float32')/255


class EdgeSequence(Sequence):
    def getitem_EdEd(self, index):
        disp_gt_path = Path(self.img_disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)
        output = {
            'file_index': file_index,
        }
        for location in self.locations:
            image_path = Path(self.img_pathstrings[location][index])
            if 'frame' not in output:
                output['frame'] = dict()
            frame = cv2.cvtColor(self.get_frame(image_path), cv2.COLOR_BGR2GRAY)
            frame = (frame * 255.).astype(np.uint8)
            edge = cv2.Laplacian(frame, cv2.CV_64F)
            edge_ori = edge.copy()
            edge[np.where(np.abs(edge) <= np.max(np.abs(edge / 50)))] = 0
            output['frame'][location] = np.expand_dims(edge, 0).astype(np.float32)
            if location == 'left':
                output['frame'][location] = np.expand_dims(edge, 0).astype(np.float32)
                img_gt = self.resize_disparity_map(self.get_disparity_map(disp_gt_path),
                                                   self.height, self.width)
                img_gt[np.where(np.abs(edge_ori) <= np.max(np.abs(edge_ori / 50)))] = 0
                output['disparity_gt'] = img_gt

        return output

    def __getitem__(self, index):
        if self.modality == "EDED":
            return self.getitem_EdEd(index)
        else:
            raise NotImplementedError


@DATASET_REGISTRY.register()
class DSECDataset(Dataset):
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
        mode = opt['mode'].upper()
        assert mode in ['EE', 'EI', 'IE', 'II', 'EDED', 'IEIE', 'EDEV']
        if mode in ['EE', 'II']:
            sequences = [Sequence(Path(dataset_path), scene, modality=mode) for scene in self.subfolder[opt['phase']]]
        elif mode in ['EI', 'IE']:
            sequences = [HybridSequence(Path(dataset_path), scene, modality=mode) for scene in self.subfolder[opt['phase']]]
        elif mode == 'EDED':
            sequences = [EdgeSequence(Path(dataset_path), scene, modality=mode) for scene in self.subfolder[opt['phase']]]
        elif mode == 'IEIE':
            sequences = [HybridSymSequence(Path(dataset_path), scene, modality=mode) for scene in self.subfolder[opt['phase']]]
        else:
            sequences = [HybridEdEvSequence(Path(dataset_path), scene, modality=mode) for scene in self.subfolder[opt['phase']]]
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


if __name__ == '__main__':
    dataset = DSECDataset({'dataset_path': '/home/chenxh/devdata1/Dataset/Event/DSEC',
                           'phase': 'train',
                           'mode': 'EE'})
    o = dataset.__getitem__(0)
    print(1)
    print(len(dataset))
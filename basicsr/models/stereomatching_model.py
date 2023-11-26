import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class StereoMatchingModel(BaseModel):
    def __init__(self, opt):
        super(StereoMatchingModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.loss_disp = build_loss(train_opt['loss_disp']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.left = data['representation']['left'].to(self.device)
        self.right = data['representation']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.left, self.right)

        l_total = 0
        loss_dict = OrderedDict()
        l_disp = self.loss_disp(self.output, self.gt)
        l_total += l_disp
        loss_dict['l_disp'] = l_disp

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.left, self.right)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            metric_data['estimated_disparity'] = self.output
            metric_data['ground_truth_disparity'] = self.gt
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {idx}')

            del self.left, self.right
            del self.output
            if hasattr(self, 'gt'):
                del self.gt
            torch.cuda.empty_cache()

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        # self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class ImgStereoMatchingModel(StereoMatchingModel):
    def __init__(self, opt):
        super(ImgStereoMatchingModel, self).__init__(opt)

    def feed_data(self, data):
        self.left = data['frame']['left'].to(self.device)
        self.right = data['frame']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)


@MODEL_REGISTRY.register()
class EIStereoMatchingModel(StereoMatchingModel):
    def __init__(self, opt):
        super(EIStereoMatchingModel, self).__init__(opt)

    def feed_data(self, data):
        self.left = data['representation']['left'].to(self.device)
        self.right = data['frame']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)


@MODEL_REGISTRY.register()
class IEStereoMatchingModel(StereoMatchingModel):
    def __init__(self, opt):
        super(IEStereoMatchingModel, self).__init__(opt)

    def feed_data(self, data):
        self.left = data['frame']['left'].to(self.device)
        self.right = data['representation']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)


@MODEL_REGISTRY.register()
class IEIEStereoMatchingModel(StereoMatchingModel):
    def __init__(self, opt):
        super(IEIEStereoMatchingModel, self).__init__(opt)

    def feed_data(self, data):
        self.left = data['cat_representation']['left'].to(self.device)
        self.right = data['cat_representation']['right'].to(self.device)
        if 'disparity_gt' in data:
            self.gt = data['disparity_gt'].to(self.device)
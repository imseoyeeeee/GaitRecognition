
import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, get_msg_mgr
from utils.run_dir import new_train_dir, parse_ckpt, write_settings

parser = argparse.ArgumentParser(description='Main program for FreQait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
parser.add_argument('--cfgs', type=str,
                    default="configs/freqait/freqait_Gait3D.yaml", help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file: <run folder>/logs/<Datetime>.txt")
parser.add_argument('--ckpt', type=str, default='',
                    help="test: checkpoint to evaluate (required). train: resume from this checkpoint "
                         "(same run folder). Without it, training starts in a new folder "
                         "output/<temporal>/<spatial>/train<k>/")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = cfgs['output_dir']
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    training = (opt.phase == 'train')

    # run folder: output/<temporal>/<spatial>/train<k>/ (see utils/run_dir.py)
    if opt.ckpt:
        run_dir, save_name, iteration = parse_ckpt(opt.ckpt)
        if training:      # resume: same folder, same checkpoint names
            cfgs['trainer_cfg']['save_name'] = save_name
            cfgs['trainer_cfg']['restore_hint'] = iteration
        else:
            cfgs['evaluator_cfg']['restore_hint'] = os.path.abspath(opt.ckpt)
    elif training:
        if cfgs['trainer_cfg'].get('restore_hint', 0) not in (0, None):
            raise ValueError('trainer_cfg.restore_hint is not used any more: resume with --ckpt <checkpoint>')
        run_dir = [new_train_dir(cfgs) if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(run_dir, src=0)
        run_dir = run_dir[0]
        if torch.distributed.get_rank() == 0:
            write_settings(cfgs, run_dir, opt.cfgs)
    else:
        raise ValueError('--phase test needs --ckpt <path to checkpoint .pt>')
    cfgs['output_dir'] = run_dir

    initialization(cfgs, training)
    get_msg_mgr().log_info('Run folder: {}'.format(run_dir))
    run_model(cfgs, training)

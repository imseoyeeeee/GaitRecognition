"""
Output folders per training run:

    output/<temporal>/<spatial>/train<k>/
        checkpoints/<save_name>-<iter>.pt
        summary/          (tensorboard)
        logs/             (--log_to_file)
        settings.md       (settings that the folder names do not show + the full config)
        config.yaml       (the training config; use it with --phase test)

<temporal> : dtfp | transformer | set                       (model_cfg.temporal.type)
<spatial>  : cnnattention | mlpattention | mlp              (model_cfg.efd.encoder / embed)
train<k>   : a new folder for every training run (k = 1, 2, 3, ...)
"""
import os
import os.path as osp
import re
from time import localtime, strftime

import yaml

TEMPORAL_NAMES = {'dtfpnet': 'dtfp', 'transformer': 'transformer', 'set': 'set'}


def temporal_name(model_cfg):
    t = model_cfg.get('temporal', {}).get('type', 'dtfpnet')
    return TEMPORAL_NAMES.get(t, t)


def spatial_name(model_cfg):
    efd = model_cfg.get('efd', {})
    if efd.get('encoder', 'attention') == 'mlp':
        return 'mlp'
    return '{}attention'.format(efd.get('embed', 'cnn'))


def new_train_dir(cfgs, root='output'):
    """output/<temporal>/<spatial>/train<k> with the next unused k (the folder is created)."""
    base = osp.join(root, temporal_name(cfgs['model_cfg']), spatial_name(cfgs['model_cfg']))
    os.makedirs(base, exist_ok=True)
    used = [int(m.group(1)) for d in os.listdir(base) for m in [re.fullmatch(r'train(\d+)', d)] if m]
    k = max(used, default=0) + 1
    while True:                                   # exist_ok=False: two runs started at once get different k
        path = osp.join(base, 'train{}'.format(k))
        try:
            os.makedirs(path)
            return path
        except FileExistsError:
            k += 1


def parse_ckpt(ckpt):
    """<run>/checkpoints/<save_name>-<iter>.pt -> (run folder, save_name, iter)"""
    ckpt = osp.abspath(ckpt)
    if not osp.isfile(ckpt):
        raise FileNotFoundError('checkpoint not found: {}'.format(ckpt))
    m = re.fullmatch(r'(.+)-(\d+)\.pt', osp.basename(ckpt))
    if m is None:
        raise ValueError('checkpoint name must be <save_name>-<iteration>.pt: {}'.format(ckpt))
    folder = osp.dirname(ckpt)
    run = osp.dirname(folder) if osp.basename(folder) == 'checkpoints' else folder
    return run, m.group(1), int(m.group(2))


def _get(d, path, default='-'):
    for k in path.split('.'):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _parts(parts):
    out = []
    for p in parts or []:
        if isinstance(p, dict):
            j = p.get('joints')
            out.append('all' if j == 'all' else '{}'.format(j))
        else:
            out.append('{}'.format(p))
    return '; '.join(out)


def write_settings(cfgs, run_dir, cfg_path=''):
    """settings.md: the ablation settings that the folder names do not show, then the full config."""
    m, t, tr = cfgs['model_cfg'], cfgs['trainer_cfg'], cfgs['trainer_cfg'].get('sampler', {})
    tmp_type = _get(m, 'temporal.type', 'dtfpnet')
    rows = [
        ('Folder', 'temporal', temporal_name(m)),
        ('Folder', 'spatial (EFD encoder / embed)', spatial_name(m)),
        ('EFD', 'input', _get(m, 'efd.input', 'raw')),
        ('EFD', 'harmonics', _get(m, 'efd.harmonics')),
        ('EFD', 'attn_layers / num_heads', '{} / {}'.format(_get(m, 'efd.attn_layers'), _get(m, 'efd.num_heads'))),
        ('EFD', 'mlp_hidden (encoder mlp)', _get(m, 'efd.mlp_hidden')),
        ('Skeleton', 'mode', _get(m, 'skeleton.mode', 'data')),
        ('Skeleton', 'use_velocity', _get(m, 'skeleton.use_velocity')),
        ('Skeleton', 'edge_conf / learn_alpha / edge_alpha / edge_sigma',
         '{} / {} / {} / {}'.format(_get(m, 'skeleton.edge_conf'), _get(m, 'skeleton.learn_alpha'),
                                   _get(m, 'skeleton.edge_alpha'), _get(m, 'skeleton.edge_sigma'))),
        ('Skeleton', 'hidden_dim', _get(m, 'skeleton.hidden_dim')),
        ('Skeleton', 'parts (joints)', _parts(_get(m, 'skeleton.parts', []))),
        ('Model', 'dim (part feature D)', _get(m, 'dim')),
        ('Model', 'part_aggregation type / out_dim',
         '{} / {}'.format(_get(m, 'part_aggregation.type', 'partwise'), _get(m, 'part_aggregation.out_dim'))),
        ('Model', 'temporal settings', _get(m, 'temporal.' + tmp_type, {}) if tmp_type != 'set' else '-'),
        ('Model', 'head type / embed_dim', '{} / {}'.format(_get(m, 'head.type', 'global'), _get(m, 'head.embed_dim'))),
        ('Model', 'fusion', _get(m, 'fusion', {})),
        ('Augmentation', 'augmentation', _get(m, 'augmentation', {})),
        ('Loss', 'loss terms', ', '.join(l.get('type', '?') for l in cfgs.get('loss_cfg', []))),
        ('Train', 'optimizer', '{} lr {} wd {}'.format(_get(cfgs, 'optimizer_cfg.solver'), _get(cfgs, 'optimizer_cfg.lr'),
                                                      _get(cfgs, 'optimizer_cfg.weight_decay'))),
        ('Train', 'scheduler', '{} milestones {} gamma {}'.format(_get(cfgs, 'scheduler_cfg.scheduler'),
                                                                 _get(cfgs, 'scheduler_cfg.milestones'),
                                                                 _get(cfgs, 'scheduler_cfg.gamma'))),
        ('Train', 'total_iter', t.get('total_iter')),
        ('Train', 'batch (IDs x seqs) / frames', '{} / {}'.format(tr.get('batch_size'), tr.get('frames_num_fixed'))),
        ('Data', 'dataset_root', _get(cfgs, 'data_cfg.dataset_root')),
    ]
    lines = ['# {}'.format(run_dir), '',
             '* started: {}'.format(strftime('%Y-%m-%d %H:%M:%S', localtime())),
             '* config file: {}'.format(cfg_path), '',
             '| group | setting | value |', '|---|---|---|']
    lines += ['| {} | {} | {} |'.format(g, k, str(v).replace('|', '/')) for g, k, v in rows]
    lines += ['', '## Result', '', '(fill in)', '',
              '## Full config', '', '```yaml', yaml.safe_dump(cfgs, sort_keys=False, allow_unicode=True).rstrip(), '```', '']
    with open(osp.join(run_dir, 'settings.md'), 'w') as f:
        f.write('\n'.join(lines))
    with open(osp.join(run_dir, 'config.yaml'), 'w') as f:      # exact training config, for --phase test
        yaml.safe_dump(cfgs, f, sort_keys=False, allow_unicode=True)

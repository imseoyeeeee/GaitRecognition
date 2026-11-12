import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr, is_bool_list, is_int_list

import torch
from pathlib import Path


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read .pkl files written with numpy >= 2 (module `numpy._core`) under numpy 1.x (`numpy.core`)."""

    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            try:
                __import__(module)
            except ModuleNotFoundError:
                module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def load_pkl(path):
    try:
        with open(path, 'rb') as f:
            return _NumpyCompatUnpickler(f).load()
    except Exception as e:   # name the broken file (e.g. truncated by an interrupted extraction)
        raise RuntimeError('cannot load {}: {}: {}'.format(path, type(e).__name__, e)) from e


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.dataset_name = data_cfg['dataset_name']
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache'] if 'cache' in data_cfg.keys() else False
        if 'd-gait' in self.dataset_name.lower():
            self.seqs_info = [[seq[0], f"{seq[0]}-{seq[1]}", *seq[2:]] for seq in self.seqs_info]
        if training:
            if 'self_supervised' in data_cfg.keys() and data_cfg['self_supervised']:
                self.seqs_info = [[f"{seq[0]}-{seq[1]}-{seq[2]}", *seq[1:]] for seq in self.seqs_info]
            self.seqs_info = [[f"{self.dataset_name}-{seq[0]}", *seq[1:]] for seq in self.seqs_info]
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list  = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        if not is_int_list(self.data_in_use) and not self.multi_root:
            # When specified as a list of int, arbitrary order can be specified by the user
            # (multi-root: the order is the order of dataset_root, so no sorting)
            paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                _ = load_pkl(pth)
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        if self.multi_root:
            # modalities come from different folders: trim to the shortest sequence
            lengths = [len(data) for data in data_list]
            if max(lengths) - min(lengths) > self.max_len_diff:
                get_msg_mgr().log_warning('Frame numbers differ by more than {} ({}): {}'.format(
                    self.max_len_diff, lengths, paths))
            data_list = [data[:min(lengths)] for data in data_list]
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        # dataset_root may be a list: one folder per modality (e.g. [efd_root, pose_root]).
        # Sequences are matched by <ID>/<CAM>/<SEQ>; data order = order of the list.
        self.multi_root = isinstance(dataset_root, (list, tuple))
        self.max_len_diff = data_config.get('max_len_diff', 5)
        try:
            self.data_in_use = data_config['data_in_use']  # [n], true or false or [m], index of data to use
        except:
            self.data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        if self.multi_root:
            # only IDs present in every root
            label_list = sorted(set.intersection(*[set(os.listdir(r)) for r in dataset_root]))
        else:
            label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s], Len: %s' %
                                 (pid_list[0], pid_list[1], pid_list[-1], len(pid_list)))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if self.data_in_use is not None:
                                if is_bool_list(self.data_in_use): # self.data_in_use is a bool mask list
                                    seq_dirs = [dir for dir, use_bl in zip(
                                        seq_dirs, self.data_in_use) if use_bl]
                                else:   # self.data_in_use is a list of index
                                    seq_dirs = [seq_dirs[idx] for idx in self.data_in_use if idx < len(seq_dirs)]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        def find_pkls(root, lab, typ, vie):
            # <root>/<ID>/<CAM>/<SEQ>/*.pkl  or  <root>/<ID>/<CAM>/<SEQ>.pkl
            seq_path = osp.join(root, lab, typ, vie)
            if osp.isdir(seq_path):
                return [osp.join(seq_path, f) for f in sorted(os.listdir(seq_path)) if f.endswith('.pkl')]
            return [seq_path + '.pkl'] if osp.isfile(seq_path + '.pkl') else []

        def get_seqs_info_list_multi(label_set):
            seqs_info_list, n_missing = [], 0
            first = dataset_root[0]
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(first, lab))):
                    vies = sorted({v[:-4] if v.endswith('.pkl') else v
                                   for v in os.listdir(osp.join(first, lab, typ))})
                    for vie in vies:
                        per_root = [find_pkls(root, lab, typ, vie) for root in dataset_root]
                        if any(len(p) != 1 for p in per_root):
                            n_missing += 1
                            msg_mgr.log_debug('Need exactly one .pkl per root for %s-%s-%s, got %s.'
                                              % (lab, typ, vie, [len(p) for p in per_root]))
                            continue
                        seq_dirs = [p[0] for p in per_root]
                        if self.data_in_use is not None:
                            if is_bool_list(self.data_in_use):
                                seq_dirs = [d for d, use_bl in zip(seq_dirs, self.data_in_use) if use_bl]
                            else:
                                seq_dirs = [seq_dirs[idx] for idx in self.data_in_use if idx < len(seq_dirs)]
                        seqs_info_list.append([lab, typ, vie, seq_dirs])
            if n_missing > 0:
                msg_mgr.log_warning('%d sequences skipped: not found in every dataset_root.' % n_missing)
            return seqs_info_list

        get_list = get_seqs_info_list_multi if self.multi_root else get_seqs_info_list
        self.seqs_info = get_list(train_set) if training else get_list(test_set)

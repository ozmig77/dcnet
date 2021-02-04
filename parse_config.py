
from datetime import datetime
import os
import logging
from pathlib import Path
from utils import read_json, write_json
from logger import setup_logging
import pprint


class ConfigParser:

    def __init__(self, args, mode, timestamp=True):
        args = args.parse_args()
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume / 'config.json'
            # self.cfg_fname = Path(args.config)
        else:
            msg_no_cfg = "Config file must be specified"
            assert args.config is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = Path(args.config)

        self._config = read_json(self.cfg_fname)

        if mode == 'train':
            save_dir = Path(self.config['trainer']['save_dir'])
            if self.resume:
                self._save_dir = self.resume
                self._log_dir = self.resume
            else:
                self._save_dir = save_dir / args.logdir
                self._log_dir = save_dir / args.logdir
        else:
            if args.mode == 'val':
                save_dir = Path(self.config['trainer']['save_dir'])
                self._save_dir = self.resume / "val"
                self._log_dir = self.resume / "val"
                self._config['data_loader']['args']['mode'] = 'val'
            elif args.mode == 'test':
                save_dir = Path(self.config['trainer']['save_dir'])
                self._save_dir = self.resume / "test"
                self._log_dir = self.resume / "test"
                self._config['data_loader']['args']['mode'] = 'test'

        #model_name = self.cfg_fname.parent.stem
        #exper_name = f"{model_name}-{self.cfg_fname.stem}"
        #self._save_dir = save_dir / logdir
        #self._log_dir = save_dir / logdir
        #self._exper_name = exper_name
        self._args = args

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        write_json(self.config, self._save_dir / 'config.json')
        self.log_path = setup_logging(self.log_dir)

    def init(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        msg = "Overwriting kwargs given in config file is not allowed"
        assert all([k not in module_args for k in kwargs]), msg
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get(self, name, default):
        return self.config.get(name, default)

    def get_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        return logger

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def __repr__(self):
        return pprint.PrettyPrinter().pformat(self.__dict__)

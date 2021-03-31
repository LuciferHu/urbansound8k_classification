# -*- coding:utf-8 -*-
import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)    # log路径
    if log_config.is_file():    # 如果路径中找到json文件
        config = read_json(log_config)   # 解析logger_config.json文件
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():    # 解析handler参数
            if 'filename' in handler:    # 如果有指示log文件保存名
                handler['filename'] = str(save_dir / handler['filename'])    # 生成路径

        logging.config.dictConfig(config)
    else:    # 如果路径中未找到文件
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
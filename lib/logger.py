import os
import logging
import time

def get_logger(args, name=None):

    timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    file_name = f"{timestring}.log"
    logging_path = os.path.join(args.save_dir, file_name)

    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    file_handler = logging.FileHandler(logging_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
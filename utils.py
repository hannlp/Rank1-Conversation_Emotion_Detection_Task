import os
import logging

def get_logger(save_path, name=__name__):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = save_path + '/log.txt'
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    consle_handler = logging.StreamHandler()
    consle_handler.setLevel(logging.INFO)
    consle_handler.setFormatter(formatter)

    logger.addHandler(consle_handler)
    logger.addHandler(file_handler)
    return logger
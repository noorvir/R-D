import os
import logging

_default_level_name = os.getenv('PYTHON_LOGGING_LEVEL', 'INFO')
_default_level = logging.getLevelName(_default_level_name.upper())


def get_logger(log_dir=None, log_file=None):
    logger = logging.getLogger()
    logger.setLevel(_default_level)
    del logger.handlers[:]

    if log_dir and log_file:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(0)
    logger.addHandler(stream_handler)
    return logger


# TODO: plotting
# TODO: logging to file

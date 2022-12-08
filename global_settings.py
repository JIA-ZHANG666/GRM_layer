import os
from datetime import datetime

DATA_DIR = 'data/datasets/'


MILESTONES = [60, 120, 160]


TIME_NOW = datetime.now().strftime(DATE_FORMAT)

LOG_DIR = 'runs'

SAVE_EPOCH = 10

GPU_ID = 0,1
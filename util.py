import os
import random

import numpy as np
import torch

# NUDE_KEYS_3_4_2_org = [
#     "FEMALE_BREAST_EXPOSED",
#     "FEMALE_GENITALIA_EXPOSED",
#     "MALE_BREAST_EXPOSED",
#     "MALE_GENITALIA_EXPOSED",
#     "BUTTOCKS_EXPOSED",
#     "ANUS_EXPOSED",
#     "FEET_EXPOSED",
#     "BELLY_EXPOSED",
#     "ARMPITS_EXPOSED",
# ]

NUDE_KEYS_3_4_2 = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED"
]



def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class multidict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

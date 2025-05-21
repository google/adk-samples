# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
import hashlib
import logging
from os.path import abspath, dirname, join
import random

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable

DEFAULT_ATTR_PATH = join(BASE_DIR, "../data/items_ins_v2.json")
DEFAULT_FILE_PATH = join(BASE_DIR, "../data/items_shuffle.json")

DEFAULT_REVIEW_PATH = join(BASE_DIR, "../data/reviews.json")

FEAT_CONV = join(BASE_DIR, "../data/feat_conv.pt")
FEAT_IDS = join(BASE_DIR, "../data/feat_ids.pt")

HUMAN_ATTR_PATH = join(BASE_DIR, "../data/items_human_ins.json")
HUMAN_ATTR_PATH = join(BASE_DIR, "../data/items_human_ins.json")


def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then

    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx


def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(user_log_dir / f"{session_id}.jsonl", mode="w")
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk

    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()

import logging
import os
import random

import numpy as np
from PIL import ImageFilter

logger = logging.getLogger()
DATA_FOLDER = os.getenv("DATA") if os.getenv("DATA") else "data"

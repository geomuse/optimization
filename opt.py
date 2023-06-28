import numpy as np
import pandas as pd
from loguru import logger

logger.add('log/error.log',level='INFO', rotation='10 MB', format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}')


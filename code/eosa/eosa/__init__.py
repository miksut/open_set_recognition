from . import approaches
from . import architectures
from . import eval
from . import experiments
from . import data_prep
from . import openset_algos
from . import tools

import logging

# instantiate and configure project top level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)

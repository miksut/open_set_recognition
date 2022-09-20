import os
import sys

# add directory to the list of directory the interpreter checks when a module is imported
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from eosa import approaches, architectures, data_prep, tools


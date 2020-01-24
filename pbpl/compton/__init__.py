# -*- coding: utf-8 -*-
"""
Package for design and simulation of FACET-II gamma diagnostics
"""

__version__ = '0.1.0'

import os, sys
f = open(os.devnull, 'w')
temp = sys.stdout
sys.stdout = f
import Geant4 as g4
sys.stdout = temp

from .boost import *
from .core import build_transformation
from .core import transform
from .tasks import *

import math
import random
import csv, itertools
import pandas as pd
from constants import *


class Point(object):
    #A point in n dimensional space

    def __init__(self, coords):
        '''
        coords - A list of values, one per dimension
        '''

        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)


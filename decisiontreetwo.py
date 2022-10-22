import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

class RootNode:
    def __init__(self, data, depth):
        self.data = data
        self.depth = depth
        self.classifier = None
        self.positives = None
        self.negatives = None

    



class ChildNode:
    def __init__(self, data, depth, positives, negatives):
        self.data = data
        self.depth = depth
        self.classifier = None
        self.positives = positives
        self.negatives = negatives

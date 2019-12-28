import numpy as np
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
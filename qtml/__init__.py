import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
torch.set_num_threads(4)

from qtml._nn_models import MLPRegressor, CNN1DRegressor
from qtml.Models.Cross_Sectional.Classification.Models.LassoNNClassifier import LassoNNClassifier
from qtml.Models.Cross_Sectional.Classification.Models.NN_Classifier import NN_Classifier


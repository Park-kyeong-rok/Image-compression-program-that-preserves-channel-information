import os
from three_machine_learning_method import *


if args.processing_data:
    os.system('python make_pca_data.py')

if args.learning_the_svm:
    svm()

if args.learning_the_regression:
    rogistic_regression()

if args.learning_the_mlp:
    mlp()

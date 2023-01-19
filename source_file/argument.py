import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--pca_d', type = int, default= 3,
                    help = 'Insert the dimension of pca')
parser.add_argument('--data_name', type=str, default='raw',
                    help = 'Write your data name in data folder')

parser.add_argument('--processing_data', action='store_true', default=True,
                    help = 'if you need to process(pac method) the data, true or false')

parser.add_argument('--learning_the_svm', action='store_true', default=False,
                    help = 'if you want to SVM test, true or false')
parser.add_argument('--svm_kernel', type=str, default='rbf',
                    help = 'You can choice linear, poly, rbf, sigmoid, precomputed kernel')

parser.add_argument('--learning_the_regression', action='store_true', default=True,
                    help =  'if you want to regression test, true or false')
parser.add_argument('--regression_epoch', type = int, default= 80,
                    help = 'Insert the regeression epoch number')
parser.add_argument('--regression_batch', type = int, default= 128,
                    help = 'Insert the regeression batch number')
parser.add_argument('--regression_learning_rate', type = int, default= 0.005,
                    help = 'Insert the regeression learning rate')

parser.add_argument('--learning_the_mlp', action='store_true', default=True,
                    help =  'if you want to regression test, true or false')
parser.add_argument('--mlp_epoch', type = int, default= 200,
                    help = 'Insert the mlpepoch number')
parser.add_argument('--mlp_batch', type = int, default= 128,
                    help = 'Insert the mlp batch number')
parser.add_argument('--mlp_learning_rate', type = int, default= 0.005,
                    help = 'Insert the mlp learning rate')

args = parser.parse_args()
This document is for the codes of part I of the project, which is to reproduce the CNN-based return trend classifer.
To train the baseline model, please put all the submitted code files at the same directory with the data file "/monthly_20d/", and run:
python trend_classifier.py

Verification of the model performance can be done by running:
python classifier_test.py

To train and test models with different structure, please the following arguments:
'--start-year', type=int, default=1993, help='start year of training and val data'
'--period', type=int, default=8, help='period of year for training and val data'
'--gpu-num', type=int, default=0, help='gpu number'
'--continue-train', type=bool, default=False
'--predict-days', type=int, default=20, help='number of days for prediction return trend'
'--lr', type=float, default=1e-5, help='learning rate'
'--epoch', type=int, default=200, help='epoch number'
'--x-init', type=bool, default=True, help='xavier initial for weights'
'--bn', type=bool, default=True, help='use batch norm'
'--first-cnn', type=int, default=64, help='out channel of 1st cnn'
'--layers', type=int, default=3, help='number of cnn layers'
'--dropout', type=float, default=0.5, help='dropout ratio for mlp'
'--activation', type=str, default='lrelu', choices=['relu','lrelu'], help='activation function'
'--mp-size', type=str, default=(2,1), help='max pooling size'
'--flt-size', type=str, default=(5,3), help='cnn kernel size'
'--dilation', type=str, default=(2,1), help='cnn dilation size'
'--stride', type=str, default=(1,1), help='cnn stride size'
'--regular-store', type=bool, default=False

Besides, plot_comparison.ipynb and grad_cam.ipynb are used to plot the ROC curves and activation heatmap for different models, respectively.
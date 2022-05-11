tflr_new:
Fit k1, k2, k3 and then start predict for n iterations with gp-smbo


print('CONFIG :', n, tr_epochs, ft_epochs, lr, freeze)

CONFIG : 1 1000 1000 0.0001 False

CONFIG : 2 1000 1000 0.0001 True
CONFIG : 3 5000 5000 0.0001 True ----
CONFIG : 4 1000 5000 0.0001 True
CONFIG : 5 5000 5000 0.001 True

CONFIG : 6 5000 5000 0.0001 False ----
CONFIG : 7 1000 5000 0.0001 False
CONFIG : 8 5000 5000 0.001 False

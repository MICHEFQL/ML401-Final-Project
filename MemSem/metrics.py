from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import tensorflow as tf

def recall(y_true, y_pred):

    tf.config.run_functions_eagerly(True)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))

# df = pd.read_csv("/Users/poppy_puppet/Downloads/sheet22.csv", header=None)
# df = df.replace({
#     "pos": 1,
#     "neu": 0,
#     "neg": -1
# })
# y_true = df[0]
# y_pred = df[1]
# p = precision_score(y_true, y_pred , average=None)
# a = accuracy_score(y_true, y_pred)
# r = recall_score(y_true, y_pred, average=None)
# f1 = f1_score(y_true, y_pred, average=None)
# print("precision  "+str(p))
# print("recall  "+str(r))
# print("f1  "+str(f1))
# print("accuracy "+str(a))


# Epoch 1/64
#  1/13 ━━━━━━━━━━━━━━━━━━━━ 27s 2s/step - accuracy: 0.4375 - f1: 0.3137 - loss: 1.1671 - precision: 0.4211  2/13 ━━━━━━━━━━━━━━━━━━━━ 17s 2s/step - accuracy: 0.3906 - f1: 0.2728 - loss: 1.1691 - precision: 0.4095  3/13 ━━━━━━━━━━━━━━━━━━━━ 16s 2s/step - accuracy: 0.3750 - f1: 0.2612 - loss: 1.1690 - precision: 0.4309  4/13 ━━━━━━━━━━━━━━━━━━━━ 14s 2s/step - accuracy: 0.3652 - f1: 0.2538 - loss: 1.1718 - precision: 0.4328  5/13 ━━━━━━━━━━━━━━━━━━━━ 12s 2s/step - accuracy: 0.3622 - f1: 0.2537 - loss: 1.1674 - precision: 0.4476  6/13 ━━━━━━━━━━━━━━━━━━━━ 10s 2s/step - accuracy: 0.3591 - f1: 0.2548 - loss: 1.1648 - precision: 0.4555  7/13 ━━━━━━━━━━━━━━━━━━━━ 9s 2s/step - accuracy: 0.3576 - f1: 0.2539 - loss: 1.1627 - precision: 0.4578 - 8/13 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.3549 - f1: 0.2522 - loss: 1.1620 - precision: 0.4570 - 9/13 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.3532 - f1: 0.2484 - loss: 1.1614 - precision: 0.4520 -10/13 ━━━━━━━━━━━━━━━━━━━━ 4s 2s/step - accuracy: 0.3529 - f1: 0.2452 - loss: 1.1604 - precision: 0.4479 -11/13 ━━━━━━━━━━━━━━━━━━━━ 3s 2s/step - accuracy: 0.3531 - f1: 0.2425 - loss: 1.1597 - precision: 0.4441 -12/13 ━━━━━━━━━━━━━━━━━━━━ 1s 2s/step - accuracy: 0.3530 - f1: 0.2401 - loss: 1.1595 - precision: 0.4409 -13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.3528 - f1: 0.2375 - loss: 1.1591 - precision: 0.4388 -13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.3526 - f1: 0.2352 - loss: 1.1587 - precision: 0.4369 - recall: 0.1670 - val_accuracy: 0.3600 - val_f1: nan - val_loss: 1.1536 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 2/64
#  1/13 ━━━━━━━━━━━━━━━━━━━━ 17s 1s/step - accuracy: 0.2500 - f1: 0.1429 - loss: 1.1453 - precision: 0.3000  2/13 ━━━━━━━━━━━━━━━━━━━━ 16s 1s/step - accuracy: 0.2891 - f1: 0.1548 - loss: 1.1329 - precision: 0.3250  3/13 ━━━━━━━━━━━━━━━━━━━━ 14s 1s/step - accuracy: 0.3038 - f1: 0.1526 - loss: 1.1258 - precision: 0.3500  4/13 ━━━━━━━━━━━━━━━━━━━━ 13s 1s/step - accuracy: 0.3079 - f1: 0.1479 - loss: 1.1311 - precision: 0.3479  5/13 ━━━━━━━━━━━━━━━━━━━━ 11s 1s/step - accuracy: 0.3139 - f1: 0.1459 - loss: 1.1327 - precision: 0.3501  6/13 ━━━━━━━━━━━━━━━━━━━━ 10s 1s/step - accuracy: 0.3162 - f1: 0.1460 - loss: 1.1347 - precision: 0.3528  7/13 ━━━━━━━━━━━━━━━━━━━━ 8s 1s/step - accuracy: 0.3170 - f1: 0.1441 - loss: 1.1359 - precision: 0.3497 - 8/13 ━━━━━━━━━━━━━━━━━━━━ 7s 1s/step - accuracy: 0.3198 - f1: 0.1470 - loss: 1.1347 - precision: 0.3540 - 9/13 ━━━━━━━━━━━━━━━━━━━━ 5s 1s/step - accuracy: 0.3236 - f1: 0.1503 - loss: 1.1330 - precision: 0.3603 -10/13 ━━━━━━━━━━━━━━━━━━━━ 4s 2s/step - accuracy: 0.3260 - f1: 0.1517 - loss: 1.1323 - precision: 0.3629 -11/13 ━━━━━━━━━━━━━━━━━━━━ 3s 2s/step - accuracy: 0.3266 - f1: 0.1527 - loss: 1.1326 - precision: 0.3649 -12/13 ━━━━━━━━━━━━━━━━━━━━ 1s 2s/step - accuracy: 0.3282 - f1: 0.1532 - loss: 1.1325 - precision: 0.3674 -13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.3297 - f1: 0.1541 - loss: 1.1322 - precision: 0.3691 -13/13 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.3310 - f1: 0.1548 - loss: 1.1320 - precision: 0.3706 - recall: 0.1002 - val_accuracy: 0.2400 - val_f1: nan - val_loss: 1.1202 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 3/64
#  1/13 ━━━━━━━━━━━━━━━━━━━━ 18s 2s/step - accuracy: 0.2812 - f1: 0.0526 - loss: 1.1547 - precision: 0.1667  2/13 ━━━━━━━━━━━━━━━━━━━━ 16s 2s/step - accuracy: 0.3125 - f1: 0.0779 - loss: 1.1381 - precision: 0.2321  3/13 ━━━━━━━━━━━━━━━━━━━━ 15s 2s/step - accuracy: 0.3194 - f1: 0.1014 - loss: 1.1259 - precision: 0.2765  4/13 ━━━━━━━━━━━━━━━━━━━━ 13s 2s/step - accuracy: 0.3275 - f1: 0.1106 - loss: 1.1211 - precision: 0.3008  5/13 ━━━━━━━━━━━━━━━━━━━━ 12s 2s/step - accuracy: 0.3332 - f1: 0.1172 - loss: 1.1188 - precision: 0.3304  6/13 ━━━━━━━━━━━━━━━━━━━━ 10s 2s/step - accuracy: 0.3376 - f1: 0.1219 - loss: 1.1181 - precision: 0.3496  7/13 ━━━━━━━━━━━━━━━━━━━━ 9s 2s/step - accuracy: 0.3385 - f1: nan - loss: 1.1180 - precision: 0.3543 - re 8/13 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.3386 - f1: nan - loss: 1.1175 - precision: 0.3607 - re 9/13 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.3400 - f1: nan - loss: 1.1178 - precision: 0.3627 - re10/13 ━━━━━━━━━━━━━━━━━━━━ 4s 2s/step - accuracy: 0.3413 - f1: nan - loss: 1.1178 - precision: 0.3656 - re11/13 ━━━━━━━━━━━━━━━━━━━━ 3s 2s/step - accuracy: 0.3433 - f1: nan - loss: 1.1174 - precision: 0.3694 - re12/13 ━━━━━━━━━━━━━━━━━━━━ 1s 2s/step - accuracy: 0.3451 - f1: nan - loss: 1.1167 - precision: 0.3732 - re13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.3462 - f1: nan - loss: 1.1161 - precision: 0.3755 - re13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.3472 - f1: nan - loss: 1.1156 - precision: 0.3774 - recall: 0.0755 - val_accuracy: 0.3300 - val_f1: nan - val_loss: 1.2021 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 4/64
#  1/13 ━━━━━━━━━━━━━━━━━━━━ 18s 2s/step - accuracy: 0.3438 - f1: 0.0571 - loss: 1.0822 - precision: 0.3333  2/13 ━━━━━━━━━━━━━━━━━━━━ 17s 2s/step - accuracy: 0.3438 - f1: 0.0823 - loss: 1.0688 - precision: 0.3750  3/13 ━━━━━━━━━━━━━━━━━━━━ 15s 2s/step - accuracy: 0.3264 - f1: 0.0843 - loss: 1.0866 - precision: 0.3565  4/13 ━━━━━━━━━━━━━━━━━━━━ 13s 2s/step - accuracy: 0.3229 - f1: 0.0900 - loss: 1.0914 - precision: 0.3648  5/13 ━━━━━━━━━━━━━━━━━━━━ 12s 2s/step - accuracy: 0.3208 - f1: 0.0911 - loss: 1.0969 - precision: 0.3599  6/13 ━━━━━━━━━━━━━━━━━━━━ 10s 2s/step - accuracy: 0.3273 - f1: 0.0949 - loss: 1.0979 - precision: 0.3630  7/13 ━━━━━━━━━━━━━━━━━━━━ 9s 2s/step - accuracy: 0.3341 - f1: 0.0975 - loss: 1.0976 - precision: 0.3657 - 8/13 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.3407 - f1: 0.1011 - loss: 1.0963 - precision: 0.3742 - 9/13 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.3468 - f1: 0.1042 - loss: 1.0953 - precision: 0.3808 -10/13 ━━━━━━━━━━━━━━━━━━━━ 4s 2s/step - accuracy: 0.3509 - f1: 0.1072 - loss: 1.0950 - precision: 0.3851 -11/13 ━━━━━━━━━━━━━━━━━━━━ 3s 2s/step - accuracy: 0.3551 - f1: 0.1110 - loss: 1.0941 - precision: 0.3906 -12/13 ━━━━━━━━━━━━━━━━━━━━ 1s 2s/step - accuracy: 0.3589 - f1: 0.1151 - loss: 1.0932 - precision: 0.3967 -13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.3625 - f1: 0.1183 - loss: 1.0923 - precision: 0.4021 -13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.3655 - f1: 0.1211 - loss: 1.0915 - precision: 0.4067 - recall: 0.0724 - val_accuracy: 0.3200 - val_f1: nan - val_loss: 1.1635 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 5/64
#  1/13 ━━━━━━━━━━━━━━━━━━━━ 19s 2s/step - accuracy: 0.4062 - f1: 0.0571 - loss: 1.0634 - precision: 0.3333  2/13 ━━━━━━━━━━━━━━━━━━━━ 16s 2s/step - accuracy: 0.4141 - f1: 0.0845 - loss: 1.0585 - precision: 0.4375  3/13 ━━━━━━━━━━━━━━━━━━━━ 15s 2s/step - accuracy: 0.4253 - f1: 0.1097 - loss: 1.0526 - precision: 0.4914  4/13 ━━━━━━━━━━━━━━━━━━━━ 14s 2s/step - accuracy: 0.4323 - f1: 0.1227 - loss: 1.0482 - precision: 0.5278  5/13 ━━━━━━━━━━━━━━━━━━━━ 12s 2s/step - accuracy: 0.4296 - f1: nan - loss: 1.0517 - precision: 0.5241 - r 6/13 ━━━━━━━━━━━━━━━━━━━━ 10s 2s/step - accuracy: 0.4292 - f1: nan - loss: 1.0527 - precision: 0.5214 - r 7/13 ━━━━━━━━━━━━━━━━━━━━ 9s 2s/step - accuracy: 0.4259 - f1: nan - loss: 1.0559 - precision: 0.5159 - re 8/13 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.4244 - f1: nan - loss: 1.0583 - precision: 0.5130 - re 9/13 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.4209 - f1: nan - loss: 1.0610 - precision: 0.5076 - re10/13 ━━━━━━━━━━━━━━━━━━━━ 4s 2s/step - accuracy: 0.4175 - f1: nan - loss: 1.0635 - precision: 0.5047 - re11/13 ━━━━━━━━━━━━━━━━━━━━ 3s 2s/step - accuracy: 0.4162 - f1: nan - loss: 1.0646 - precision: 0.5053 - re12/13 ━━━━━━━━━━━━━━━━━━━━ 1s 2s/step - accuracy: 0.4148 - f1: nan - loss: 1.0660 - precision: 0.5040 - re13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.4132 - f1: nan - loss: 1.0673 - precision: 0.5039 - re13/13 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.4119 - f1: nan - loss: 1.0684 - precision: 0.5038 - recall: 0.0787 - val_accuracy: 0.2400 - val_f1: nan - val_loss: 1.1436 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 6/64
#  1/13 ━━━━━━━━━━━━━━━━━━━━ 18s 2s/step - accuracy: 0.4375 - f1: 0.1111 - loss: 1.0625 - precision: 0.5000  2/13 ━━━━━━━━━━━━━━━━━━━━ 16s 1s/step - accuracy: 0.4375 - f1: 0.1474 - loss: 1.0634 - precision: 0.5536  3/13 ━━━━━━━━━━━━━━━━━━━━ 14s 1s/step - accuracy: 0.4410 - f1: 0.1511 - loss: 1.0612 - precision: 0.5484  4/13 ━━━━━━━━━━━━━━━━━━━━ 13s 1s/step - accuracy: 0.4401 - f1: 0.1498 - loss: 1.0605 - precision: 0.5372  5/13 ━━━━━━━━━━━━━━━━━━━━ 12s 2s/step - accuracy: 0.4371 - f1: 0.1478 - loss: 1.0602 - precision: 0.5370  6/13 ━━━━━━━━━━━━━━━━━━━━ 10s 2s/step - accuracy: 0.4354 - f1: 0.1472 - loss: 1.0598 - precision: 0.5428  7/13 ━━━━━━━━━━━━━━━━━━━━ 9s 2s/step - accuracy: 0.4344 - f1: 0.1471 - loss: 1.0595 - precision: 0.5475 - 8/13 ━━━━━━━━━━━━━━━━━━━━ 7s 2s/step - accuracy: 0.4334 - f1: 0.1466 - loss: 1.0597 - precision: 0.5577 - 9/13 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.4327 - f1: 0.1459 - loss: 1.0592 - precision: 0.5702 -10/13 ━━━━━━━━━━━━━━━━━━━━ 4s 2s/step - accuracy: 0.4313 - f1: 0.1445 - loss: 1.0591 - precision: 0.5835 -11/13 ━━━━━━━━━━━━━━━━━━━━ 3s 2s/step - accuracy: 0.4303 - f1: 0.1428 - loss: 1.0592 - precision: 0.5968 -12/13 ━━━━━━━━━━━━━━━━━━━━ 1s 2s/step - accuracy: 0.4296 - f1: 0.1410 - loss: 1.0592 - precision: 0.6098 -13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.4292 - f1: 0.1394 - loss: 1.0591 - precision: 0.6222 -13/13 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.4289 - f1: 0.1380 - loss: 1.0591 - precision: 0.6329 - recall: 0.0794 - val_accuracy: 0.3500 - val_f1: nan - val_loss: 1.1062 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 7/64
#  1/13 ━━━━━━━━━━━━━━━━━━━━ 18s 2s/step - accuracy: 0.4062 - f1: 0.0588 - loss: 1.0636 - precision: 0.5000  2/13 ━━━━━━━━━━━━━━━━━━━━ 16s 2s/step - accuracy: 0.4375 - f1: 0.0858 - loss: 1.0460 - precision: 0.5625  3/13 ━━━━━━━━━━━━━━━━━━━━ 14s 1s/step - accuracy: 0.4444 - f1: 0.0946 - loss: 1.0488 - precision: 0.5694 13/13 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.4289 - f1: nan - loss: 1.0643 - precision: 0.5538 - recall: 0.0779 - val_accuracy: 0.2600 - val_f1: nan - val_loss: 1.1427 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 8/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.4709 - f1: nan - loss: 1.0470 - precision: 0.6031 - recall: 0.1356 - val_accuracy: 0.3800 - val_f1: nan - val_loss: 1.1157 - val_precision: 0.2667 - val_recall: 0.0312
# Epoch 9/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4391 - f1: nan - loss: 1.0330 - precision: 0.6792 - recall: 0.1101 - val_accuracy: 0.3400 - val_f1: nan - val_loss: 1.1569 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
# Epoch 10/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4247 - f1: 0.1843 - loss: 1.0398 - precision: 0.6382 - recall: 0.1095 - val_accuracy: 0.2800 - val_f1: nan - val_loss: 1.1365 - val_precision: 0.3125 - val_recall: 0.0156
# Epoch 11/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4794 - f1: 0.1959 - loss: 1.0266 - precision: 0.5847 - recall: 0.1230 - val_accuracy: 0.2900 - val_f1: nan - val_loss: 1.1565 - val_precision: 0.3000 - val_recall: 0.0156
# Epoch 12/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 23s 2s/step - accuracy: 0.4258 - f1: 0.2068 - loss: 1.0538 - precision: 0.5644 - recall: 0.1289 - val_accuracy: 0.2700 - val_f1: nan - val_loss: 1.1584 - val_precision: 0.3333 - val_recall: 0.0156
# Epoch 13/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4224 - f1: 0.2444 - loss: 1.0592 - precision: 0.5779 - recall: 0.1593 - val_accuracy: 0.2900 - val_f1: nan - val_loss: 1.1202 - val_precision: 0.2083 - val_recall: 0.0156
# Epoch 14/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4620 - f1: 0.2591 - loss: 1.0192 - precision: 0.7076 - recall: 0.1609 - val_accuracy: 0.2600 - val_f1: nan - val_loss: 1.2462 - val_precision: 0.1833 - val_recall: 0.0625
# Epoch 15/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4603 - f1: 0.3388 - loss: 1.0484 - precision: 0.5414 - recall: 0.2642 - val_accuracy: 0.3700 - val_f1: nan - val_loss: 1.1345 - val_precision: 0.1875 - val_recall: 0.0312
# Epoch 16/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4717 - f1: nan - loss: 1.0119 - precision: 0.7940 - recall: 0.1212 - val_accuracy: 0.2500 - val_f1: nan - val_loss: 1.1614 - val_precision: 0.3000 - val_recall: 0.0156
# Epoch 17/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4871 - f1: 0.2776 - loss: 0.9786 - precision: 0.9016 - recall: 0.1671 - val_accuracy: 0.3700 - val_f1: nan - val_loss: 1.1399 - val_precision: 0.1714 - val_recall: 0.0312
# Epoch 18/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4541 - f1: 0.3242 - loss: 1.0131 - precision: 0.6469 - recall: 0.2215 - val_accuracy: 0.3000 - val_f1: nan - val_loss: 1.1389 - val_precision: 0.1625 - val_recall: 0.0234
# Epoch 19/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4493 - f1: 0.2912 - loss: 0.9941 - precision: 0.7242 - recall: 0.1878 - val_accuracy: 0.2600 - val_f1: nan - val_loss: 1.2696 - val_precision: 0.1888 - val_recall: 0.1641
# Epoch 20/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4617 - f1: 0.3280 - loss: 1.0157 - precision: 0.5820 - recall: 0.2390 - val_accuracy: 0.3500 - val_f1: nan - val_loss: 1.1699 - val_precision: 0.1458 - val_recall: 0.0234
# Epoch 21/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4929 - f1: 0.2720 - loss: 0.9695 - precision: 0.7935 - recall: 0.1677 - val_accuracy: 0.2400 - val_f1: nan - val_loss: 1.2033 - val_precision: 0.2173 - val_recall: 0.0391
# Epoch 22/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.5110 - f1: 0.4413 - loss: 0.9506 - precision: 0.6877 - recall: 0.3288 - val_accuracy: 0.3700 - val_f1: nan - val_loss: 1.1932 - val_precision: 0.2500 - val_recall: 0.0859
# Epoch 23/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4678 - f1: 0.3134 - loss: 0.9603 - precision: 0.6501 - recall: 0.2102 - val_accuracy: 0.3800 - val_f1: nan - val_loss: 1.1606 - val_precision: 0.2604 - val_recall: 0.1094
# Epoch 24/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.5051 - f1: 0.3253 - loss: 0.9579 - precision: 0.7327 - recall: 0.2125 - val_accuracy: 0.3200 - val_f1: nan - val_loss: 1.1950 - val_precision: 0.1071 - val_recall: 0.0234
# Epoch 25/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.5378 - f1: 0.3780 - loss: 0.9402 - precision: 0.7485 - recall: 0.2578 - val_accuracy: 0.3600 - val_f1: 0.3283 - val_loss: 1.2129 - val_precision: 0.3808 - val_recall: 0.2891
# Epoch 26/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4705 - f1: 0.3449 - loss: 0.9689 - precision: 0.6418 - recall: 0.2415 - val_accuracy: 0.2900 - val_f1: nan - val_loss: 1.2501 - val_precision: 0.1742 - val_recall: 0.0547
# Epoch 27/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.5098 - f1: 0.4414 - loss: 0.9319 - precision: 0.7563 - recall: 0.3201 - val_accuracy: 0.2700 - val_f1: nan - val_loss: 1.3028 - val_precision: 0.1676 - val_recall: 0.0547
# Epoch 28/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4844 - f1: 0.4311 - loss: 0.9427 - precision: 0.5952 - recall: 0.3491 - val_accuracy: 0.2900 - val_f1: 0.2012 - val_loss: 1.1866 - val_precision: 0.3576 - val_recall: 0.1719
# Epoch 29/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.5089 - f1: 0.3843 - loss: 0.9320 - precision: 0.6812 - recall: 0.2722 - val_accuracy: 0.3000 - val_f1: nan - val_loss: 1.1821 - val_precision: 0.2167 - val_recall: 0.0938
# Epoch 30/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.4779 - f1: 0.3717 - loss: 0.9273 - precision: 0.7146 - recall: 0.2552 - val_accuracy: 0.3300 - val_f1: nan - val_loss: 1.1949 - val_precision: 0.2222 - val_recall: 0.0938
# Epoch 31/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.5179 - f1: 0.3705 - loss: 0.9360 - precision: 0.7092 - recall: 0.2527 - val_accuracy: 0.2800 - val_f1: nan - val_loss: 1.2065 - val_precision: 0.2583 - val_recall: 0.1562
# Epoch 32/64
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 24s 2s/step - accuracy: 0.5317 - f1: 0.4097 - loss: 0.9425 - precision: 0.7165 - recall: 0.2911 - val_accuracy: 0.3000 - val_f1: 0.1437 - val_loss: 1.2756 - val_precision: 0.2538 - val_recall: 0.1094
# Epoch 33/64
#  2/13 ━━━━━━━━━━━━━━━━━━━━ 16s 2s/step - accuracy: 0.5703 - f1: 0.4941 - loss: 0.8963 - precision: 0.6985 - recall: 0.3828sdjf
#  9/13 ━━━━━━━━━━━━━━━━━━━━ 6s 2s/step - accuracy: 0.5325 - f1: 0.4185 - loss: 0.9536 - precision: 0.6161 - recall: 0.3220^CTraceback (most rece
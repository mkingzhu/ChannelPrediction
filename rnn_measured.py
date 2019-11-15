from data import extract_file_content_binary_float64
from data import difference
from data import gen_scaler
from data import modify_x
from data import write_csv_file

import sys
import numpy

from keras.layers import Dropout, Input
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Model

neurons = 32
dropout = 0.2
batch_size = 10

data_start = 0
data_length = 44400
data_dim = 2

offset = 0
seq = 0

train_length = 300
train_time = 300
data_seg = 300
predict_length = 10
init_epochs = 20
epochs = 1

known_length = 19


def build_model():
    in1 = Input(shape=(known_length, data_dim))
    x1 = LSTM(neurons, return_sequences=False)(in1)
    x1 = Dropout(dropout)(x1)
    y1 = Dense(data_dim * predict_length, activation='linear')(x1)

    mo = Model(inputs=in1, outputs=y1)
    mo.compile(loss="mse", optimizer="adam")

    return mo


def get_scm_train_file_name():
    return 'data/measured/SNR15_AAPlantD1_2GHz_TX1_vpol_run3_pp_21.dat'


def get_min_max(d):
    min_ = sys.float_info.max
    max_ = sys.float_info.min

    start = 0
    while True:
        end = start + data_seg
        if end > train_length:
            break

        diff = difference(d[start:end])

        if diff.min() < min_:
            min_ = diff.min()
        if diff.max() > max_:
            max_ = diff.max()

        start = start + data_seg

    return min_, max_


def train(d, m, e):
    min_, max_ = get_min_max(d)
    scaler = gen_scaler(min_, max_)

    diff_x = []
    diff_y = []

    start = 0
    while True:
        end = start + data_seg
        if end > train_length:
            break

        diff = difference(d[start:end])
        scaled = scaler.transform(diff)

        x = modify_x(scaled, known_length)[:-known_length - predict_length]
        y = modify_x(scaled[known_length:], predict_length)[:-predict_length]
        y = y.reshape(-1, data_dim * predict_length)

        diff_x.append(x)
        diff_y.append(y)

        start = start + data_seg

    x = numpy.concatenate(diff_x, axis=0)
    y = numpy.concatenate(diff_y, axis=0)
    m.fit(x, y, batch_size=batch_size, epochs=e, validation_split=0, verbose=0)

    return scaler


def test(d, m, s):
    predictions = []
    errors = []
    start = 0
    while True:
        end = start + known_length + 1 + predict_length
        if end > len(d):
            break

        prediction, error = test_single(d[start:end], m, s)
        predictions.append(prediction)
        errors.append(error)

        start = start + predict_length
    return numpy.array(predictions).reshape(-1, 2), numpy.array(errors).reshape(-1, 1)


def test_single(d, m, s):
    diff = difference(d)
    scaled = s.transform(diff)

    x = scaled[:known_length].reshape(1, known_length, data_dim)

    y = s.inverse_transform(m.predict(x)[0].reshape(-1, data_dim))

    prediction = numpy.zeros((predict_length, data_dim))
    prediction[0] = d[known_length] + y[0]
    for i in range(1, predict_length):
        prediction[i] = prediction[i - 1] + y[i]

    return prediction, numpy.sum(numpy.power(prediction - d[known_length + 1:], 2), axis=1)


data_end = data_start + data_length

data = extract_file_content_binary_float64(get_scm_train_file_name(), (data_length, 2))
data = data[data_start:data_end, :data_dim]

model = build_model()

train_start = 0
train_end = train_length

scaler = train(data[train_start:train_end], model, e=init_epochs)

errors = []
predictions = []
while True:
    test_start = train_end + train_time
    test_end = test_start + train_time
    if test_end > data_end:
        break
    else:
        if test_end == data_end:
            prediction, error = test(data[test_start:test_end], model, scaler)
        else:
            prediction, error = test(data[test_start:test_end + known_length + 1], model, scaler)
        predictions.append(prediction)
        errors.append(error)

        if test_end == data_end:
            break

    train_start = train_start + train_time
    train_end = train_end + train_time
    scaler = train(data[train_start:train_end], model, e=epochs)

predictions = numpy.concatenate(predictions).reshape(-1, 2)
errors = numpy.concatenate(errors).reshape(-1, 1)

write_csv_file('matlab/SNR15_predictions_AAPlantD1_2GHz_TX1_vpol_run3_pp_21.csv', predictions)
write_csv_file('matlab/SNR15_errors_AAPlantD1_2GHz_TX1_vpol_run3_pp_21.csv', errors)

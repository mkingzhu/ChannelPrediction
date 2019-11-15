import numpy

from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler


def extract_file_content_binary_float64(filename, shape, offset=0):
    """
    extract the content of a binary file which contains 64-bit float data into a 2-d array
    :param filename: the name of the file that needs to be extracted
    :param shape: the shape of the 2-d array
    :param offset: the offset of the file from where the content is read
    :return: a 2-d array contains the file content
    """
    dt = numpy.dtype(numpy.float64).newbyteorder('>')
    with open(filename, 'rb') as bytestream:
        if offset != 0:
            bytestream.seek(offset)
        buf = bytestream.read(shape[0] * shape[1] * 8)
        data = numpy.frombuffer(buf, dtype=dt)
        data = data.reshape(shape)
        return data


def extract_file_content_float64(filename, shape):
    """
    extract the content of a csv file which contains 64-bit float data into a 2-d array
    :param filename: the name of the file that needs to be extracted
    :param shape: the shape of the 2-d array
    :return: a 2-d array contains the file content
    """
    dt = numpy.dtype(numpy.float64).newbyteorder('>')
    data = numpy.ndarray(shape=(0, 0), dtype=dt)
    with open(filename, 'r') as file_stream:
        for line in file_stream:
            data = numpy.append(data, numpy.array(line.split(',')).astype(dt))
    data = data.reshape(shape)
    return data


def convert2csv_float64(src_filename, dest_filename, shape):
    """
    convert a binary file which contains 64-bit float data into a csv file
    :param src_filename: the name of the file to be converted
    :param dest_filename: the name of the file to be generated
    :param shape: the shape of the data set
    :return:
    """
    data = extract_file_content_binary_float64(src_filename, shape)

    with open(dest_filename, 'w') as file_stream:
        for i in range(0, shape[0]):
            file_stream.write(",".join(data[i, ...].astype('str')) + "\n")


def write_csv_file(file_name, data):
    with open(file_name, 'w') as file_stream:
        for row in data:
            file_stream.write(','.join(row.astype(numpy.str)) + '\n')


def modify_x(train_x, num):
    x = train_x.copy()
    for i in range(num - 1):
        data = numpy.roll(train_x, -(i + 1), axis=0)
        x = numpy.concatenate((x, data), axis=1)
    return x.reshape(train_x.shape[0], num, train_x.shape[1])


def time_series_to_supervised(data_frame, lag=1):
    """
    frame a sequence as a supervised learning problem
    :param data_frame:
    :param lag:
    :return:
    """
    df = DataFrame(data_frame)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(data_set, interval=1):
    """
    create a difference series
    :param data_set: 
    :param interval: 
    :return: 
    """
    diff = list()
    for i in range(interval, len(data_set)):
        value = data_set[i] - data_set[i - interval]
        diff.append(value)
    return numpy.array(diff).reshape(data_set.shape[0] - interval, -1)


def inverse_difference(history, y_hat, interval=1):
    """
    invert difference value
    :param history:
    :param y_hat:
    :param interval:
    :return:
    """
    return y_hat + history[-interval]


def gen_scaler(min_, max_):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(numpy.concatenate((numpy.ones((1, 2)) * min_, numpy.ones((1, 2)) * max_), axis=0))
    return scaler

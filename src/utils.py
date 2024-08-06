
import csv
import numpy as np
import os

from config import TIMESTAMP, DATA_LABELS

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    plt = None

def read_csv(csv_file, delimiter=';', transpose=False, skip_header=True, dtype=None):

    data = []
    header = []

    with open(csv_file, 'r') as f:
        csv_data = csv.reader(f, delimiter=delimiter)
        
        if skip_header:
            temp = next(csv_data)
            header = {k: v for v, k in enumerate(temp)}
        
        for row in csv_data:
            data.append(row)

    data = np.array(data, dtype=dtype)

    if transpose:
        data = np.transpose(data)

    return data, header
    
def prepare_csv_data(data, prediction_labels, training_path, x_shape=None, data_labels = DATA_LABELS): # prediction_labels：预测标签列表

    prediction_indicies = [ data_labels[label] for label in prediction_labels ] # 将 预测标签 转换为 对应的索引，并存储在 prediction_indicies 列表中。
    
    counter = 0
    for row in data:
        if ([ row[index] for index in prediction_indicies ].count('NA')  == 0) & (os.path.isfile(f'{ training_path }/{ row[0] }.asc')) : 
            # 如果 预测标签索引对应的数据  没有缺失值（即值不为 'NA'）且文件存在于 training_path 中，则计数器 counter 增加1。
            counter += 1
    
    tempcount = 0
    while not 'median_data' in locals():
        tempcount += 1
        try:
            median_data, _ = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False)
        except FileNotFoundError:
            pass
    
    x_data = np.ones( ((counter,) + median_data.shape), dtype = np.int16 )
    y_data = []
    
    counter = 0
    for row in data:
        try:
            if [ row[index] for index in prediction_indicies ].count('NA') == 0: # 循环尝试读取文件，直到 median_data 被成功加载。 使用 read_csv 函数尝试读取第一个存在的 .asc 文件。
                median_data, _ = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False)
                y_data.append([ row[index] for index in prediction_indicies ])
                x_data[counter,:,:] = np.array(median_data).astype(np.int16) 
                counter += 1
        except FileNotFoundError:
            pass
    
    y_data = np.array(y_data)
    
    if not x_data.shape[0] == y_data.shape[0]:
        raise ValueError('Shape of x_data and y_data do not match')
    
    return x_data, y_data

def k_fold(x_data, y_data, k=3):  # 这段代码用于将数据划分为训练集和测试集，其中当前的折用作测试集，其他的折用作训练集。

    assert len(x_data) == len(y_data)
    
    x_split_length = len(x_data) // k
    y_split_length = len(y_data) // k

    x_folds = []
    y_folds = []

    for k_index in range(k - 1):
        x_folds += [ x_data[ k_index * x_split_length : (k_index + 1) * x_split_length ] ]
        y_folds += [ y_data[ k_index * y_split_length : (k_index + 1) * y_split_length ] ]

    x_folds += [ x_data[ (k - 1) * x_split_length : len(x_data) ] ] 
    y_folds += [ y_data[ (k - 1) * y_split_length : len(y_data) ] ]

    for fold_index in range(k):
        
        x_train = []
        y_train = []

        for train_index in range(k):
            if train_index != fold_index:  # 目的是确保当前循环中的  训练集折（train_index）不等于正在处理的 测试集折（fold_index）
                x_train.extend(x_folds[train_index])
                y_train.extend(y_folds[train_index])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = x_folds[fold_index]
        y_test = y_folds[fold_index]
    
        yield x_train, x_test, y_train, y_test

def plot_loss(history, file_name):  # 用于绘制训练和验证过程中损失值的变化曲线

    if plt is None: # 检查 matplotlib.pyplot 模块是否成功导入。如果没有导入 matplotlib.pyplot 模块，plt 将为 None，函数将直接返回，不执行任何操作。
        return
    
    plt.plot(history.history['val_loss'])  # history：这是模型训练过程中返回的 历史对象，包含训练和验证损失值 等信息。
    plt.plot(history.history['loss'])

    plt.title(str(TIMESTAMP))

    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.legend(['validation','training'], loc='upper left') # 添加图例（legend）来区分验证损失和训练损失曲线。

    plt.savefig(file_name)
    plt.gcf().clear() # 清除当前图像，释放内存。plt.gcf() 获取当前图像对象，clear() 方法清除该图像的内容。

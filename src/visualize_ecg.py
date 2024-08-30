import cv2
import os
import math
import glob
import csv
import logging

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mline

from matplotlib.collections import LineCollection
from tensorflow import keras

__MODEL_PATH__ = ""
__ECG_PATHS__ = [ '' + str(sub) + '.asc' for sub in list(range(600,610)) ]
__ECG_PATHS__ = [ '' for sub in list(range(139271,139281))]

__OUTPUT_PATH__ = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)

def load_ecg_data_from_file(filepath, delimiter=" "):
    
    data_item_list = []

    with open(filepath) as f:
        for row in csv.reader(f, delimiter=delimiter):
            data_item_list.append(row)

    return np.array(data_item_list)

def explain(image, model, class_index, layer_name, weighted=None): 
    # layer_name 是目标层

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)  # 这行代码的作用是将输入的 image 张量转换为 tf.float32 类型。
        conv_outputs, predictions = grad_model(inputs)  # 这里的 conv_outputs 对应的就是目标层的输出， predictions 是图像经过分类网络的输出
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs) # 计算给定 loss 相对于 conv_outputs 的梯度。

    guided_grads = (
        tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )  # 会将卷积层输出大于 0 且梯度大于 0 的部分保留，其他部分设为 0。这种操作有效地保留了对正激活和正梯度的梯度贡献，抑制了负值的影响，这也是 Guided Backpropagation 的核心思想。

    output = conv_outputs[0]  # 是目标层（或特征层）的输出
    guided_grad = guided_grads[0]  # 在目标层计算得到的梯度图
     
    if weighted is not None:
        guided_grad *= weighted.T

    weights = tf.reduce_mean(guided_grad, axis=(0, 1))  # 计算特征图各通道的平均权重。这些权重代表了各个特征图通道对最终分类决策的重要性。
    heatmap = tf.reduce_sum(tf.multiply(weights, output), axis=-1) 
    # 生成热力图，表示输入图像中各区域对预测结果的贡献。output 是从目标卷积层提取的特征图
    # weights 是一个 1D 张量，形状为 [channels]，表示每个通道的权重。

    
    image = np.squeeze(image)

    heatmap = cv2.resize(heatmap.numpy(), tuple([image.shape[1], image.shape[0]]))
    heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min())

    return cv2.applyColorMap(
        cv2.cvtColor((heatmap * 255).astype("uint8"), cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET
    )

def plot_ecg_image(ax, sensor_data, heatmap, name): # sensor_data: 包含ECG传感器数据的数组或列表，表示心电图的波形数据。
    # 这段代码的整体作用是将心电图的波形数据和与之相关的热图叠加在一起，并在图中显示。
    # 它通过去除额外的图像标记来简化视图，同时以灰色线条表示 ECG 波形，用色块表示热图。
    # 这种可视化方法可以帮助分析和解释心电图数据与某些热图数据之间的关系。

    ax.set_yticklabels([])  # 移除图像的X轴和Y轴的标签，以及隐藏所有轴线的边框。
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False) 
    ax.spines["left"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 

    ax.tick_params(axis=u'both', which=u'both',length=0)
    # 隐藏所有轴的刻度线，使得图像中不显示任何刻度。axis='both' 指的是X轴和Y轴，which='both' 指的是主刻度和次刻度，length=0 指定刻度线的长度为0。

    heatmap = heatmap / 255

    data_points = np.zeros((len(sensor_data), 1, 2))  
   # 初始化一个零矩阵 data_points，它的形状为 (len(sensor_data), 1, 2)。这个矩阵用于存储每个数据点的坐标，len(sensor_data)：表示数据点的数量，这个2表示 其中第一列是X坐标（时间），第二列是Y坐标（ECG数据值）。

    for row_index, point in enumerate(sensor_data): # 将每个数据点的位置和对应的ECG数据值填入 data_points 矩阵中。
        data_points[ row_index, 0, 0 ] = row_index  # row_index 是数据点的索引，用于表示时间（X坐标）。
        data_points[ row_index, 0, 1 ] = point  # point 是当前数据点的值（ECG数据值），用于表示Y坐标。

    segments = np.hstack([data_points[:-1], data_points[1:]])   # np.hstack 将 data_points 的相邻点组合起来，生成形状为 (n-1, 2, 2) 的线段数组。
    coll = LineCollection(segments, colors=[ [ 0.5, 0.5, 0.5 ] ] * len(segments), linewidths=(1.3)) 
    # LineCollection 是 matplotlib 中用于一次性绘制多个线段的类。这里它使用灰色（[0.5, 0.5, 0.5]）绘制所有线段，线宽为 1.3。
    
    ax.add_collection(coll)   # 将这些线段添加到 ax 上，以显示ECG波形。
    ax.autoscale_view()   # 自动调整视图范围，使得所有绘制的内容都能被完整显示。

    colors = np.mean(heatmap, axis=0)  # 计算每列热图数据的平均值，以生成颜色值。假设热图是一个二维数组，每列表示一个时间点对应的颜色强度。
    for c_index, color in enumerate(colors):
        ax.axvspan(c_index, c_index+1, facecolor=color)  # 绘制垂直的色块（即热图），每个色块的X范围为 [c_index, c_index+1]，颜色为计算得到的 color。这些色块表示热图叠加在ECG波形的背景上

def visualize_ecg_prediction_tf_explain(model_path, ecg_paths, output_path, class_indicies=[ 0 ]):
    # 该函数的主要功能是加载预训练的深度学习模型，对一组心电图（ECG）数据进行预测，并可视化模型的解释性输出。
    # model_path: 预训练模型的路径。
    # 需要处理的ECG数据文件路径列表。
    # 保存生成的可视化结果的输出路径。
    # 
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU，强制TensorFlow在CPU上运行。这通常用于避免资源争夺或调试。
    
    try:
        model = keras.models.load_model(model_path, compile=False) # 尝试加载指定路径的Keras模型。只加载模型结构和权重，而不编译模型（因为这里只需要预测，不需要训练）。
    except Exception:
        logging.error("Could not load %s!" % model_path)
        return
    
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # 从模型路径中提取模型的名称（去除文件扩展名），用于生成输出文件名。

    # 确定可视化层。 选择模型中的最后一个 Add 层和最后一个 Conv1D 层的名称，用于可视化。这些层通常对模型的决策过程至关重要。
    layer_names = [ layer.name for layer in model.layers if type(layer) == keras.layers.Add ][ -1 : ]
    layer_names.extend ([ layer.name for layer in model.layers if type(layer) == keras.layers.Conv1D ][ -1 : ])

    logging.info("Starting to visualize the predictions of %i ECGs" % len(ecg_paths))  # 记录日志，表示即将处理的ECG文件数量。

    for ecg_index, path in enumerate(ecg_paths):  # 迭代处理每个ECG文件，记录日志并尝试加载ECG数据。
        
        logging.info("Visualizing %s..." % path)
        try: 
            ecg_values = load_ecg_data_from_file(path)
        except:
            continue
        ecg_values = np.array(ecg_values, dtype=np.float32)
        ecg_values = np.reshape(ecg_values, (1, *ecg_values.shape))
        # 将其形状调整为 (1, length)，以适应模型的输入格式（通常模型期望输入为 batch_size x length 的形式）。
        
    
        cols = [ "Last Add", "Last Conv"]  # 设置列标题，对应于可视化的层。
        rows = [ 
            os.path.splitext(os.path.basename(path))[0]
        ]  # 设置行标题，这里是ECG文件的名称。

        ecg_name = os.path.splitext(os.path.basename(path))[0]
        
        plt.figure()
        plt.axis("off")  # 关闭轴线，使得图形只显示内容。

        fig, axes = plt.subplots(len(rows), len(cols), figsize=(9, 3))
        
        for ax, col in zip(axes, cols): # 为每个子图设置标题和行标签，便于区分不同层和ECG文件。
            ax.set_title(col, fontsize= 9)
        
        for index, (ax, row) in enumerate(zip(axes, rows)):
            ax.set_ylabel(row, fontsize= 9)

        # 迭代每个选定的层和分类索引，生成该层的解释性输出。
        for layer_index, layer_name in enumerate(layer_names):

            logging.info("Visualizing layer %s..." % layer_name)

            for class_index in class_indicies:

                output = explain(
                    ecg_values,
                    model=model,
                    layer_name=layer_name,
                    class_index=class_index
                )

                sensor_values = np.transpose(ecg_values.squeeze())[6]  #  提取特定通道的ECG数据（这里假设是第7个通道）。
                output = cv2.cvtColor(np.transpose(output, (1, 0, 2)), cv2.COLOR_BGR2RGB)  # 将解释性输出的颜色空间从BGR转换为RGB（OpenCV默认使用BGR）。
                plot_ecg_image(axes[ layer_index ], sensor_values, output, "%s_%s" % (ecg_name, layer_name))  # 调用之前定义的函数，绘制ECG波形与热图叠加的图像。

        file_format = "png"
        image_output_name = "%s_%s.%s" % (model_name, ecg_name, file_format)
    
        logging.info("Saving %s..." % image_output_name)

        plt.savefig(    # 将生成的图像保存到指定路径，文件名由模型名称和ECG名称组成，格式为PNG。
            fname=os.path.join(output_path, image_output_name),
            format=file_format,
            dpi=1000,
        )

        plt.cla()  # 清除当前图形的内容。
        plt.clf()  # 清除当前图形对象，释放内存。
        plt.close("all")  #  关闭所有图形窗口，防止内存泄漏
    
if __name__ == "__main__":

    visualize_ecg_prediction_tf_explain(__MODEL_PATH__, __ECG_PATHS__, __OUTPUT_PATH__)

import datetime
import os
import GPUtil

import tensorflow as tf

DATA_LABELS = {
    'TestID': 0, 'VentricularRate': 1, 'P_RInterval': 2, 'QRSDuration': 3,
    'Q_TInterval': 4, 'QTCFridericia': 5, 'PAxis': 6, 'RAxis': 7, 'TAxis': 8,
    'QRSCount': 9, 'QOnset': 10, 'QOffset': 11, 'POnset': 12, 'POffset': 13,
    'TOffset': 14, 'MCS': 15, 'Flatness': 16, 'Asymmetry': 17, 'Notch': 18,
    'NotchBinom': 19, 'Potassium': 20, 'Sodium': 21, 'Sex': 22, 'Age': 23,
    'BMI': 24, 'ECGCategory': 25, 'TPeakAmp_I': 26, 'TPeakAmp_II': 27, 
    'TPeakAmp_V1': 28, 'TPeakAmp_V2': 29, 'TPeakAmp_V3': 30, 'TPeakAmp_V4': 31,
    'TPeakAmp_V5': 32, 'TPeakAmp_V6': 33,
    'RPeakAmp_I': 34, 'RPeakAmp_II': 35,'RPeakAmp_V1': 36, 'RPeakAmp_V2': 37,
    'RPeakAmp_V3': 38, 'RPeakAmp_V4': 39,'RPeakAmp_V5': 40, 'RPeakAmp_V6': 41,
    'STM_I': 42, 'STM_II': 43, 'STM_V1': 44, 'STM_V2': 45,
    'STM_V3': 46, 'STM_V4': 47,'STM_V5': 48, 'STM_V6': 49,
    'fysiskaktiv_1': 50, 'styrketraening': 51, 'fysisk_fritid': 52,
    'age10': 53, 'fysisk_score': 54, 'fysisk_score_w': 55,
    'STJ_I': 56, 'STJ_II': 57, 'STJ_V1': 58, 'STJ_V2': 59,
    'STJ_V3': 60, 'STJ_V4': 61,'STJ_V5': 62, 'STJ_V6': 63,
    'SPeakAmp_I': 64, 'SPeakAmp_II': 65, 'SPeakAmp_V1': 66, 'SPeakAmp_V2': 67,
    'SPeakAmp_V3': 68, 'SPeakAmp_V4': 69, 'SPeakAmp_V5': 70, 'SPeakAmp_V6': 71,
    'RPeakAmp_aVL': 72, 'SPeakAmp_aVL': 73, 'TPeakAmp_aVL': 74, 'NormalECG': 75,
    'CornellVoltage': 76, 'SokolowLyonVoltage': 77, 'CornellVoltageCriterion': 78,
    'SokolowLyonCriterion': 79
}


ROOT_DIRECTORY = ''

SEED = 2

X_TRANSPOSE = False
IMAGE_SHAPE = (8, 600, 3)

TIMESTAMP = datetime.datetime.now().strftime("%y%m%d_%H%M%S")   #  获取当前时间，并将其格式化为YYMMDD_HHMMSS的字符串形式，用于命名文件。
PLOT_FILE = os.path.join(ROOT_DIRECTORY, 'models', str(TIMESTAMP) + '_loss_plot.png') # 保存损失曲线的文件路径。
MODEL_FILE = os.path.join('models', str(TIMESTAMP) + '_model.h5')  # 模型文件的完整路径。
LOG_FILE = os.path.join('models', str(TIMESTAMP) + '_log.txt')  # 日志文件的完整路径。
HISTORY_FILE = os.path.join('models', str(TIMESTAMP) + '_history.npy')  # 历史记录文件的完整路径。

EXPERIMENT_ROOT = os.path.join(ROOT_DIRECTORY, '..', 'experiments')  # 实验的根目录路径。
MODEL_PATH = os.path.join(ROOT_DIRECTORY, MODEL_FILE)  #  模型文件的完整路径。
LOG_PATH = os.path.join(ROOT_DIRECTORY, LOG_FILE)  # 日志文件的完整路径。
HISTORY_PATH = os.path.join(ROOT_DIRECTORY, HISTORY_FILE)  # 历史记录文件的完整路径。

EXPERIMENT_NAME = 'ECG_CNN_MODEL'  #  实验名称。

DATASET = 'gesustimeshift'
INPUT_TYPE = 'medians'

PREDICTION_LABELS = [  # 一个列表，包含要预测的标签（这里是Asymmetry）。
    'Asymmetry'
]

DEVICE_ID_LIST = GPUtil.getFirstAvailable(   #  获取可用的GPU设备列表，按照内存使用量排序，最大负载不超过90%，最大内存使用不超过80%。尝试3次，每次间隔15秒，并输出详细信息
    order = 'memory',
    maxLoad=0.9,
    maxMemory=0.8,
    attempts=3,
    interval=15,
    verbose=True
)

DEVICE_ID = DEVICE_ID_LIST[0]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 设置CUDA设备顺序为PCI总线ID。
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)  # 设置可见的CUDA设备为选择的DEVICE_ID。

EPOCHS = 100 
SAVE_AFTER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 40, 50, 62, 75, 87, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
SAVE_AFTER = [ ]
BATCH_SIZE = 32 
K_FOLDS = 5 
METRICS = [ 'mse', 'mae' ]
LOSS_FUNCTION = 'mean_squared_error' 
OPTIMIZER = tf.keras.optimizers.Nadam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

STRATEGY = tf.distribute.OneDeviceStrategy(device="/gpu:0")

GROUND_TRUTH_PATH = os.path.join(ROOT_DIRECTORY, '..', DATASET, 'asc', 'ground_truth.csv')
MEDIANS_PATH = os.path.join(ROOT_DIRECTORY, '..',  DATASET, 'asc', 'medians')
RHYTHM_PATH = os.path.join(ROOT_DIRECTORY, '..', DATASET, 'asc', 'rhythm')

if INPUT_TYPE == 'rhythm':
    ECG_PATH = RHYTHM_PATH
else:
    ECG_PATH = MEDIANS_PATH

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    pass

'''
本文件用于预处理数据，包括将wav数据转化为npy，将已预测的pitch（npy格式）和lables对齐并打包
对于cqt而言，将把读取的wav通过cqt变换转化为矩阵，并将矩阵切割为3秒的音频格式，最后合并cqt
label, pitch和cqt文件对应，也会依次处理
'''

import os
import time

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

filePath = './'
pitchfile = 'PredictPitch/'
sr = 44100
stride = 512
metadata = pd.read_csv("musicnet_metadata.csv")
poa = {1: 0, 7: 1, 41: 2, 42: 3, 43: 4, 44: 5, 71: 6, 72: 7, 74: 8, 61: 9, 69: 10}
cul = {1: 0, 7: 0, 41: 0, 42: 0, 43: 0, 44: 0, 71: 0, 72: 0, 74: 0, 61: 0, 69: 0}
cul_file = {1: 0, 7: 0, 41: 0, 42: 0, 43: 0, 44: 0, 71: 0, 72: 0, 74: 0, 61: 0, 69: 0}
inscul = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}

test_list = ['2303', '2191', '2382', '2628', '2416', '2556', '2298', '1819', '1759', '2106']

'''
1 Piano
7 Harpsichord 1
41 Violin
42 Cello
43 Viola
44 String Bass 5
71 Bassoon
72 Clarinet
74 Flute 8
61 Horn
69 Oboe 10
'''


# h: 泛音数目
def logCQT(y, h):
    # 采样频率
    # sr 采样频率， hop_length 连续CQT列之间的样本数， fmin 最低频率, A0=27.5Hz
    # n_bins 频率窗口的数量，可以理解为采样中的总窗口数， bins_per_octave 每个八度的窗口数量
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5*float(h), n_bins=88, bins_per_octave=12)
    # 振幅转分贝，并将最大值设置为0db，其余设置为赋值，并设置阈值为-80db
    delta = librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)
    # print(delta)
    # 最终返回归一化的数据
    return ((1.0/80.0) * delta) + 1.0


def chunk_data(f, trans=True):
    # 512是每帧的采样点数，这里s是采样率44100Hz在3秒内的帧数
    s = int(44100 * 3 / 512)

    # num = 88
    if trans:
        xdata = np.transpose(f)
    else:
        xdata = f
    x = []

    # 预计片段的长度(填充为258的倍数)
    length = int(np.ceil((int(len(xdata) / s) + 1) * s))
    # print(int(len(xdata) / s), int(len(xdata) / s) + 1, np.ceil((int(len(xdata) / s) + 1) * s))
    # 将app定义为一个剩余长度（预计长度length减实际长度xada.shape[0]）行，xdata列数列的一个全零矩阵
    app = np.zeros((length - xdata.shape[0], xdata.shape[1]))
    # 拼接xdata和app，这样一来能够保证所有音频矩阵的长度相同
    xdata = np.concatenate((xdata, app), 0)

    # 由于xdata被分为了length/s个片段数，我们要将所有片段整合为一个三维数组，最后得到一个[length/s, 88, 258]的切割片段
    for i in range(int(length / s)):
        data = xdata[int(i * s):int(i * s + s)]
        x.append(np.transpose(data[:258, :]))

    return np.array(x)


x_test, x_train, y_test, y_train = [], [], [], []
x_ptr, x_pte, cqcctr, cqccte = [], [], [], []
# 计算文件数
lin = 0
for root, dirs, files in os.walk(filePath):
    for file in files:
        path = os.path.join(root, file)
        if '.wav' in path:
            lin += 1

sums = 0
print("正在将数据处理为整体文件")
with tqdm(total=lin) as pbar:
    for root, dirs, files in os.walk(filePath):
        for file in files:
            path = os.path.join(root, file)
            if '.wav' in path:
                name = file.replace('.wav', '')
                x, sr = librosa.load(path, sr=sr)
                path = path.replace("_data", "_labels")
                path = path.replace(".wav", ".csv")



                # print(x.shape)
                x = logCQT(x, 1)
                # print(x.shape)

                # 音频采样点数
                samples = max((metadata[metadata['id'] == int(name)]['seconds'].values[0]) * sr, x.shape[1] * stride)
                
                # 采样点标记矩阵
                # csv_matrix = np.zeros([11, samples])
                df = pd.read_csv(path)
                p = True
                calf = []
                for index, row in df.iterrows():

                    start_time = row["start_time"]
                    end_time = row["end_time"]
                    instrument = row["instrument"]
                    '''
                    csv_matrix[poa[instrument], start_time:end_time] = instrument'''
                    if p:
                        cul_file[instrument] += 1
                        p = False
                    if instrument not in calf:
                        calf.append(instrument)
                    cul[instrument] += 1  # 用于统计数目统计样本

                inscul[len(calf)] += 1

                Yvec = np.zeros((11, x.shape[1]))
                csv_matrix = np.transpose(csv_matrix)
                # 对采样点进行标注
                for window in range(Yvec.shape[1]):
                    # 从这行代码能够猜出y的结构，y应该为一个二维数组，一个维度为采样点，另一个维度为标记，但我们文件格式不同，需要另外的处理办法
                    labels = csv_matrix[window * stride]
                    for label in labels:
                        if label in poa:
                            Yvec[poa[label], window] = 1
                
                x_ptemp = (np.load(filePath + pitchfile + name + '_norm.npy'))[:, 21:109]
                # chunk the data to 3 seconds
                x = chunk_data(x)
                sums += x.shape[0]

                x_ptemp = chunk_data(x_ptemp, trans=False)

                Yvec = chunk_data(Yvec)
                # print(path, x.shape)
                # 至此我们得到了切割完成的3秒样本，包含了音频原始数据和分类标签，x->y
                # x: 帧数*88*258
                # y: 帧数*11*258
                # 测试赋值与全体数据拼接
                if name in test_list:
                    if len(x_test) != 0:
                        x_test = np.concatenate((x, x_test), 0)
                        y_test = np.concatenate((Yvec, y_test), 0)
                        x_pte = np.concatenate((x_ptemp, x_pte), 0)
                    else:
                        x_test = x
                        y_test = Yvec
                        x_pte = x_ptemp
                else:
                    if len(x_train) != 0:
                        x_train = np.concatenate((x, x_train), 0)
                        y_train = np.concatenate((Yvec, y_train), 0)
                        x_ptr = np.concatenate((x_ptemp, x_ptr), 0)
                    else:
                        x_train = x
                        y_train = Yvec
                        x_ptr = x_ptemp
            else:
                continue

                pbar.update(1)


print("正在保存整合数据..")

np.savez('all.npz', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, x_pitchtr=x_ptr, x_pitchte=x_pte)
print("保存完毕")

print(cul)
print(cul_file)
print(inscul)
print(sums)

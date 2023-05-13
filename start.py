import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import math
features_num = 0

def starttrain(featuresstate={"cqcc":False, "pitch":False, "OD":False}, featurename="", har_num=1, modelid=2,od=1):
    # 数据集路径
    CSVFILE = '/home/aistudio/labels.csv'
    # 模型保存路径
    PATH='/home/aistudio/output-'+featurename+'/'
    print("Now " + featurename + " is going to train.")
    '''
    har_num  # 最高泛音序列倍数(务必>1，因为这里代表的是倍数，1倍等于没有变化)
    modelid  # 0: base_line, 1:base_line with resblock, 2: DRNet
    featuresstate  #训练考虑的特征
    '''
    datafile = "./data/data206584/"  # 数据集位置
    batch_size = 100
    num_labels = 11
    order_difference = od  # 差分次数
    featuresstate = featuresstate
    
    '''
    0: base_line
    1: base_line with ResBlock and in time dimension
    2: DRNet
    '''

    npz = np.load(datafile + "all.npz")
    print(npz.files)
    # data = cqt 
    cqtr = npz['x_train']
    cqtr = cqtr.astype("float32")
    cqte = npz['x_test']
    cqte = cqte.astype("float32")
    data = np.concatenate([cqtr, cqte], axis=0)
    cqtr, cqte = None, None

    odsum = []
    pitch = []
    cqcc = []
    # 设置迭代轮数
    EPOCH_NUM = 20

    # 差分
    if featuresstate["OD"]:
        delx = []
        for i in range(order_difference):
            if i == 0:
                delx = data
            delx = np.diff(delx, 1, axis=2)
            temp = np.zeros([data.shape[0], data.shape[1], 1])
            delx = np.concatenate([temp, delx], axis=2)
            odsum.append((np.array(delx)).astype("float32"))
        delx = None
        odsum = np.array(odsum)
        odsum = odsum.reshape([data.shape[0], data.shape[1], order_difference, data.shape[2]])

    ytr = npz['y_train']
    ytr = ytr.astype("float32")
    yte = npz['y_test']
    yte = yte.astype("float32")
    y = np.concatenate([ytr, yte], axis=0)
    ytr, yte = None, None

    def pitchnorm(pitch, har_num, Thre=False):
        # 泛音序列处理，先读取泛音序列，之后便不再考虑该数组
        minum = np.min(pitch)
        maxum = np.max(pitch)
        temp_pitch = pitch
        for i in range(har_num - 1):
            times = int(math.log2(i + 2) * 12)
            temp = np.roll(pitch, times, axis=1)
            temp[:, :times, :] = minum
            temp_pitch += temp

        pitch = temp_pitch
        temp_pitch = []
        # 归一化
        pitch = pitch/(maxum - minum)
        # 二值化
        if Thre:
            pitch = np.int64(data>0.5)
        return pitch

    # pitch
    if featuresstate["pitch"]:
        pitr = npz['x_pitchtr']
        pitr = pitr.astype("float32")
        pite = npz['x_pitchte']
        pite = pite.astype("float32")
        pitch = np.concatenate([pitr, pite], axis=0)
        pitr, pite = [], []

        pitch = pitchnorm(pitch, har_num)
        pitch = pitch.reshape([pitch.shape[0], pitch.shape[1], 1, pitch.shape[2]])

    # cqcc

    if featuresstate["cqcc"]:
        cqcc = np.zeros(data.shape)
        cqcc = cqcc.astype('float32')
        with tqdm(total=data.shape[0]) as pbar:
            for i in range(data.shape[0]):
                cqcc[i] = cv2.dct(data[i])
                pbar.update(1)
                
        cqcc = cqcc.reshape([data.shape[0], data.shape[1], 1, data.shape[2]])
        print(cqcc.shape)

    # 数据读取，扩充与合并
    # , "x_1OD.npy", "x_2OD.npy", "cqcc.npy"

    def reshac(data):
        if modelid == 0:
            data = data.reshape([data.shape[0], 1, data.shape[2], data.shape[1]])
        else:
            data = data.reshape([data.shape[0], data.shape[1], 1, data.shape[2]])
        return data

    def yre(y):
        if modelid == 0:
            y = y.reshape([y.shape[0], y.shape[1], y.shape[2], 1])
        else:
            y = y.reshape([y.shape[0], y.shape[1], 1, y.shape[2]])
        return y


    data = reshac(data)
    features_num = 1
    print(len(pitch),len(cqcc),len(odsum))
    if len(pitch) != 0:
        data = np.concatenate([data, pitch], axis=2)
        features_num += 1
        pitch = []
    if len(cqcc) != 0:
        data = np.concatenate([data, cqcc], axis=2)
        features_num += 1
        cqcc = []
    if len(odsum) != 0:
        data = np.concatenate([data, odsum], axis=2)
        features_num += order_difference
        odsum = []



    y = yre(y)

    bins = data.shape[1]

    print(data.shape)
    print(y.shape)

    # 拆分数据
    train_x,test_x,train_y,test_y = train_test_split(data, y, test_size=0.2, random_state=11, shuffle=True)
    data , y = [], []

    import paddle
    import paddle.nn.functional as F
    from paddle.nn import Conv1D, MaxPool1D, BatchNorm1D, Linear, Conv2D, MaxPool2D, BatchNorm2D, Dropout, LSTM

    print(train_x.shape)

    class CQTDataset(paddle.io.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __getitem__(self, index):
            data = self.data[index]
            label = self.labels[index]

            return data, label

        def __len__(self):
            return len(self.data)

    train_loader = paddle.io.DataLoader(CQTDataset(train_x, train_y), shuffle=True, batch_size=batch_size, num_workers=5, drop_last=True)
    test_loader = paddle.io.DataLoader(CQTDataset(test_x, test_y), shuffle=True, batch_size=batch_size, num_workers=5, drop_last=True)
    train_x,test_x,train_y,test_y = [], [], [], []

    class conv_block(paddle.nn.Layer):
        def __init__(self, in_channels, out_channels, kernel_size, padding):
            super(conv_block, self).__init__()
            self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, padding=padding)
            self.bn = BatchNorm2D(out_channels)
        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = F.relu(out)
            return out

    class base_line(paddle.nn.Layer):
        def __init__(self, num_classes=1):
            super(base_line, self).__init__()
            self.head = paddle.nn.Sequential(
                # 可以将卷积层改为一维或二维进行测试
                conv_block(1, 32, 7, 3),
                MaxPool2D([1,2],[1,2]),
                conv_block(32, 32, 7, 3),
                MaxPool2D([1,4],[1,4]),
                conv_block(32, 512, [1,1], [0,0]),
                MaxPool2D([1,11],[1,11]),
                conv_block(512, 512, [1,1], [0,0]),
                Conv2D(512, num_labels, (1,1), padding=(0,0)),
                )
            

        def forward(self, x):
            x = self.head(x)
            return x

    class ResBlock(paddle.nn.Layer):
        def __init__(self, in_channels, out_channels, features_num=1):
            super(ResBlock, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.features_num = features_num
            # in_channels, 卷积层的输入通道数
            # out_channels, 卷积层的输出通道数
            # stride, 卷积层的步幅
            # 创建第一个卷积层 1x1
            self.bn1 = BatchNorm2D(in_channels)
            self.conv1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=[1,3], stride=1, padding=[0,1])
            self.bn2 = BatchNorm2D(out_channels)
            self.conv2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=[1,3], stride=1, padding=[0,1])

            self.sk = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=[1,1], padding=[0,0])

        def forward(self, y):
            self.temp = y  # Restemp
            
            y = self.bn1(y)
            y = self.conv1(y)
            y = self.bn2(y)
            y = F.relu(y)
            y = self.conv2(y)
            y = self.bn2(y)

            y = paddle.add(y, self.sk(self.temp))
            return y




    class base_line_res(paddle.nn.Layer):
        def __init__(self, num_classes=1 , resnum=8):
            super(base_line_res, self).__init__()
            self.head = paddle.nn.Sequential(
                    BatchNorm2D(bins),       
                    # 视野为5的二维卷积核
                    Conv2D(bins, bins, 5, padding=2),
                    ResBlock(bins, bins*2),
                    Dropout(p=0.2),
                    MaxPool2D([2,1],[2,1]),
                    ResBlock(bins*2, bins*3),
                    Dropout(p=0.2),
                    MaxPool2D([3,1],[3,1]),
                    ResBlock(bins*3, bins*3),
                    BatchNorm2D(bins*3),
                    paddle.nn.ReLU(),
                    Conv2D(bins*3, num_labels, [3, 1], padding=[1, 0]))
            
        def forward(self, x):
            x = self.head(x)
            return x

    from paddle.nn import Conv1D, MaxPool1D, BatchNorm1D, Linear, Conv2D, MaxPool2D, BatchNorm2D, Dropout, LSTM, AvgPool2D

    class LsBlock(paddle.nn.Layer):
        def __init__(self, in_channels, out_channels, num_layers, features_num=1):
            super(LsBlock, self).__init__()
            self.out_channels = out_channels
            self.ls1 = LSTM(input_size=in_channels*features_num, hidden_size=out_channels*features_num, num_layers=num_layers, direction="bidirect")
            self.lsbn1 = BatchNorm2D(in_channels*features_num*2)
            self.pool1 = AvgPool2D([2,1],[2,1])
            self.bn = BatchNorm2D(out_channels)
            self.lsconv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=[1,3], stride=[2*features_num,1], padding=[0,1])

        def forward(self, x):
            x = x.reshape([x.shape[0], -1, x.shape[3]])  # LSTMtemp
            x = paddle.transpose(x, perm=[0, 2, 1])  # 维度交换
            x, (h, c) = self.ls1(x) #LSTM
            x = x.reshape([x.shape[0], x.shape[1], -1, self.out_channels])
            x = paddle.transpose(x, perm=[0, 3, 2, 1])  # 还原维度
            x = self.lsconv(x)
            x = self.bn(x)
            
            return x

    class DRNet(paddle.nn.Layer):
        def __init__(self, num_classes=1 , resnum=8):
            super(DRNet, self).__init__()
            self.head = paddle.nn.Sequential(
                    BatchNorm2D(bins),
                    LsBlock(bins, bins, 1, features_num=features_num),
                    Conv2D(bins, bins, [1, 5], padding=[0, 2]),
                    ResBlock(bins, bins*2, features_num=features_num),
                    Dropout(p=0.2),
                    MaxPool2D([2,1],[2,1]),
                    ResBlock(bins*2, bins*3),
                    Dropout(p=0.2),
                    MaxPool2D([3,1],[3,1]),
                    ResBlock(bins*3, bins*3),
                    BatchNorm2D(bins*3),
                    LsBlock(bins*3, bins*3, 2, 1),
                    BatchNorm2D(bins*3),
                    paddle.nn.ReLU(),
                    Conv2D(bins*3, num_labels, [3, 1], padding=[1, 0]))
            
        def forward(self, x):
            x = self.head(x)
            return x

    from TrainRunner import Runner
    loss_fn = paddle.nn.BCEWithLogitsLoss()


    if modelid == 0:
        m = base_line()
    elif modelid == 1:
        m = base_line_res()
    elif modelid == 2:
        m = DRNet()

    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    # 定义优化器
    opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=m.parameters(), weight_decay=1e-4)
    # opt = paddle.optimizer.SGD(learning_rate=0.01,
    #                            weight_decay=1e-4,
    #                            parameters=m.parameters())

    runner = Runner(m, opt, loss_fn, train_loader, test_loader, modelid, num_labels)

    import os

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    # 启动训练过程
    runner.train_pm(num_epochs=EPOCH_NUM, csv_file=CSVFILE, save_path=PATH)

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    runner.save_train_data(PATH)

    plt.figure()
    accImage = runner.acc_bet
    fig, ax = plt.subplots(1,1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.plot(accImage,c="g")
    plt.savefig(PATH + "acc.svg", dpi=300,format="svg")
    plt.show()
    plt.figure()
    fig, ax = plt.subplots(1,1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    accImage = runner.f1_ma_bet
    plt.plot(accImage,c="r")
    plt.savefig(PATH + "f1_ma_bet.svg", dpi=300,format="svg")
    plt.show()
    accImage = runner.f1_mi_bet
    fig, ax = plt.subplots(1,1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.plot(accImage,c="b")
    plt.savefig(PATH + "f1_mi_bet.svg", dpi=300,format="svg")
    plt.show()

    lossImage = runner.bestloss
    plt.plot(lossImage)
    plt.savefig("loss.svg", dpi=300,format="svg")
    plt.show()

if __name__ == '__main__':
    starttrain({"cqcc":False, "pitch":False, "OD":False})
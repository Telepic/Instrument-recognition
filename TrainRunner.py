import paddle
import numpy as np
import paddle.nn.functional as F


class Runner(object):
    def __init__(self, model, optimizer, loss_fn, train, test, modelid, num_labels):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train = train
        self.test = test
        self.bestloss = []
        # 记录全局最优指标
        self.best_f1 = 0
        self.modelid = modelid
        self.num_labels = num_labels
        self.ins = []
        self.acc_bet = []
        self.f1_mi_bet = []
        self.f1_ma_bet = []
        if num_labels != 11:
            self.ins = {0:"Piano",
            1:"Violin",
            2:"Cello",
            3:"Viola",
            4:"Bassoon",
            5:"Clarinet",
            6:"Horn",}
        else:
            self.ins = { 
            0:"Piano",
            1:"Harpsichord",
            2:"Violin",
            3:"Cello",
            4:"Viola",
            5:"String Bass",
            6:"Bassoon",
            7:"Clarinet",
            8:"Flute",
            9:"Horn",
            10:"Oboe"}
    # 定义训练过程
    def train_pm(self, **kwargs):
        print('start training ... ')
        self.model.train()
        
        num_epochs = kwargs.get('num_epochs', 0)
        csv_file = kwargs.get('csv_file', 0)
        save_path = kwargs.get("save_path", "/home/aistudio/output/")
        train_loader = self.train
        loss_rec = []

        for epoch in range(num_epochs):
            print("\n\n epoch", epoch, "now starting.")
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                temp = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)
                # 运行模型前向计算，得到预测值
                logits = self.model(temp) 
                avg_loss = self.loss_fn(logits, label)

                if batch_id % 50 == 0 and batch_id != 0:
                    print("batch_id: {}, loss is: {:.4f}".format(batch_id, float(avg_loss.numpy())))
                if batch_id % 2 == 0:
                    loss_rec.append(float(avg_loss.numpy()))
                
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                
            predsum, actsum, f1_mi, f1_ma = self.evaluate_pm(csv_file)
            self.model.train()

            if f1_ma > self.best_f1:
                print("This model is better than the best saved one, and now replacing ...")
                self.save_model(save_path)
                self.best_f1 = f1_ma
                # 为节约时间，可以不保存预测和实际结果
                # np.save(save_path + "predsum.npy", np.array(predsum))
                # np.save(save_path + "actsum.npy", np.array(actsum))

            # else: 动态学习率
                # self.optimizer = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=self.model.parameters(), weight_decay=1e-4)
                #  self.optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=self.model.parameters(), weight_decay=1e-4)
            
        self.bestloss = loss_rec

    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    
    @paddle.no_grad()
    def evaluate_pm(self, csv_file):
        # accuracies，正确率数组
        # np.mean(accuracies) 正确率
        self.model.eval()
        predsum = []
        actsum = []
        losses = []
        # 验证数据读取器
        valid_loader = self.test
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            pred = self.model(img)
            loss = self.loss_fn(pred, label)
            pred = F.sigmoid(pred)
            # 多分类，使用softmax计算预测概率
            predsum.append(pred.numpy())
            actsum.append(label.numpy())
            losses.append(loss.numpy())

        f1_mi, f1_ma, acc = self.accplu(predsum, actsum)
        print("[result] accuracy/loss/f1_micro/f1_macro: {:.4f}/{:.4f}/{:.4f}/{:.4f}".format(acc, np.mean(losses), f1_mi, f1_ma))
        self.acc_bet.append(acc)
        self.f1_mi_bet.append(f1_mi)
        self.f1_ma_bet.append(f1_ma)
        
        return predsum, actsum, f1_mi, f1_ma

    
    # 模型评估阶段，使用'paddle.no_grad()'控制不计算和存储梯度
    @paddle.no_grad()
    def predict_pm(self, x, **kwargs):
        # 将模型设置为评估模式
        self.model.eval()
        # 运行模型前向计算，得到预测值
        logits = self.model(x)
        return logits
    
    def save_train_data(self, path):
        np.save(path + "acc_bet.npy", np.array(self.acc_bet))
        np.save(path + "f1_mi_bet.npy", np.array(self.f1_mi_bet))
        np.save(path + "f1_ma_bet.npy", np.array(self.f1_ma_bet))

    def accplu(self, pred, actual):
        pred = np.array(pred)
        actual = np.array(actual)
        # pred, actual
        # [100, 11, 258, 1]
        # results: 
        # [f1, acc, 

        # F1, 采用动态阈值模式:
        F1 = 0
        bp = 0
        actual = actual.astype(int)
        ac = np.sum(actual)
        besttre = 0
        temp = []
        F1_matrix = []
        F1_macro_best = 0
        cont = 0
        for i in range(100):
            thre = i * 0.01
            temp = (pred>thre).astype(int)
            F1_macro = self.f1_marco(temp, actual) 
            pc = np.sum(temp)
            # 节约算量的步骤，理论上大于阈值的预测标签数量不变时，一定程度上能够说明TP数量不变
            if pc == bp:
                continue
            else:
                bp = pc
            # 两个矩阵相加再除以2，两者都为1时输出1，从而达到计算TP的效果
            if pc == 0 or ac == 0:
                F1_n = 0
                continue
            TP = np.sum((np.int64((temp + actual) /2) == np.ones(temp.shape)).astype("int"))
            P = TP / pc
            R = TP / ac
            if P == 0 or R == 0:
                F1_n = 0
                continue
            F1_ma = 2*(P*R)/(R+P)
            F1_n = np.mean(F1_macro)
            # 计算小于的连续阈值
            if F1_n > F1:
                F1 = F1_n
                besttre = thre
                F1_matrix = F1_macro
                F1_macro_best = F1_ma
            else:
                if i>50:
                    cont += 1
            # 当F1在阈值不断增大后呈现10次连续减小，则跳过之后的所有运算，应该能节约50%以上的计算时间
            if cont>5:
                break
        for i in range(self.num_labels):
            print(self.ins[i], ": ", F1_matrix[i])
            
        acc = (temp == actual)
        acc = acc.astype("int")
        acc = np.mean(acc)
        print("best threshold:", besttre)
        return F1_macro_best, F1, acc

    def f1_micro(self, x, y):
        if np.sum(y) == 0 or np.sum(x) == 0:
            return 0
        TP = np.sum((np.int64((x + y) /2) == np.ones(x.shape)).astype("int"))
        P = TP / np.sum(x)
        R = TP / np.sum(y)
        if P == 0 or R == 0:
            return 0
        F1_n = 2*(P*R)/(R+P)
        return F1_n
    
    def f1_marco(self,x ,y):
        p = []
        for i in range(self.num_labels):
            ft = self.f1_micro(x[:, :, i, :, :], y[:, :, i, :, :])
            p.append(ft) 
        p = np.array(p)
        return p

    def save_model(self, save_path):
        paddle.save(self.model.state_dict(), save_path + 'palm.pdparams')
        paddle.save(self.optimizer.state_dict(), save_path + 'palm.pdopt')
    
    def load_model(self, model_path):
        model_state_dict = paddle.load(model_path)
        self.model.set_state_dict(model_state_dict)
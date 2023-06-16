# Instrument-recognition
乐器识别，这是利用深度学习和基于MusicNet的乐器识别源代码，详细请参考论文：（还未发表）

## 文件说明
musicnet：对数据集MusicNet的预处理，包括统一化数据（剪裁为3s），对音级的预测（pitch提取）和音频与音级的对齐的源代码文件。<br>
output：乐器识别模型的参数配置，palm.pdopt为优化器参数，palm.pdparams为模型参数。<br>
thickstun：Thickstun的音级预测模型，采用tensorflow实现，实现该预测时请确保您的计算机有相关的框架。<br>
Predict_music：根据paddle中的已训练模型对音频进行11种乐器的预测。<br>
start.py：直接调用的训练源代码，由Train.ipynb调用。<br>
Train.ipynb： 所有特征组合的训练源代码。<br>
Train_Process.ipynb：训练过程的分步骤ipynb文件，该文件同start.py相同。<br>
TrainRunner.py：训练过程的封装文件，直接调用用于模型训练。<br>


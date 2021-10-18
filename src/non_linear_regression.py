import numpy as np
import matplotlib.pyplot as plt
import keras

# 建立随机数作为数据集
x_data = np.linspace(-0.5, 0.5, 200)  # 从-0.5到0.5取均匀的200个数
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

print(x_data.shape)
print(y_data.shape)

# 使用keras的Sequential函数建立一个顺序模型
model = keras.Sequential()
# 在模型中添加全连接层和激活函数，注意这里与线性回归模型不同的是，在每个全连接层之间添加激活函数，以将模型变为非线性的，以此能够拟合非线性数据

# sigmoid是平滑（smoothened）的阶梯函数（step function），可导（differentiable）。sigmoid可以将任何值转换为0~1概率，用于二分类。
# tanh减轻了梯度消失的问题。tanh的输出和输入能够保持非线性单调上升和下降关系，符合BP（back propagation）网络的梯度求解，容错性好，有界。
# model.add(keras.layers.Dense(units=10, input_dim=1, activation='relu'))
# relu整流线性单元，激活部分神经元，增加稀疏性

model.add(keras.layers.Dense(units=10, input_dim=1, activation='relu'))
model.add(keras.layers.Dense(units=1))

# model.add(keras.layers.Dense(units=10, input_dim=1))
# model.add(keras.layers.Activation('tanh'))
# model.add(keras.layers.Dense(units=1))
# model.add(keras.layers.Activation('tanh'))
model.summary()

# 定义优化算法，sgd优化重新定义其学习率，以较快地完成学习，如果使用默认学习率，建议增加训练迭代次数
# sgd = keras.optimizers.SGD(lr=0.3)
adam = keras.optimizers.Adam()

# 优化方法：sgd（随机梯度下降算法）
# 损失函数：mse（均方误差）
model.compile(optimizer=adam, loss='mse')

# for step in range(3001):
#     # 每次训练一个批次
#     cost = model.train_on_batch(x_data, y_data)
#     # 每500个batch打印一次cost
#     if step % 500 == 0:
#         print('cost:', cost)

model.fit(x_data, y_data, batch_size=8, epochs=200, verbose=2)

# x_data输入到网络中，得到预测值y_pred
y_pred = model.predict(x_data)

# 显示随机点的结果
plt.scatter(x_data, y_data)

# 显示预测点的结果
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()

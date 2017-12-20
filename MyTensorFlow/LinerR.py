import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 随机生成1000个点，围绕在y=0.1x+0.3的直线周围
vectors_set = []
for _ in range(1000):
    x1 = np.random.normal(0.0,0.55)
    y1 = 0.1 * x1 + 0.3 + np.random.normal(0.0,0.23)
    vectors_set.append([x1,y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

#plt.scatter(x_data,y_data,c = 'r')
## 生成一些样本
#plt.show()

# 生成1维的W矩阵，取值是[-1,1]之间的随机数
W = tf.Variable(tf.random_uniform([1],-1,1,dtype=tf.float32),name='W')
# 生成1维的b矩阵，初始值是0
b = tf.Variable(0.0,name='b')
# 经过计算得出预估值y
y = W * x_data + b

# 以预估值y和实际值y_data之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data),name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.1)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss,name='train')

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# 初始化的W和b是多少
print("W =",sess.run(W),"b =",sess.run(b),"loss =",sess.run(loss))
for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        print("W =",sess.run(W),"b =",sess.run(b),"loss =",sess.run(loss))

plt.scatter(x_data,y_data,c = 'r')
plt.plot(x_data,sess.run(y))
plt.show()
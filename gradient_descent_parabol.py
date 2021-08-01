import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation

# hàm fx
def cost(x):
    m = A.shape[0]
    return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2 # hàm tính khoảng cách, norm lấy 2 giá trị Ax - b và Euclidean norm

# đạo hàm fx
def grad(x):
    m = A.shape[0]
    return 1/m * A.T.dot(A.dot(x) - b)

def check_grad(x):
    eps = 1e-4 
    g = np.zeros_like(x) # g là f' tính theo công thức
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        g[i] = (cost(x1)-cost(x2))/(2*eps)
    
    g_grad = grad(x)
    if np.linalg.norm(g-g_grad) > 1e-7:
        print("Warning: Check gradient function")

def gradient_descent(x_init, learnig_rate, iteration):
    x_list = [x_init]

    # interaction là định sẵn số lần di chuyển x0
    for i in range(iteration):
        x_new = x_list[-1] - learnig_rate*grad(x_list[-1]) # dịch chuyển x0
        
        if np.linalg.norm(grad(x_new))/len(x_new) < 0.5: # Normalize (Chuẩn hóa) và check xem thuật toán nên dừng hay chưa, dừng khi đạo hàm gần cực trị
            break
        x_list.append(x_new)

    return x_list

# Random data
A = np.array([[2, 5, 7, 9, 13, 16, 18, 20, 24, 29, 34, 37, 39, 40]]).T
b = np.array([[2, 3, 6, 15, 22, 28, 34, 36, 40, 37, 35, 31, 28, 25]]).T


# Draw data
fig1 = plt.figure("GD for Linear Regression")
# Thay doi chieu dai truc x
ax = plt.axes(xlim=(-40, 60), ylim=(-40, 60))
plt.plot(A, b, 'ro')

# Create A square
x_square = np.array([A[:, 0]**2]).T

# Combine x_square and A
A = np.concatenate((A, x_square), axis=1)

# Add one to A
ones = np.ones((A.shape[0], 1), dtype=np.int8) # np.ones() la ham dac biet tao toan vecto 1
A = np.concatenate((ones, A), axis=1) 

# Use fomular
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b) # (A.T*A)^-1*A.T* => tim dc x la gia tri a,b,c trong y = ax^2 + bx +c

# x0 chua 2 diem tu 1 den 46
x0_gd = np.linspace(1, 46, 10000)
y0_sklearn = x[0][0] + x[1][0]*x0_gd + x[2][0]*x0_gd*x0_gd

plt.plot(x0_gd, y0_sklearn, color="green")




# Random initial line
x_init = np.array([[-11.],[3.],[-0.3]])
y0_init = x_init[0][0] + x_init[1][0]*x0_gd + x_init[2][0]*x0_gd*x0_gd # Tuong duong y = b + ax
plt.plot(x0_gd, y0_init, color="black")

check_grad(x_init)

# run gradient descent
iteration = 90
learning_rate = 0.0000001

x_list = gradient_descent(x_init, learning_rate, iteration)
print(x_list)

# draw x_list (solution by GD)
for i in range(len(x_list)):
    y0_x_list = x_list[i][0][0] + x_list[i][1][0]*x0_gd +x_list[i][2][0]*x0_gd*x0_gd # y = b + ax
    plt.plot(x0_gd, y0_x_list, "black", alpha=0.3)

# print(len(x_list))

# Draw animation 
line , = ax.plot([], [], color = "blue")
def update(i): 
    y0_gd = x_list[i][0][0] + x_list[i][1][0]*x0_gd +x_list[i][2][0]*x0_gd*x0_gd
    line.set_data(x0_gd, y0_gd)
    return line,

iters = np.arange(1, len(x_list), 1) # tao list bat dau tu 1, do dai = x_list, cach nhau 1 don vi
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True) # interval la toc do cua animation, blit la lam muot

# legend for plot
plt.legend(('Value in each GD iteration', 'Solution by formular', 'Inital value for GD'), loc=(0.52, 0.1))
ltext = plt.gca().get_legend().get_texts()

# title
plt.title('Grandient Descent Animation')

plt.show()

# Plot cost per iteration to determine when to stop
# cost_list = []
# iter_list = []
# for i in range(len(x_list)):
#     iter_list.append(i)
#     cost_list.append(cost(x_list[i]))

# plt.plot(iter_list, cost_list)
# plt.xlabel("Iteration")
# plt.ylabel("Cost value")

# plt.show()
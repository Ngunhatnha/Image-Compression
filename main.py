import math
import cv2
import numpy as np
from numpy.linalg import eig
import matplotlib.image as image
from sklearn.decomposition import PCA

saveSigma = list()
saveMean = list()
matrixB = list() #Tạo ma trận để chứa m các eigenvector

#Tính mean
def meanCalc(matrix):
    sum = 0
    for i in matrix:
        sum = sum + i
    return sum/(len(matrix))

#Tính sigma
def sigmaCalc(matrix, mean):
    sumSquare = 0
    for i in matrix:
        sumSquare = sumSquare + i*i
    return math.sqrt(sumSquare/(len(matrix)) - mean*mean)

#Standardize matrix
def standardize(matrix):
    for line in matrix: # Chạy từng hàng trong ma trận ban đầu
        mean = meanCalc(line) # Tính mean của hàng
        sigma = sigmaCalc(line, mean) #Tính sigma hàng
        saveMean.append(mean) #Lưu mean vào 1 ma trận để bước cuối reconstruct
        saveSigma.append(sigma)#Lưu sigma vào 1 ma trận để bước cuối reconstruct
        for i in range(0, len(line)):
            line[i] = (line[i] - mean)/ sigma #Giá trị sau khi chuẩn hóa = (Xcũ - mean)/ sigma


def PCA_Algorithm(matrix, m): #PCA giảm  chiều ma trận về m
    matrixB.clear()
    standardize(matrix)
    covMatrix = np.matmul(matrix, np.transpose(matrix)) / 3 # Tính covariance matrix = (1/n)*Matrix*Transpose(Matrix)
    eigenValue, eigenVector = eig(covMatrix)#Tính eigenValue vs eigenvector của ma trận covariance
    idx = np.argsort(eigenValue) #sắp sếp eigen value theo thứ tự giảm dần
    idx = idx[::-1]
    eigenVector = eigenVector[:, idx]
    eigenValue = eigenValue[idx]#sắp sếp eigen vector theo thứ tự giảm dần

    for line in eigenVector: #gán các eigenvector theo thứ tự giảm dần vào ma trận B
        temp = list()
        for i in range(0, m):
            temp.append(line[i])
        matrixB.append(temp)
    projMatrix = np.matmul(matrixB, np.matmul(np.transpose(matrixB), matrix))

    reconstruct(projMatrix)
    # print(projMatrix)
    # print("-------------------")
    #return np.matmul(np.transpose(matrixB), matrix) #Trả về ma trận đã giảm chiều = transpose(matrixB)*matrix
    return projMatrix

def reconstruct(matrix):
    for lineIdx in range(0, len(matrix)):
        for i in range(0, len(matrix[lineIdx])):
            matrix[lineIdx][i] = matrix[lineIdx][i]*saveSigma[lineIdx] + saveMean[lineIdx]


img = image.imread("anh1.jpg")
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.size, img.shape, img.ndim)
print("Shape trước khi giảm chiều: ",RGB_img.shape)
cv2.imwrite("oldsave1.jpg", RGB_img)
cv2.imshow("Hi", RGB_img)
cv2.waitKey(0)
blue, green, red = cv2.split(RGB_img)
print(blue)
print(green)

blue, green, red = blue / 255, green / 255, red / 255

print("Blue channel")
print(type(blue))
print(blue)
print("Green channel")
print(type(green))
print(green)
print("Red channel")
print(type(red))
print(red)
m = int(input("Nhập m: "))

pca_b_trans = PCA_Algorithm(blue, m)
pca_r_trans = PCA_Algorithm(red, m)
pca_g_trans = PCA_Algorithm(green, m)


img_compressed = cv2.merge((pca_b_trans, pca_g_trans, pca_r_trans))
print(img_compressed.shape)
save = np.floor(img_compressed*255)
cv2.imwrite("anh1_50.jpg", save)

# print(pca_b_trans)
# combined = np.array([pca_r, pca_g, pca_b])
# print("Shape sau khi đã giảm chiều: ",combined.shape)
# print("Giá trị sau khi combine 3 ma trận: ",combined)
cv2.imshow("Hi", img_compressed)
cv2.waitKey(0)
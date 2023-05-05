import numpy as np
import math
import cv2
Q1 = cv2.imread('Q1.jpg')
Q2 = cv2.imread('Q2.jpg')
Q3 = cv2.imread('Q3.jpg')

def create_image(image):
    h, w = image.shape[0], image.shape[1]
    new_img = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            mean = int(0)
            for c in range(3):
                mean += image[i][j][c]
            mean =  np.uint8(mean / 3)
            new_img[i][j] = mean
    return new_img

a1 = create_image(Q1)
intensity_count= [0]*256
for i in range(a1.shape[0]):
    for j in range(a1.shape[1]):
        intensity_count[a1[i][j]] += 1

equ_1 = []
inverse = {}
for i in range(len(intensity_count)):
    value = sum(intensity_count[: i+1]) * 255 / (a1.shape[0] * a1.shape[1])
    if round(value) not in inverse: inverse[round(value)] = []
    inverse[round(value)].append(i)
    equ_1.append(np.uint8(value))

for i in range(a1.shape[0]):
    for j in range(a1.shape[1]):
        a1[i][j] = equ_1[a1[i][j]]

cv2.imwrite('a1.jpg', a1)
#*************************************************
Q2 = create_image(Q2)
Q2_intensity = [0]*256
for i in range(Q2.shape[0]):
    for j in range(Q2.shape[1]):
        Q2_intensity[Q2[i][j]] += 1

equ_2_inverse = {}
for i in range(len(Q2_intensity)):
    value = 255 * sum(Q2_intensity[:i+1]) / (Q2.shape[0]* Q2.shape[1])
    if round(value) not in equ_2_inverse: equ_2_inverse[round(value)] = []
    equ_2_inverse[round(value)].append(i)

for i in range(a1.shape[0]):
    for j in range(a1.shape[1]):
        try: a1[i][j] = equ_2_inverse[a1[i][j]][-1]
        except :
            above = min([key for key in equ_2_inverse.keys() if a1[i][j] < key])
            a1[i][j] = equ_2_inverse[above][-1]
cv2.imwrite('a2.jpg', a1)

#*************************************************
Q3 = create_image(Q3)

def gaussian_kernel(i, j):
    global Q3
    K = 1
    kernel_size = 5
    offset = kernel_size//2
    sigma = 25

    sum = 0
    coeff_sum = 0
    for x in range(kernel_size):
        s = x - offset
        for y in range(kernel_size):
            t = y - offset
            r_square = pow(s, 2) + pow(t, 2)
            coeff = K * math.exp(-r_square / (2* pow(sigma, 2)))
            coeff_sum += coeff
            if i + t >= 0 and j + s >= 0 and i + t < Q3.shape[0] and j + s < Q3.shape[1]:
                sum +=  Q3[i + t][j + s] * coeff
            else: continue            
    return sum/coeff_sum

a3 = np.zeros(Q3.shape)
for i in range(a3.shape[0]):
    for j in range(a3.shape[1]):
        a3[i][j] = np.uint8(gaussian_kernel(i, j))

cv2.imwrite('a3.jpg', a3)
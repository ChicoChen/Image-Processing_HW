import numpy as np
import math
import cv2

def pre_processing(img_name):
    I = cv2.imread(img_name)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    h, w = I.shape[0], I.shape[1]
    new_img = np.zeros((h*2, w*2), dtype= np.intc )
    for i in range(h):
        for j in range(w):
            new_img[i][j] = I[i][j] * pow(-1, i + j)
    return new_img

Q1 = pre_processing("test1.tif")
Q2 = pre_processing("test2.tif")

def fourier_processing(img, filename1, filename2, D0):
    # Fourier transform
    img = np.fft.fft2(img)
    temp = np.abs(img)
    temp = cv2.normalize(temp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
    temp = cv2.equalizeHist(temp)
    cv2.imwrite(filename1, temp)
    
    # frequency domain processing
    H = np.zeros(img.shape)
    h0, w0 = img.shape[0]//2, img.shape[1]//2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            H[i][j] = math.exp(-(pow(i- h0, 2) + pow(j- w0, 2)) / (2*pow(D0, 2)))
    
    img = np.multiply(img, H)
    temp = np.abs(img)
    temp = cv2.normalize(temp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
    temp = cv2.equalizeHist(temp)
    cv2.imwrite(filename2, temp)
    
    # inverse Fourier
    img = np.fft.ifft2(img)[:h0, :w0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = img[i][j] * pow(-1, i+j)
    
    return img.astype(np.uint8)

Q1 = fourier_processing(Q1, "f1_normalize.jpg", "f1_processed.jpg", 50)
cv2.imwrite("A1.jpg", Q1)

Q2= fourier_processing(Q2, "f2_normalize.jpg", "f2_processed.jpg", 30)
cv2.imwrite("A2.jpg", Q2)

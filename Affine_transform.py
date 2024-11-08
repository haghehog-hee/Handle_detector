import cv2 as cv
import numpy as np
import os

#path = "C:\\Users\\MuhametovRD\\AppData\\Roaming\\EasyClient\\Picture\\"
path = "C:\\Tensorflow\\Dataset\\AffineSide\\raw\\"
savepath = "C:\\Tensorflow\\Dataset\\AffineSide\\rotated\\img2\\"
thresh = 0
IMAGE_PATHS = os.listdir(path)
canny = False
cn = ""

def Atransform(img):
    rows, cols, ch = img.shape

    pts1 = np.float32([[0, rows * 0.05], [cols * 1.05, rows * 0.05], [cols * 0, rows * 0.98], [cols * 0.95, rows*0.9]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (cols, rows))
    first_third = int(cols / 3)
    second_third = cols - first_third
    if canny:
        cn = "canny"
        if img[0][0][0] == img[0][0][1] and img[0][0][0] == img[0][0][2]:
            alpha = 1
            beta = 1.2
            dst = cv.convertScaleAbs(dst, alpha, beta)
            dst = cv.Canny(dst, 30, 40)
        else:
            dst = cv.Canny(dst, 80, 110)
    cropped_image1 = dst[0:rows, 0:first_third]
    cropped_image2 = dst[0:rows, first_third:second_third]
    cropped_image3 = dst[0:rows, second_third:cols]
    return cropped_image1, cropped_image2, cropped_image3

def AtransformSide(img):
    dst = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    rows, cols, ch = dst.shape
    # im = dst[round(rows*0.22):round(rows*0.95), round(cols*0.16):round(cols*0.8)]
    dst = dst[round(rows*0.25):round(rows*1), round(cols*0.08):round(cols*0.75)]
    rows, cols, ch = dst.shape
    first_third = int(cols / 3)
    second_third = cols - first_third
    cropped_image1 = dst[0:rows, 0:first_third]
    cropped_image2 = dst[0:rows, first_third:second_third]
    cropped_image3 = dst[0:rows, second_third:cols]
    # pts1 = np.float32([[cols * 0.0, rows * -0.1], [cols * 0.8, rows * 0.2], [cols * 0.0, rows * 0.7], [cols * 1, rows * 1.8]])
    # # pts1 = np.float32([[cols * 0.15, rows * 0.2], [cols * 0.88, rows * 0.42], [cols * 0.15, rows * 0.76], [cols * 0.78, rows * 1.35]])
    # pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    # M = cv.getPerspectiveTransform(pts1, pts2)
    # dst = cv.warpPerspective(dst, M, (cols, rows))
    # dst = cv.resize(dst, (0, 0), fx=1, fy=0.5)
    # dst = dst[int(rows*0):int(rows*0.5), int(cols*0):int(cols)]
    # dst = cv.resize(dst, (0, 0), fx=3, fy=1)
    return cropped_image1, cropped_image2, cropped_image3

if __name__ == "__main__":
    for PATH in IMAGE_PATHS:
        img = cv.imread(path+PATH)
        assert img is not None, "file could not be read, check with os.path.exists()"

        # cropped_image1, cropped_image2, cropped_image3 = Atransform(img)
        #
        # #cv.imwrite(savepath + PATH, dst)
        # cv.imwrite(savepath + cn + "cropped1" + PATH, cropped_image1)
        # cv.imwrite(savepath + cn + "cropped2" + PATH, cropped_image2)
        # cv.imwrite(savepath + cn + "cropped3" + PATH, cropped_image3)
        affine_image = AtransformSide(img)
        cv.imwrite(savepath + cn + "rotated" + PATH, affine_image)


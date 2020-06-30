import numpy as np
from scipy.cluster.vq import *
import cv2


class Vocabulary(object):
    def __init__(self, name, feature):
        self.feature = feature
        self.name = name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbr_words = 0

    def train(self, featurefiles, k, subsampling=10):
        """使用k个单词数为k的均值，从特征文件中列出的文件中的特征训练词汇。"""
        """训练数据的下采样（subsampling）可用于加速"""
        """k为聚类中心数（词汇单词数）"""

        nbr_images = len(featurefiles)
        des_list = []  # 特征描述
        descriptors = np.zeros((1, 32))
        for path in featurefiles:
            des = extract_feature(path, feature=self.feature)
            if des is not None:
                descriptors = np.row_stack((descriptors, des))
            des_list.append(des)
        descriptors = descriptors[1:, :]  # the des matrix of orb

        # K-means: 最后一个参数决定kmeans运行次数
        self.voc, distortion = kmeans(descriptors[::subsampling, :], k, 4)
        self.nbr_words = self.voc.shape[0]

        # 遍历所有的训练图像，并投影到词汇上
        imwords = np.zeros((nbr_images, self.nbr_words))
        for i in range(nbr_images):
            imwords[i] = self.project(des_list[i])

        nbr_occurences = np.sum((imwords > 0) * 1, axis=0)
        self.idf = np.log((1.0 * nbr_images) / (1.0 * nbr_occurences + 1))
        self.trainingdata = featurefiles

    def project(self, descriptors):
        """ 将描述子投影到词汇上，以创建单词直方图  """
        # 图像单词直方图
        imhist = np.zeros((self.nbr_words))
        words, distance = vq(descriptors, self.voc)
        for w in words:
            imhist[w] += 1
        return imhist

    def get_words(self, descriptors):
        """ Convert descriptors to words. """
        return vq(descriptors, self.voc)[0]

def extract_feature(path, feature='orb'):
    if feature is 'orb':
        try:
            orb = cv2.ORB_create()
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # find the keypoints with ORB
            kp = orb.detect(gray, None)
            # compute the descriptors with ORB
            kp, des = orb.compute(gray, kp)
            return des
        except:
            print('o')
    else:
        # add other features
        pass


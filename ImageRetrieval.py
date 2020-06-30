import os
import pickle
import numpy as np
from sqlite3 import dbapi2 as sqlite
import cv2
from BOW.imagesearch.vocabulary import Vocabulary, extract_feature
from BOW.imagesearch import imagesearch


def get_img_paths(training_path):
    #  根据图像数据文件夹路径获取所有图片路径
    training_names = os.listdir(training_path)
    img_paths = []  # 所有图片路径
    for name in training_names:
        img_path = os.path.join(training_path, name)
        img_paths.append(img_path)
    return img_paths

class ImageRetrieval():
    def __init__(self):
        self.feature = 'orb'
        self.base_dir = 'D:\\MyFiles\\SceneTransformation\\Relocalization_all\\Town02\\W000_P100_V000_P000'
        self.training_path = os.path.join(self.base_dir, 'RGB')  # 训练样本文件夹路径

        self.vocabulary_path = os.path.join(self.base_dir, 'vocabulary.pkl')
        self.vocabulary_name = 'swallow'
        self.database_name = 'ImaAdd.db'
        self.img_paths = get_img_paths(self.training_path)

        self.all_img_paths = get_img_paths(self.training_path) + \
                             get_img_paths(os.path.join('D:\\MyFiles\\SceneTransformation\\Relocalization_all\\Town02\\W000_P100_V050_P200', 'RGB')) + \
                             get_img_paths(os.path.join('D:\\MyFiles\\SceneTransformation\\Relocalization_all\\Town02\\W000_P100_V075_P300', 'RGB'))

        self.gen_vocabulary(word_num=1000)

    def gen_vocabulary(self, word_num=100, subsampling=10):
        # subsampling: 训练数据的下采样（subsampling）可用于加速
        voc = Vocabulary(self.vocabulary_name, self.feature)
        voc.train(self.img_paths, word_num, subsampling)
        # 保存词汇
        with open(self.vocabulary_path, 'wb') as f:
            pickle.dump(voc, f)
        print('vocabulary generated:', voc.name, 'words_num', voc.nbr_words, 'feature:', voc.feature)

    def commit_database(self):
        # 载入词汇
        with open(self.vocabulary_path, 'rb') as f:
            voc = pickle.load(f)

        # 创建索引
        indx = imagesearch.Indexer(self.database_name, voc)
        indx.create_tables()
        # 遍历所有的图像，并将它们的特征投影到词汇上
        for path in self.all_img_paths:
            des = extract_feature(path, self.feature)
            indx.add_to_index(path, des)
        # 提交到数据库
        indx.db_commit()

    def image_query(self, query_image_path, nbr_results=5, show_plot=False, src_return=False):
        # nbr_results 结果图像数
        # 载入词汇
        with open(self.vocabulary_path, 'rb') as f:
            voc = pickle.load(f)
        src = imagesearch.Searcher(self.database_name, voc)
        # 遍历所有的图像，并将它们的特征投影到词汇上
        # iw = voc.project(descr)
        # print('当前图像单词词频（直方图）：'), print(iw)
        # print('候选图像列表：'), print(src.candidates_from_histogram(iw)[:10])  # 获取具有相似单词的图像列表
        # print('匹配结果：')
        res_info = src.query(query_image_path)[:nbr_results]  # ((distance, id),())
        res_id = [w[1] for w in res_info]
        if show_plot:
            imagesearch.plot_results(src, res_id)

        res = []
        for item in res_info:
            index = item[1]-1
            path = self.all_img_paths[index]
            score = item[0]
            tmp = [index, path, score]
            res.append(tmp)
        if src_return:
            return res, src
        else:
            return res


def get_frame_info():
    # 统一不同动态设置下的图像及其位置坐标
    dir1 = './Town02/W000_P100_V000_P000'
    dir2 = './Town02/W000_P100_V050_P200'
    dir3 = './Town02/W000_P100_V075_P300'
    static_list = os.listdir(os.path.join(dir1, 'RGB'))
    dynamic1_list = os.listdir(os.path.join(dir2, 'RGB'))
    dynamic2_list = os.listdir(os.path.join(dir3, 'RGB'))  # frame 最多

    for frame in dynamic2_list:
        if frame not in dynamic1_list or frame not in static_list:
            for dir_ in [dir1, dir2, dir3]:
                for parse in ['Depth', 'RGB', 'SemanticSegmentation']:
                    try:
                        os.remove(os.path.join(dir_, parse, frame))
                    except:
                        pass

    frame_img = os.listdir(os.path.join(dir1, 'RGB'))
    trajectory = np.loadtxt(os.path.join(dir3, 'Trajectory.txt'))
    frame_pos = []
    for path in frame_img:
        frame_num = int(path[-10:-4])
        frame_pos.append(trajectory[frame_num,[1,2]])
    return frame_img, frame_pos


def get_topN_from_training(res, training_parse, topN=5):
    # res[0]: [index, path, score]
    save_res = []
    for i in range(len(res)):
        index = res[i][0]
        path = res[i][1]
        score = res[i][2]
        if training_parse in path:
            save_res.append(res[i])
    res_filter = save_res[:topN]
    return res_filter



if __name__ == '__main__':

    frame_img, frame_pos = get_frame_info()

    query_dir = 'D:\\MyFiles\\SceneTransformation\\Relocalization_all\\Town02\\W000_P100_V000_P000'
    img_names = os.listdir(os.path.join(query_dir, 'RGB'))  # 训练样本文件夹路径

    ImgRetrieval = ImageRetrieval()
    # ImgRetrieval.gen_vocabulary(word_num=1000)
    ImgRetrieval.commit_database()

error_rate_save = []
for i in range(len(img_names)):
    query_image_path = os.path.join(query_dir, 'RGB', img_names[i])
    ind = frame_img.index(query_image_path[-10:])
    query_image_pos = frame_pos[ind]
    res, src = ImgRetrieval.image_query(query_image_path, nbr_results=3*len(img_names), src_return=True)
    res_filter = get_topN_from_training(res, training_parse='W000_P100_V000_P000', topN=5)
    # imagesearch.plot_results(src, [w[0]+1 for w in res_filter])  # 此处，数据库序号从1开始，故1

    if len(res_filter):
        pos = np.array([frame_pos[w[0]] for w in res_filter])
        error_rate = np.abs((pos[0][0]-query_image_pos[0])/(query_image_pos[0]+1e-3)) + \
                     np.abs((pos[0][1]-query_image_pos[1])/(query_image_pos[1]+1e-3))
        error_rate_save.append(error_rate)
    else:
        print('no matched')
        error_rate = 1
        error_rate_save.append(error_rate)
    # print(query_image_path, query_image_pos)
    # print(res_filter)
    # print(pos)
print('mean_error', np.mean(error_rate_save))


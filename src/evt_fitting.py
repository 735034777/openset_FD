# import os, sys, pickle, glob
# import os.path as path
# import argparse
# import scipy.spatial.distance as spd
# import scipy as sp
# from scipy.io import loadmat
import sys
import os
from src.config import BASE_FILE_PATH
import src.config as config

import torch

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_directory, 'utils'))

import pickle
import numpy as np
import pandas as pd
from metric_learn import LMNN
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import paired_distances
from src.utils.train_base_model import load_data,load_model
from multiprocessing import Pool, cpu_count
from functools import partial
import libmr
# from openmax_utils import compute_distance


try:
    from utils import libmr      # noqa: F401
except ImportError:
    pass
    # print("LibMR not installed or libmr.so not found")
    # print("Install libmr: cd libMR/; ./compile.sh")
    # sys.exit()

# ---------------------------------------------------------------------------------
NCHANNELS = 1
ACTIVATE_VECTOR_DIM=30
# ---------------------------------------------------------------------------------

def calculate_metrix(y_true,y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    # Precision and Recall
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')

    # Confusion Matrix for Specificity Calculation
    cm = confusion_matrix(y_true, y_pred)
    tps = cm.diagonal()
    fps = cm.sum(axis=0) - tps
    fns = cm.sum(axis=1) - tps
    tns = cm.sum() - (tps + fps + fns)

    # 为每个类别计算指标
    recalls = tps / (tps + fns)
    specificities = tns / (tns + fps)

    # 计算所有类别的Youden's Index的平均值
    youdens_index = np.mean(recalls + specificities - 1)

    # For Normalized Accuracy, you'll need to implement a custom function
    # using the formulas provided earlier.
    return precision,recall,f1,youdens_index


def weibull_fit(metric_type="cosine",tail_size=5):
    # 定义了一个路径 model_path，表示保存或加载韦布尔分布模型的文件路径，这里假设为 'data/mnist_fashion_weibull_model.pkl'。
    model_path = BASE_FILE_PATH+'/src/data/mnist_fashion_weibull_model.pkl'
    # 检查是否已经存在保存的韦布尔分布模型文件，如果存在，则直接加载并返回该模型。
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    # 如果不存在保存的模型文件，
    x, _,y, _, dim = load_data("test")
    np.save("train_label.npy",y)

    # 调用 get_score_and_prob 函数获取数据的分数（score）和概率（prob），其中概率是通过模型预测得到的。
    prob, scores,highDV = get_score_and_prob(x,dim)
    np.save("SNEtrain_score.npy",scores)
    # 通过 np.argmax(prob, axis=1) 得到每个样本的预测类别
    predicted_y = np.argmax(prob, axis=1)
    # 获取数据集中的唯一标签（labels）
    labels = np.unique(y)
    # 初始化一个字典 av_map，用于存储每个类别的分数。
    av_map = {}
    highDV_map={}
    # 遍历每个类别，将对应类别的真实标签为该类别且模型预测的标签也为该类别的样本的分数存储在 av_map 中
    for label in labels:
        if config.HIGH_DIMENSION_OUTPUT==True:
            highDV_map[label] = highDV[(y.detach().numpy() == label) & (predicted_y == y.detach().numpy()),:]
        else:
            av_map[label] = scores[(y.detach().numpy() == label) & (predicted_y == y.detach().numpy()),:]



    if metric_type=="lmnn":
        # 初始化lmnn度量学习
        model_path = BASE_FILE_PATH + '/src/data/CWRU_lmnn_model.pkl'
        # 检查是否已经存在保存的韦布尔分布模型文件，如果存在，则直接加载并返回该模型。
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                lmnn = pickle.load(f)
        else:
        # global LMNN_LR
        # print("lmnn_lr is {}".format(config.LMNN_LR))
            lmnn = LMNN(k=dim,learn_rate=config.LMNN_LR)
            if config.HIGH_DIMENSION_OUTPUT==True:#验证是否使用高阶激活向量
                lmnn.fit(highDV, y)
            else:
                lmnn.fit(scores,y)
            # 将拟合得到的韦布尔分布模型保存到文件中，以便下次直接加载使用
            with open(model_path, 'wb') as f:
                pickle.dump(lmnn, f)
    else:
        lmnn=None

    # 调用 weibull_fit_tails 函数拟合韦布尔分布的尾部，该函数返回一个韦布尔分布模型。
    model = weibull_fit_tails(av_map, highDV_map,dim,lmnn,tail_size=tail_size,metric_type=metric_type)

    # 返回拟合得到的韦布尔分布模型
    return model,lmnn


def weibull_fit_tails(av_map,highDV,dim,lmnn=None, tail_size=200, metric_type='cosine'):


    # 初始化一个空字典 weibull_model，用于存储每个类别的韦布尔分布模型
    weibull_model = {}
    # 获取 av_map 中所有类别的标签。
    if config.HIGH_DIMENSION_OUTPUT==True:
        labels = highDV.keys()
    else:
        labels = av_map.keys()

    for label in labels:
        # print(f'EVT fitting for label {label}')
        # 初始化一个子字典，并将其赋值给 weibull_model[label]。
        weibull_model[label] = {}
        # 从 av_map 中取出当前类别的分数信息 class_av
        if config.HIGH_DIMENSION_OUTPUT==True:
            class_av = highDV[label]
        else:
            class_av = av_map[label]
        # 计算当前类别的平均分数向量 class_mav
        if metric_type=="lmnn":
            class_av = lmnn.transform(class_av)
        else:
            pass
        class_mav = np.mean(class_av, axis=0, keepdims=True)
        # 初始化一个用于存储样本到平均向量的距离的数组 av_distance。
        av_distance = np.zeros((1, class_av.shape[0]))
        # 遍历当前类别的每个样本，计算其到平均向量的距离，并将距离存储在 av_distance 中
        for i in range(class_av.shape[0]):
            if metric_type=="lmnn":
                L = lmnn.components_
                M = np.dot(L.T, L)
                inv_M = np.linalg.inv(M)
                distances = mahalanobis(class_av[i], class_mav[0], inv_M)
                av_distance[0,i] = np.mean(distances)
            else:
                av_distance[0, i] = compute_distance(class_av[i, :].reshape(1, -1), class_mav, metric_type=metric_type,dim=class_av.shape[1])
        # 将平均向量和距离信息保存到 weibull_model[label] 中
        weibull_model[label]['mean_vec'] = class_mav
        weibull_model[label]['distances'] = av_distance
        # 使用 libmr 库中的 MR 类来拟合韦布尔分布的尾部
        mr = libmr.MR()
        # 限制 tail_size 不超过样本数，并将样本的距离排序后取最大的 tail_size 个作为尾部拟合的样本
        tail_size_fix = min(tail_size, av_distance.shape[1])
        tails_to_fit = sorted(av_distance[0, :])[-tail_size_fix:]
        mr.fit_high(tails_to_fit, tail_size_fix)
        # 将拟合得到的韦布尔分布模型存储在 weibull_model[label]['weibull_model'] 中。
        weibull_model[label]['weibull_model'] = mr
    return weibull_model
# ---------------------------------------------------------------------------------
def compute_distance(a, b, metric_type,dim):
    """
    计算两个向量或矩阵之间的距离。具体来说，这个函数使用了 sklearn.metrics.pairwise.paired_distances 函数，该函数可以计算两个集合中对应样本之间的距离
    :param a: 输入的向量或矩阵
    :param b: 输入的向量或矩阵
    :param metric_type:
    :return: 是距离度量的类型，表示计算距离的方法。
    """
    # dim = ACTIVATE_VECTOR_DIM
    # b_r = np.tile(b, (a.T.shape[0], 1))
    a = a.reshape((1,dim))
    return paired_distances(a, b, metric=metric_type, n_jobs=1)

def recalibrate_scores(activation_vector,highDV, weibull_model, labels, alpha_rank=10,metric_type="cosine",lmnn=None):
    #使用argsort对activation_vector进行排序，得到排名最高的索引列表，
    # 并通过ravel()和[::-1]进行反转，得到一个排名从高到低的列表 ranked_list。
    ranked_list = activation_vector.argsort().ravel()[::-1]
    #计算根据排名计算的权重 alpha_weights，该权重用于对激活分数进行重新校准
    alpha_weights = [((alpha_rank + 1) - i) / float(alpha_rank) for i in range(1, alpha_rank + 1)]
    # 初始化一个长度为类别数的零向量ranked_alpha，用于存储每个类别的重新校准权重。
    ranked_alpha = np.zeros((len(labels)))
    #遍历前 alpha_rank 个排名，将对应的权重赋值给 ranked_alpha。
    for i in range(alpha_rank):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate score for each class to include probability of unknown
    #初始化两个空列表 openmax_score 和 openmax_score_unknown，用于存储重新校准后的激活分数和未知类别的分数。
    openmax_score = []
    openmax_score_unknown = []
    #遍历每个类别，计算当前类别到韦布尔模型的距离，并使用韦布尔模型计算 w_score，然后用这个得分来重新校准激活分数
    for label_index, label in enumerate(labels):
        # get distance between current channel and mean vector
        weibull = query_weibull(label, weibull_model)
        if metric_type=="lmnn":
            L = lmnn.components_
            M = np.dot(L.T, L)
            inv_M = np.linalg.inv(M)
            if config.HIGH_DIMENSION_OUTPUT==True:
                av_distance = mahalanobis(highDV, weibull[0][0][0], inv_M)
            else:
                av_distance = mahalanobis(activation_vector, weibull[0][0][0], inv_M)
        else:
            if config.HIGH_DIMENSION_OUTPUT ==False:
                av_distance = compute_distance(activation_vector, weibull[0][0],metric_type=metric_type,dim=alpha_rank)
            else:
                av_distance = compute_distance(highDV, weibull[0][0],metric_type=metric_type,dim=highDV.shape[0])

        # obtain w_score for the distance and compute probability of the distance being unknown wrt to mean training vector
        wscore = weibull[2][0].w_score(av_distance)
        modified_score = activation_vector[label_index] * (1 - wscore * ranked_alpha[label_index])
        openmax_score += [modified_score]
        openmax_score_unknown += [activation_vector[label_index] - modified_score]
    #计算 openmax_probab 和 softmax_probab，分别表示经过OpenMax校准后的概率和原始Softmax概率。
    openmax_score = np.array(openmax_score)
    openmax_score_unknown = np.array(openmax_score_unknown)

    # Pass the re-calibrated scores for the image into OpenMax
    openmax_probab = compute_openmax_probability(openmax_score, openmax_score_unknown, labels)
    softmax_probab = compute_softmax_probability(activation_vector)  # Calculate SoftMax ???
    #返回经过OpenMax校准后的概率和原始Softmax概率
    return np.array(openmax_probab), np.array(softmax_probab)

def query_weibull(label, weibull_model):
    """
    从给定的韦布尔模型中查询特定标签（类别）对应的信息。
    :param label: 是要查询的类别标签
    :param weibull_model: 包含了各个类别的韦布尔分布模型的字典
    :return: 函数返回一个列表，包含三个子列表
    weibull_model[label]['mean_vec']：对应类别 label 的平均向量。
weibull_model[label]['distances']：对应类别 label 的样本到平均向量的距离。
weibull_model[label]['weibull_model']：对应类别 label 的韦布尔分布模型。
    """
    return [
        [weibull_model[label]['mean_vec']],
        [weibull_model[label]['distances']],
        [weibull_model[label]['weibull_model']]
    ]


def compute_openmax_probability(openmax_score, openmax_score_unknown, labels):
    """
    根据OpenMax方法计算归一化后的概率分数
    :param openmax_score: 通过重新校准的激活分数，对应于每个类别
    :param openmax_score_unknown: 与未知类别相关的分数
    :param labels: 类别标签的列表
    :return:
    """
    #计算指数化的分数： 使用循环遍历每个类别，计算每个类别对应的指数化激活分数，并将结果存储在 exp_scores 列表中
    exp_scores = []
    for label_index, label in enumerate(labels):
        exp_scores += [np.exp(openmax_score[label_index])]
    # 计算总分母： 计算所有类别的指数化激活分数之和，以及未知类别的指数化分数之和。
    total_denominator = np.sum(np.exp(openmax_score)) + np.exp(np.sum(openmax_score_unknown))
    # 计算归一化概率分数： 对每个类别的指数化激活分数进行归一化，得到概率分数
    prob_scores = np.array(exp_scores) / total_denominator
    # 计算未知类别概率： 计算未知类别的概率
    prob_unknown = np.exp(np.sum(openmax_score_unknown)) / total_denominator

    return prob_scores.tolist() + [prob_unknown]

def compute_softmax_probability(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

def get_score_and_prob(x,dim):
    model = load_model(dim,"test")
    predictive,activate_val,high_dim_vector = model(x)
    return predictive.detach().numpy(),activate_val.detach().numpy(),high_dim_vector.detach().numpy()

def recalibrate_score(score, highDV,weibull_model,dim,metric_type ="cosine", lmnn=None):
    openmax, softmax_probab = recalibrate_scores(score, highDV,weibull_model, weibull_model.keys(),metric_type=metric_type, alpha_rank=dim,lmnn=lmnn)
    return np.argmax(openmax),np.argmax(softmax_probab)

def calculate_openmax_accuracy(metric_type="cosine",tail_size = 5):
    # 拟合韦布尔模型： 调用 weibull_fit() 函数来获取韦布尔分布模型，该模型将在后续的步骤中用于激活分数的重新校准。
    weibull_model,lmnn = weibull_fit(metric_type,tail_size)
    # 加载测试数据：CWRU。
    _,x_test,_, y_test ,dim= load_data(phase="test")
    # 获取激活分数： 调用 get_score_and_prob 函数获取测试数据的激活分数 scores
    _, scores,highDV = get_score_and_prob(x_test,dim)
    softmax_scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    np.save("activation_vectory_"+metric_type,scores)

    if metric_type == "lmnn":
        if config.HIGH_DIMENSION_OUTPUT==False:
            scores = lmnn.transform(scores)
        else:
            highDV = lmnn.transform(highDV)
    # 激活分数的重新校准： 使用多进程（Pool）并行地对每个样本的激活分数进行重新校准，
    # 其中使用了 recalibrate_score 函数，该函数依赖于之前拟合的韦布尔模型。校准后的结果存储在 y_predicted 中
    # print('Recalibrating scores...')
    score_list = [scores[i, :] for i in range(scores.shape[0])]
    if config.HIGH_DIMENSION_OUTPUT==True:
        highDV_list = [highDV[i, :] for i in range(highDV.shape[0])]
    else:highDV_list=[0 for i in range(len(score_list))]
    pool = Pool(cpu_count())
    y_predicted_values = []
    y_softmax_predicted_values= []

    # for score in score_list:
    for i in range(len(score_list)):

    # 这里的 partial(recalibrate_score, weibull_model=weibull_model) 是一个部分应用，将 weibull_model 固定在 recalibrate_score 函数中，以便并行处理
        y_predicted,y_softmax_predicted = recalibrate_score(np.array(score_list[i]),np.array(highDV_list[i]),weibull_model=weibull_model,metric_type = metric_type,dim=dim,lmnn=lmnn)
    # y_predicted = np.array(pool.map(partial(recalibrate_score, weibull_model=weibull_model), np.array(score_list)))
        y_predicted_values.append(y_predicted)
        y_softmax_predicted_values.append(y_softmax_predicted)
    results_compare = pd.DataFrame(dict(zip(["y_test","y_softmax","y_openmax"],[list(np.array(y_test)),y_softmax_predicted_values,y_predicted_values])))
    y_test = y_test.detach().numpy()
    y_test[y_test >= dim] = dim
    np.save("y_predicted_values_"+metric_type,np.array(y_predicted_values))
    np.save("y_softmax_predicted_values_"+metric_type,np.argmax(softmax_scores,axis=1))
    np.save("y_test_"+metric_type,y_test)


    accuracy = np.mean(y_test == y_predicted_values)
    soft_accuracy = np.mean(y_test == y_softmax_predicted_values)
    soft_metrix = calculate_metrix(y_test,y_softmax_predicted_values)
    precision,recall,f1,youdens_index = calculate_metrix(y_test,y_predicted_values)
    # print('Accuracy =', accuracy)
    del x_test,y_test
    return accuracy,precision,recall,f1,youdens_index,soft_accuracy,soft_metrix

def calculate_openness(n_tr, n_te):
    import math
    """
    Calculate the openness for Open Set Recognition.
    n_tr: Number of known classes in the training set.
    n_te: Total number of classes in the test set (known and unknown).
    """
    return 1 - math.sqrt((2 * n_tr) / (n_tr + n_te))

if __name__=="__main__":

    # accuracy = calculate_openmax_accuracy(10)
    # print("结果为{}".format(accuracy))
    best_i = 0
    best_r = 0
    config.HIGH_DIMENSION_OUTPUT=False
    for i in range(1,2):
        accuracy = calculate_openmax_accuracy("cosine",20)
        # if accuracy>best_r:
        #     best_r = accuracy
        #     best_i = i
    print("最优结果为{}，步数为{}".format(best_r,best_i*5))
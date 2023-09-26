import copy

import numpy as np
from numpy import random
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


class CS_SMOTE:
    def __init__(self, b=1, random_seed=42, value_a=1, dim=3):
        self.b = b # 影响边界样本生成
        self.random_seed = random_seed
        self.value_a = value_a
        self.dim = dim

    # 输出平衡数据集
    def fit_resample(self, X, y, state=False):
        np.random.seed(self.random_seed)
        # 先标准化数据再pca降维
        pca = PCA(n_components=self.dim)
        z_scaler = preprocessing.StandardScaler()
        X_1 = z_scaler.fit_transform(X)
        X_2 = pca.fit_transform(X_1)

        # 密铺区域
        field_list, a = self.field_generation(X_2, y) # 得到区域中心点列表和区域边长a

        # 划分多数类少数类样本
        min_index, max_index, min_x, max_x = self.class_division(X, y)

        # 确定每个区域有哪些样本的和哪些少数类样本
        field_point_dict, field_min_point_dict = self.sample_in_field(X_2, y, field_list, a)

        # 划分各个域类别
        field_class_dict = self.field_division(field_point_dict, y)

        # 判断邻域并同化过滤+
        neighbor_dict_face, neighbor_dict_edge, neighbor_dict_point, field_class_dict = self.neighbor_field(field_class_dict, field_list, a)

        # 打分，获得概率
        field_select_probability = self.score_and_probability(field_class_dict, neighbor_dict_face, neighbor_dict_edge, neighbor_dict_point)

        # 根据概率随机取点合成
        X_new, y = self.synthesis(field_select_probability, X, y, max_x, min_x, field_min_point_dict, field_class_dict,
                  neighbor_dict_face, neighbor_dict_edge, neighbor_dict_point, td_state=state)

        if X_new == 0:
            return 0, 0

        return X_new, y


    # 密铺区域生成
    def field_generation(self, X, y):
        # 分开少数类和多数类
        _, _, min_x, _ = self.class_division(X, y)

        # 根据点分布区域确定密铺区域大小
        len_min = [] # 离原点最近坐标
        len_max = [] # 离原点最远坐标
        len_gap = [] # 最近最远坐标之间差距
        center_point = [] # 中心点位置坐标
        for i in range(len(X[0])):
            len_min.append(min([j[i] for j in X]))
            len_max.append(max([j[i] for j in X]))
        for i in range(len(len_min)):
            len_gap.append(len_max[i] - len_min[i])
        for i in range(len(len_min)):
            center_point.append((len_max[i] + len_min[i]) / 2)


        # 确定正方体边长
        # TODO：边长自己可以换
        p1 = pdist(X, 'euclidean')
        p2 = pdist(min_x, 'euclidean')
        A = squareform(p1, force='no', checks=True) # 所有样本的距离矩阵
        B = squareform(p2, force='no', checks=True) # 少数类样本距离矩阵

        a1 = sum([sum(i) for i in B]) / (len(min_x) * (len(min_x) - 1))
        a2 = sum([sum(i) for i in A]) / (len(y) * (len(y) - 1))

        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] == 0:
                    A[i][j] = float('inf')
        for i in range(len(B)):
            for j in range(len(B)):
                if B[i][j] == 0:
                    B[i][j] = float('inf')

        a3 = sum([min(i) for i in B]) / len(min_x)
        a4 = sum([min(i) for i in A]) / len(y)
        a = ((a3/a4 - a1/a2) + self.value_a * (a1/a2))

        # 根据点的分布确定整个密铺区域各维度的图形个数
        field_num = []
        for i in len_gap:
            if (i // a) % 2 == 0:
                field_num.append((i // a) + 3)
            elif (i // a) % 2 == 1:
                field_num.append((i // a) + 2)

        # 根据密铺区域的边界。各维度图形个数、图形边长确定个个区域的中心点
        field_list = [center_point]
        # 先求x一条线上，再扩展到xy平面，再扩展到xyz空间。
        for i in range(len(field_num)):
            field_list_old = copy.deepcopy(field_list)
            for num in range(int((field_num[i]-1)/2)):
                for j in field_list_old:
                    new_point1 = j[0:i] + [j[i] - (num + 1) * a] + j[i+1:]
                    new_point2 = j[0:i] + [j[i] + (num + 1) * a] + j[i+1:]
                    field_list.append(new_point1)
                    field_list.append(new_point2)

        return field_list, a

    # 区分少数类多数类样本
    def class_division(self, X, y):
        min_index = []
        max_index = []
        min_x = []
        max_x = []
        for i in range(len(y)):
            if int(y[i]) == 1:
                min_index += [i]
                min_x += [X[i]]
            else:
                max_index += [i]
                max_x += [X[i]]
        return min_index, max_index, min_x, max_x

    # 确定每个区域内都有哪些样本点
    # TODO:不同图形和维度计算是否在内的方法有所不同，self.in_field需要被修改
    def sample_in_field(self, X, y, field_list, a):
        field_point_dict = {}
        field_min_point_dict = {}
        for i in range(len(X)):
            for j in range(len(field_list)):
                if self.in_field(X[i], field_list[j], a) == 1:
                    if int(y[i]) == 1:
                        if j not in field_min_point_dict.keys():
                            field_min_point_dict[j] = [i]
                        else:
                            field_min_point_dict[j] += [i]

                    if j not in field_point_dict.keys():
                        field_point_dict[j] = [i]
                    else:
                        field_point_dict[j] += [i]
                    break
        return field_point_dict, field_min_point_dict

    # 判断点是否在该几何区域内,
    # TODO:使用其他图形时这个需要修改
    def in_field(self, sample, field_center_point, length):
        for i in range(len(sample)):
            if abs(sample[i] - field_center_point[i]) >= 1.0001 * length/2:
                return 0
        return 1


    # 划分各个域类别
    def field_division(self, field_point_dict, y):
        field_class_dict = {'dis': [], 'min': []}
        for field, point in field_point_dict.items():
            for i in range(len(point)-1):
                if y[point[i]] != y[point[i+1]]:
                    if 'dis' not in field_class_dict.keys():
                        field_class_dict['dis'] = [field]
                    else:
                        field_class_dict['dis'] += [field]
                    break

            if 'dis' in field_class_dict.keys() and field in field_class_dict['dis']:
                continue

            elif int(y[point[0]]) == 1:
                if 'min' not in field_class_dict.keys():
                    field_class_dict['min'] = [field]
                else:
                    field_class_dict['min'] += [field]
            # 以下俩步判断没有必要
            # elif y[point[0]] == 0:
            #     if 'max' not in field_class_dict.keys():
            #         field_class_dict['max'] = [field]
            #     else:
            #         field_class_dict['max'] += [field]
            # elif point == []:
            #     if 'zero' not in field_class_dict.keys():
            #         field_class_dict['zero'] = [field]
            #     else:
            #         field_class_dict['zero'] += [field]
        return field_class_dict

    # 判断邻域并同化过滤
    # TODO:不同图形邻域定义不同
    def neighbor_field(self, field_class_dict, field_list, a):
        # 判断并存储邻域
        neighbor_dict_face = {}  # 面相邻
        neighbor_dict_edge = {}  # 边相邻
        neighbor_dict_point = {}  # 点相邻

        for key in field_class_dict.keys():
            for i in field_class_dict[key]:
                for j in range(len(field_list)):
                    if distance.euclidean(field_list[i], field_list[j]) < 1.01 * a and i != j:  # 计算俩域中心点的距离
                        if i not in neighbor_dict_face.keys():
                            neighbor_dict_face[i] = [j]
                        else:
                            neighbor_dict_face[i] += [j]
                    if 1.01 * a < distance.euclidean(field_list[i], field_list[j]) < 1.01 * a * (2 ** 0.5) and i != j:
                        if i not in neighbor_dict_edge.keys():
                            neighbor_dict_edge[i] = [j]
                        else:
                            neighbor_dict_edge[i] += [j]
                    if 1.01 * a * (2 ** 0.5) < distance.euclidean(field_list[i], field_list[j]) < 1.01 * a * (3 ** 0.5) and i != j:
                        if i not in neighbor_dict_point.keys():
                            neighbor_dict_point[i] = [j]
                        else:
                            neighbor_dict_point[i] += [j]

                    # 同化过滤。使用其他维度和图形时需要重新考虑。
                    # # TODO: 这里外层的if为了减少算力所以在达到条件时停止了，达到的条件不同时情况不一样，可以不使用。
                    if i in neighbor_dict_face.keys() and i in neighbor_dict_edge.keys() and i in neighbor_dict_point.keys():
                        if len(neighbor_dict_face[i]) == 6 and len(neighbor_dict_edge[i]) == 12 and len(neighbor_dict_point[i]) == 8:
                            num = 0
                            for m in neighbor_dict_face[i]:
                                if m in field_class_dict['min'] or m in field_class_dict['dis']:
                                    num = 1
                                    break
                            if num == 0:
                                for m in neighbor_dict_edge[i]:
                                    if m in field_class_dict['min'] or m in field_class_dict['dis']:
                                        num = 1
                            if num == 0:
                                for m in neighbor_dict_point[i]:
                                    if m in field_class_dict['min'] or m in field_class_dict['dis']:
                                        num = 1
                            if num == 0:
                                del neighbor_dict_face[i]
                                del neighbor_dict_edge[i]
                                del neighbor_dict_point[i]
                                field_class_dict[key].remove(i)
                            break

        return neighbor_dict_face, neighbor_dict_edge, neighbor_dict_point, field_class_dict



    # 分数和权重计算
    def score_and_probability(self, field_class_dict, neighbor_dict_face, neighbor_dict_edge, neighbor_dict_point, num=2.5):
        P_field = {}
        score_field = {}
        # 打分
        for key in field_class_dict.keys():
            for i in field_class_dict[key]:
                if key == 'min':
                    point = 2
                if key == 'dis':
                    point = 1
                if i in neighbor_dict_face.keys():
                    for j in neighbor_dict_face[i]:
                        if j in field_class_dict['min']:
                            point += 1
                        elif j in field_class_dict['dis']:
                            point += 0.5
                    for j in neighbor_dict_edge[i]:
                        if j in field_class_dict['min']:
                            point += 0.5
                        elif j in field_class_dict['dis']:
                            point += 0.25
                    for j in neighbor_dict_point[i]:
                        if j in field_class_dict['min']:
                            point += 0.2
                        elif j in field_class_dict['dis']:
                            point += 0.1

                    if point < num:
                        continue

                    score_field[i] = (point * (self.b ** point)) / (15 * (self.b ** 15))

        for field, score in score_field.items():
            P_field[field] = score / sum(score_field.values())

        return P_field

    # 采样合成样本
    def synthesis(self, field_select_probability, X, y, max_x, min_x, field_min_point_dict, field_class_dict,
                  neighbor_dict_face, neighbor_dict_edge, neighbor_dict_point, td_state=False):

        if len(field_select_probability.values()) < 2:
            return 0, 0

        if td_state:
            pca = PCA(n_components=self.dim)
            z_scaler = preprocessing.StandardScaler()
            X_1 = z_scaler.fit_transform(X)
            X = pca.fit_transform(X_1)


        X_new = list(copy.deepcopy(X))
        y = list(y)
        np.random.seed(self.random_seed)
        num = 0
        p_field = np.array(list(field_select_probability.values()), dtype='float64')

        while num <= len(max_x)-len(min_x):
            # 依概率选取第一个域
            field_index = np.random.choice(list(field_select_probability.keys()), p=p_field.ravel())
            # 在第一个域中随机选取一个少数类样本点
            sample_list = [np.random.choice(field_min_point_dict[field_index])]
            # 在第一个域的min和dis邻域中选取其他的点
            for i in neighbor_dict_face[field_index]:
                if i in field_class_dict['min'] or i in field_class_dict['dis']:
                    sample_list += [np.random.choice(field_min_point_dict[i])]
            for i in neighbor_dict_edge[field_index]:
                if i in field_class_dict['min'] or i in field_class_dict['dis']:
                    sample_list += [np.random.choice(field_min_point_dict[i])]
            for i in neighbor_dict_point[field_index]:
                if i in field_class_dict['min'] or i in field_class_dict['dis']:
                    sample_list += [np.random.choice(field_min_point_dict[i])]

            new_sample = copy.deepcopy(X[sample_list[0]])
            for i in sample_list:
                for j in range(len(new_sample)):
                    new_sample[j] = new_sample[j] + random.random(1) * (X[i][j] - new_sample[j])

            X_new += list([new_sample])
            y += list([1.0])
            num += 1

        return X_new, y

    # 用于计算已知中心点和边长时，该图形的各顶点位置
    def top_point_location(self, center_point, length):
        # 计算正方体的八个顶点，对xyz的八种情况：+++，++-，+-+，-++，+--，-+-，--+，---
        new_point1 = [center_point[0] + length / 2] + [center_point[1] + length / 2] + [
            center_point[2] + length / 2]
        new_point2 = [center_point[0] + length / 2] + [center_point[1] + length / 2] + [
            center_point[2] - length / 2]
        new_point3 = [center_point[0] + length / 2] + [center_point[1] - length / 2] + [
            center_point[2] + length / 2]
        new_point4 = [center_point[0] - length / 2] + [center_point[1] + length / 2] + [
            center_point[2] + length / 2]
        new_point5 = [center_point[0] + length / 2] + [center_point[1] - length / 2] + [
            center_point[2] - length / 2]
        new_point6 = [center_point[0] - length / 2] + [center_point[1] + length / 2] + [
            center_point[2] - length / 2]
        new_point7 = [center_point[0] - length / 2] + [center_point[1] - length / 2] + [
            center_point[2] + length / 2]
        new_point8 = [center_point[0] - length / 2] + [center_point[1] - length / 2] + [
            center_point[2] - length / 2]

        return [new_point1, new_point2, new_point3, new_point4, new_point5, new_point6, new_point7, new_point8]

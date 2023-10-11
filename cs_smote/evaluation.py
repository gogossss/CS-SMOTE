import sklearn
import smote_variants as sv
from gsmote import GeometricSMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import imbalanced_databases as imbd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from sklearn import decomposition
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from numpy import where
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as Accuracy
from sklearn.metrics import precision_score as Precision
from sklearn.metrics import recall_score as Recall
from sklearn.metrics import f1_score as F1_measure
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import imbalanced_databases as imbd
import warnings
from sklearn.svm import NuSVC
import smote_variants
import xlwt
from model import CS_SMOTE
from elm import ELM
from sklearn.metrics import roc_auc_score

def evaluation(model):
    if 'ttsmote' in str(model):
        p = str(model).split('.')[1]
        p = p.split(' object')[0]
    else:
        p = str(model).split('(')[0]
    book = xlwt.Workbook()  # 创建Excel
    sheet = book.add_sheet('sheet1')  # 创建sheet页
    title = ['model', 'classify', 'dataset', 'estimate', 'value']  # 把表头名称放入list里面
    row = 0
    # 循环把表头写入
    for t in title:
        sheet.write(0, row, t)
        row += 1
    row = 1  # 从表格的第二行开始写入数据

    list = [
        imbd.load_hepatitis(),
        imbd.load_spectf(),
        imbd.load_abalone_17_vs_7_8_9_10(),
        imbd.load_abalone_19_vs_10_11_12_13(),
        imbd.load_abalone_20_vs_8_9_10(),
        imbd.load_abalone_21_vs_8(),
        imbd.load_abalone_3_vs_11(),
        imbd.load_abalone9_18(),
        # imbd.load_car_good(),
        # imbd.load_car_vgood(),
        imbd.load_cleveland_0_vs_4(),
        imbd.load_dermatology_6(),
        # imbd.load_ecoli_0_1_3_7_vs_2_6(),
        imbd.load_ecoli_0_1_4_6_vs_5(),
        imbd.load_ecoli_0_1_4_7_vs_2_3_5_6(),
        imbd.load_ecoli_0_1_4_7_vs_5_6(),
        imbd.load_ecoli_0_1_vs_2_3_5(),
        imbd.load_ecoli_0_1_vs_5(),
        imbd.load_ecoli_0_2_3_4_vs_5(),
        imbd.load_ecoli_0_2_6_7_vs_3_5(),
        imbd.load_ecoli_0_3_4_6_vs_5(),
        imbd.load_ecoli_0_3_4_7_vs_5_6(),
        # imbd.load_ecoli_0_3_4_vs_5(),
        imbd.load_ecoli_0_4_6_vs_5(),
        imbd.load_ecoli_0_6_7_vs_3_5(),
        imbd.load_ecoli_0_6_7_vs_5(),
        imbd.load_ecoli4(),
        # imbd.load_flaref(),
        imbd.load_glass_0_1_4_6_vs_2(),
        imbd.load_glass_0_1_5_vs_2(),
        imbd.load_glass_0_1_6_vs_2(),
        imbd.load_glass_0_1_6_vs_5(),
        imbd.load_glass_0_4_vs_5(),
        imbd.load_glass_0_6_vs_5(),
        imbd.load_glass2(),
        imbd.load_glass4(),
        imbd.load_glass5(),
        # imbd.load_kddcup_buffer_overflow_vs_back(),
        # imbd.load_kddcup_guess_passwd_vs_satan(),
        # imbd.load_kddcup_land_vs_portsweep(),
        # imbd.load_kddcup_land_vs_satan(),
        # imbd.load_kddcup_rootkit_imap_vs_back(),
        # imbd.load_kr_vs_k_one_vs_fifteen(),
        # imbd.load_kr_vs_k_three_vs_eleven(),
        # imbd.load_kr_vs_k_zero_one_vs_draw(),
        # imbd.load_kr_vs_k_zero_vs_eight(),
        # imbd.load_kr_vs_k_zero_vs_fifteen(),
        # imbd.load_led7digit_0_2_4_5_6_7_8_9_vs_1(),
        imbd.load_lymphography_normal_fibrosis(),
        # imbd.load_poker_8_9_vs_5(),
        # imbd.load_poker_8_9_vs_6(),
        imbd.load_poker_8_vs_6(),
        imbd.load_poker_9_vs_7(),
        # imbd.load_shuttle_2_vs_5(),
        imbd.load_shuttle_6_vs_2_3(),
        imbd.load_shuttle_c0_vs_c4(),
        imbd.load_shuttle_c2_vs_c4(),
        imbd.load_vowel0(),
        # imbd.load_winequality_red_3_vs_5(),
        imbd.load_winequality_red_4(),
        imbd.load_winequality_red_8_vs_6(),
        imbd.load_winequality_red_8_vs_6_7(),
        # imbd.load_winequality_white_3_vs_7(),
        imbd.load_yeast_0_2_5_6_vs_3_7_8_9(),
        imbd.load_yeast_0_2_5_7_9_vs_3_6_8(),
        imbd.load_yeast_0_3_5_9_vs_7_8(),
        imbd.load_yeast_0_5_6_7_9_vs_4(),
        imbd.load_yeast_1_2_8_9_vs_7(),
        imbd.load_yeast_1_4_5_8_vs_7(),
        imbd.load_yeast_1_vs_7(),
        imbd.load_yeast_2_vs_4(),
        imbd.load_yeast_2_vs_8(),
        imbd.load_yeast4(),
        imbd.load_yeast5(),
        imbd.load_yeast6(),
        # imbd.load_ecoli_0_vs_1(),
        imbd.load_ecoli1(),
        imbd.load_ecoli2(),
        imbd.load_ecoli3(),
        imbd.load_glass_0_1_2_3_vs_4_5_6(),
        imbd.load_glass0(),
        imbd.load_glass1(),
        imbd.load_glass6(),
        # imbd.load_habarman(),
        # imbd.load_iris0(),
        imbd.load_new_thyroid1(),
        imbd.load_pima(),
        # imbd.load_segment0(),
        imbd.load_vehicle0(),
        imbd.load_vehicle1(),
        imbd.load_vehicle2(),
        imbd.load_vehicle3(),
        # imbd.load_wisconsin(),
        imbd.load_yeast1(),
        imbd.load_yeast3()]
    # list2 = [SVC(kernel='linear', probability=True, random_state=42),
    #          AdaBoostClassifier(algorithm="SAMME.R", n_estimators=100, random_state=42),
    #          GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
    #         MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=5)]
    # list2 = [AdaBoostClassifier(algorithm="SAMME.R", n_estimators=100, random_state=42),
    #          GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42)]
    list2 = [KNeighborsClassifier(n_neighbors=7)]
    list2 = [ELM]
    # NuSVC(probability=True, random_state=42),
    SMOTE = model
    list1 = []
    for j in list2:
        result_f = []
        result_g = []
        result_auc = []
        result = []
        for i in list:
            warnings.filterwarnings("ignore")
            print(i['name'])
            dataset = i
            X, y = dataset['data'], dataset['target']

            # 运用smote方法
            if model == 'non':
                smote_X, smote_Y = X, y
            else:
                smote_X, smote_Y = SMOTE.fit_resample(X, y)  # smote方法对训练数据进行扩增，通过训练出来的模型对测试集进行分类

            list1.append(i['name'])

            list1_ACC=[]   #存储准确率
            list2_Pre=[]   #存储精准率
            list3_Recall=[]   #存储召回率
            list4_f1=[]   #存储f1_measure
            list5_G_means=[]   #存储G—means
            list6_FPR=[]   #存储假正率
            list7_spe=[]   #存储特效性
            list8_AUC=[]   ##存储AUC面积

            if 'ttsmote' in str(model) and smote_X == 0:
                result.append(0)
                result.append(0)
                result.append(0)
                print(f'ERROR: No sample!')
                continue

            # 分类器
            classify = j

            # 进行交叉验证
            kf=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
            # kf=KFold(n_splits=5)
            smote_X=np.array(smote_X)
            smote_Y=np.array(smote_Y)

            for i ,(train ,test) in enumerate(kf.split(smote_X,smote_Y)):
                if j == ELM:
                    t = smote_Y[train].astype('int64')
                    elm = ELM(smote_X[train], np.eye(3)[t], 3000, 0.1)
                    predict = elm.TEST(smote_X[test])
                    smote_y_predict = np.argmax(predict, axis=1)  # OneHot编码形式 取每行最大值的索引即类别
                else:
                    classify=classify.fit(smote_X[train],smote_Y[train])
                    smote_y_predict = classify.predict(smote_X[test])    ## 运用smote算法得到测试集预测之后的分类
                    smote_score = classify.predict_proba(smote_X[test])
                    # print(smote_score)
                cm = CM(smote_Y[test], smote_y_predict, labels=[1, 0])  # 输出混淆矩阵
                smote_ACC = Accuracy(smote_Y[test], smote_y_predict)  # 输出准确率
                smote_precision = Precision(smote_Y[test], smote_y_predict)  # 精准率
                smote_recall = Recall(smote_Y[test], smote_y_predict)  # 召回率
                smote_F1measure = F1_measure(smote_Y[test], smote_y_predict)  # f1_measure 越接近于1越好
                smote_TP = cm[0, 0]  # 原本正类，预测后也是正类
                smote_TN = cm[1, 1]  # 原本是负类，预测后也是负类
                smote_FP = cm[1, 0]  # 原本是负类，预测后是正类
                smote_FN = cm[0, 1]  # 原本是正类，预测后成为负类
                smote_G_means = (smote_recall * ((smote_TN / (smote_TN + smote_FP)))) ** 0.5  # G_means计算方法
                smote_FPR = smote_FP / (np.sum(cm[1, :]))  # 假正率 负类中被错误分类的比例
                smote_Spe = smote_TN/ np.sum(cm[1, :])  # 特效性 负类中被正确分类的比例

                list1_ACC.append(smote_ACC)
                list2_Pre.append(smote_precision)
                list3_Recall.append(smote_recall)
                list4_f1.append(smote_F1measure)
                list5_G_means.append(smote_G_means)
                list6_FPR.append(smote_FPR)
                list7_spe.append(smote_Spe)
                #Roc曲线 与 AUC面积
                # smote_rocfpr,smote_rocrecallr,smote_threshold = roc_curve(smote_Y[test],smote_score[:,1],pos_label=1)
                # smote_AUC_area = AUC(smote_Y[test], smote_score[:,1])  #smote之后的AUC面积
                smote_AUC_area = roc_auc_score(smote_Y[test], smote_y_predict)
                list8_AUC.append(smote_AUC_area)

            print(np.mean(list4_f1))
            print(np.mean(list5_G_means))
            print(np.mean(list8_AUC))
            result.append(np.mean(list4_f1))
            result.append(np.mean(list5_G_means))
            result.append(np.mean(list8_AUC))
            result_f.append(np.mean(list4_f1))
            result_g.append(np.mean(list5_G_means))
            result_auc.append(np.mean(list8_AUC))

        m = str(j).split('(')[0]
        m = m.split('Cla')[0]

        # 一行一行的写，一行对应的所有列
        j = 0
        for i in range(len(result)):
            col = 0
            n = list1[j]
            if i % 3 == 0:
                sheet.write(row, col, f'{p}')  # rou代表列，col代表行，one写入
                col += 1
                sheet.write(row, col, f'{m}')
                col += 1
                sheet.write(row, col, f'{n}')
                col += 1
                sheet.write(row, col, f'F1')
                col += 1
                sheet.write(row, col, result[i])
                row += 1
            if i % 3 == 1:
                sheet.write(row, col, f'{p}')  # rou代表列，col代表行，one写入
                col += 1
                sheet.write(row, col, f'{m}')
                col += 1
                sheet.write(row, col, f'{n}')
                col += 1
                sheet.write(row, col, f'G')
                col += 1
                sheet.write(row, col, result[i])
                row += 1
            if i % 3 == 2:
                sheet.write(row, col, f'{p}')  # rou代表列，col代表行，one写入
                col += 1
                sheet.write(row, col, f'{m}')
                col += 1
                sheet.write(row, col, f'{n}')
                col += 1
                sheet.write(row, col, f'AUC')
                col += 1
                sheet.write(row, col, result[i])
                row += 1
                j += 1
    book.save(f'{p}.xls')

if __name__ == '__main__':
    evaluation(CS_SMOTE())

import numpy as np
from dtosmote import DTO
from gsmote import GeometricSMOTE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imbalanced_databases as imbd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

from smote_function.model.smote_variant import KNNOR_SMOTE
from model import TT_SMOTE
from numpy import where
import smote_variants as sv

def plot(x1, y1, x2, name, model):
    plt.figure(dpi=80)
    ax1 = plt.axes(projection='3d')
    row_ix = where(y1 == 0)[0]
    ax1.scatter3D(x1[row_ix, 0], x1[row_ix, 1], x1[row_ix, 2], c='red', label='majority sample')
    row_ix = where(y1 == 1)[0]
    ax1.scatter3D(x1[row_ix, 0], x1[row_ix, 1], x1[row_ix, 2], c='blue', label='minority sample')
    # new sample plot\
    if x2 != []:
        row_ix = where(x2)[0]
        ax1.scatter3D(x2[row_ix, 0], x2[row_ix, 1], x2[row_ix, 2], c='green', label='new sample')
    plt.title(f'{name}')
    plt.savefig(f'figure/{name}+{model}.png')
    plt.show()

data = [imbd.load_ecoli_0_1_4_6_vs_5(),
        imbd.load_ecoli1(),
        imbd.load_glass4(),
        imbd.load_yeast_0_2_5_7_9_vs_3_6_8()]
# data = [
#         imbd.load_abalone_20_vs_8_9_10(),
#         imbd.load_abalone_3_vs_11(),
#         imbd.load_abalone9_18(),
#         imbd.load_ecoli_0_1_4_6_vs_5(),
#         imbd.load_ecoli_0_2_3_4_vs_5(),
#         imbd.load_ecoli_0_3_4_6_vs_5(),
#         imbd.load_ecoli4(),
#         imbd.load_glass_0_1_4_6_vs_2(),
#         imbd.load_glass_0_1_5_vs_2(),
#         imbd.load_glass_0_1_6_vs_2(),
#         imbd.load_glass_0_1_6_vs_5(),
#         imbd.load_glass_0_4_vs_5(),
#         imbd.load_glass_0_6_vs_5(),
#         imbd.load_glass2(),
#         imbd.load_glass4(),
#         imbd.load_glass5(),
#         imbd.load_yeast_0_2_5_6_vs_3_7_8_9(),
#         imbd.load_yeast_0_2_5_7_9_vs_3_6_8(),
#         imbd.load_yeast_1_2_8_9_vs_7(),
#         imbd.load_yeast_1_4_5_8_vs_7(),
#         imbd.load_yeast_1_vs_7(),
#         imbd.load_yeast_2_vs_4(),
#         imbd.load_yeast_2_vs_8(),
#         imbd.load_ecoli1(),
#         imbd.load_ecoli2(),
#         imbd.load_ecoli3(),
#         imbd.load_glass_0_1_2_3_vs_4_5_6(),
#         imbd.load_glass0(),
#         imbd.load_glass6(),
#         imbd.load_new_thyroid1(),
#         imbd.load_pima(),
#         imbd.load_yeast1(),
#         imbd.load_yeast3()]

model = ['NON', 'SMOTE', 'Borderline-SMOTE', 'ADASYN', 'G-SMOTE', 'MWMOTE', 'Gaussian-SMOTE', 'DTO-SMOTE', 'KNNOR-SMOTE', 'CS-SMOTE']
model = ['CS-SMOTE']

for j in data:
    x, y = j['data'], j['target']
    num = len(x)


    pca = PCA(n_components=3)
    z_scaler = preprocessing.StandardScaler()
    X_b = z_scaler.fit_transform(x)
    X_r = pca.fit(X_b).transform(X_b)

    for i in model:
        smote_x = []
        if i != 'CS-SMOTE':
            if i == 'SMOTE':
                smote_model = SMOTE(random_state=42)
                smote_x, smote_y = smote_model.fit_resample(X_r, y)
            elif i == 'Borderline-SMOTE':
                smote_model = BorderlineSMOTE(random_state=42, kind='borderline-1')
                smote_x, smote_y = smote_model.fit_resample(X_r, y)
            elif i == 'ADASYN':
                smote_model = ADASYN(random_state=42)
                smote_x, smote_y = smote_model.fit_resample(X_r, y)
            elif i == 'G-SMOTE':
                smote_model = GeometricSMOTE(random_state=42)
                smote_x, smote_y = smote_model.fit_resample(X_r, y)
            elif i == 'DTO-SMOTE':
                smote_model = DTO(dataset_name='txt', geometry='radius_ratio', dirichlet=1, random_state=42)
                smote_x, smote_y = smote_model.fit_resample(X_r, y)
            elif i == 'MWMOTE':
                smote_model = sv.MWMOTE(oversampler_params={'random_state': 42})
                smote_x, smote_y = smote_model.fit_resample(X_r, y)
            elif i == 'Gaussian-SMOTE':
                smote_model = sv.Gaussian_SMOTE(oversampler_params={'random_state': 42})
                smote_x, smote_y = smote_model.fit_resample(X_r, y)
            elif i == 'KNNOR-SMOTE':
                smote_model = KNNOR_SMOTE(random_state=42)
                smote_x, smote_y = smote_model.fit_resample(X_r, y)

        else:
            smote_model = TT_SMOTE()
            smote_x, smote_y = smote_model.fit_resample(j['data'], y, state=True)
            smote_x = np.array([list(i) for i in smote_x])
            print(smote_x)

        smote_x = smote_x[num:]
        plot(X_r, y, smote_x, i, j['name'])
















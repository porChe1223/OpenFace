from re import A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.animation import ArtistAnimation

RESULT_CSV_PATH = './DataResult_Speak/speak1.csv'
RESULT_2D_GRAPH_PATH = './DataResult_Speak/speak1_2d.png'
RESULT_3D_GRAPH_PATH = './DataResult_Speak/speak1_3d.png'

# OpenFace出力ファイル(csv)の読み込み
data = pd.read_csv(RESULT_CSV_PATH)

# face_idが1(person1)となる配列のインデックスを取得
index = np.where(data['face_id'] == 0)

frame = data['frame'][index[0]]
AU10 = data['AU10_r'][index[0]] # 上唇を上げる
# AU11 = data['AU11_r'][index[0]] # 鼻唇溝を深める
AU12 = data['AU12_r'][index[0]] # 唇両端を上げる
# AU13 = data['AU13_r'][index[0]] # 唇を鋭く引き上げる
AU14 = data['AU14_r'][index[0]] # えくぼをつくる
AU15 = data['AU15_r'][index[0]] # 唇両端を下げる
# AU16 = data['AU16_r'][index[0]] # 下唇を下げる
AU17 = data['AU17_r'][index[0]] # 下顎(頤(オトガイ))を上げる
# AU18 = data['AU18_r'][index[0]] # 唇をすぼめる
# AU19 = data['AU19_r'][index[0]] # 舌を見せる
AU20 = data['AU20_r'][index[0]] # 唇両端を横に引く
# AU21は首に関する内容のため除外
# AU22 = data['AU22_r'][index[0]] # 唇を丸める
AU23 = data['AU23_r'][index[0]] # 唇を固く閉じる
# AU24 = data['AU24_r'][index[0]] # 唇を押し付ける
AU25 = data['AU25_r'][index[0]] # 顎を下げずに唇を開く
AU26 = data['AU26_r'][index[0]] # 顎を下げて唇を開く
# AU27 = data['AU27_r'][index[0]] # 口を大きく開く
# AU28 = data['AU28_r'][index[0]] # 唇を吸い込む

# 2Dグラフ図示
plt.figure(figsize = (8, 4))
plt.plot(frame, AU10, label = '上唇を上げる')
# plt.plot(frame, AU11, label = '鼻唇溝を深める')
plt.plot(frame, AU12, label = '唇両端を上げる')
# plt.plot(frame, AU13, label = '唇を鋭く引き上げる')
plt.plot(frame, AU14, label = 'えくぼをつくる')
plt.plot(frame, AU15, label = '唇両端を下げる')
# plt.plot(frame, AU16, label = '下唇を下げる')
plt.plot(frame, AU17, label = '下顎(頤(オトガイ))を上げる')
# plt.plot(frame, AU18, label = '唇をすぼめる')
# plt.plot(frame, AU19, label = '舌を見せる')
plt.plot(frame, AU20, label = '唇両端を横に引く')
# plt.plot(frame, AU22, label = '唇を丸める')
plt.plot(frame, AU23, label = '唇を固く閉じる')
# plt.plot(frame, AU24, label = '唇を押し付ける')
plt.plot(frame, AU25, label = '顎を下げずに唇を開く')
plt.plot(frame, AU26, label = '顎を下げて唇を開く')
# plt.plot(frame, AU27, label = '口を大きく開く')
# plt.plot(frame, AU28, label = '唇を吸い込む')

plt.xlabel('Frame')
plt.ylabel('Gaze')

plt.legend()
plt.savefig(RESULT_2D_GRAPH_PATH)

# # 3Dグラフ
# fig = plt.figure(figsize = (8, 8))
# ax = fig.add_subplot(111, projection = '3d')
# ax.plot(AU10, AU11, AU12)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.view_init(elev = 300, azim = 270) # 270,270がカメラと同じ視点

# fig.savefig(RESULT_3D_GRAPH_PATH)
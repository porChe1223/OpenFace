from re import A
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.animation import ArtistAnimation
import cv2
import os
from langchain_openai import ChatOpenAI
import openai
from langchain.prompts import PromptTemplate
from PIL import Image, ImageDraw, ImageFont
import subprocess
import logging

# ロギングの基本設定
logging.basicConfig(
    level=logging.INFO,  # INFOレベル以上のログを表示
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日時、ログレベル、メッセージを表示
)

# OpenAI APIキー
openai_api_key = os.getenv("OPENAI_API_KEY")

#==============
# パスの設定
#==============
DATA_PATH = './Data/speak1.mp4'
AUDIO_PATH = './DataResult_Speak/speak1.wav'
AUDIO_CSV_PATH = './DataResult_Speak/speak1_audio.csv'
RESULT_PATH = './DataResult_Speak'
RESULT_AVI_PATH = './DataResult_Speak/speak1.avi'
RESULT_MP4_PATH = './DataResult_Speak/speak1.mp4'
RESULT_CSV_PATH = './DataResult_Speak/speak1.csv'
RESULT_2D_GRAPH_PATH = './DataResult_Speak/speak1_2d.png'
RESULT_VOWEL_PATH = './DataResult_Speak/vowel_detection.csv'
RESULT_TEXTED_PATH = './DataResult_Speak/speak1_text.mp4'
RESULT_AUDIO_PATH = './DataResult_Speak/speak1_audio.mp4'
RESULT_CONSIDER_PATH = './DataResult_Speak/speak1_consider.mp4'
RESULT_FINAL_PATH = './DataResult_Speak/speak1_final.mp4'


######################
######################
## OpenFaceを実行
##
# OpenFaceを実行
cmd_1 = f'./OpenFace/build/bin/FaceLandmarkVidMulti -f {DATA_PATH} -out_dir {RESULT_AVI_PATH}'
subprocess.run(cmd_1, shell=True)

# AVIファイルをMP4ファイルに変換
cmd_2 = f'ffmpeg -i {RESULT_AVI_PATH} -vcodec h264 {RESULT_MP4_PATH}'
subprocess.run(cmd_2, shell=True)


###################################
###################################
## 発話に該当するActionUnitの検出
## - ActionUnitの検出
## - 2Dグラフの作成 
##
#====================
# ActionUnitの検出
#====================
logging.info("[START]: 表情認識")
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
logging.info("[END]: 表情認識")

#=================
# 2Dグラフの作成
#=================
logging.info("[START]: 2Dグラフの作成")
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

# ラベルの設定
plt.xlabel('Frame')
plt.ylabel('Gaze')

# グラフの描画
plt.legend()

# グラフの保存
plt.savefig(RESULT_2D_GRAPH_PATH)
logging.info("[END]: 2Dグラフの作成")


#############################
#############################
## 発話部分を音声認識で検出
##
#=====================
# 動画から音声を抽出
#=====================
cmd_audio = f"ffmpeg -i {DATA_PATH} -q:a 0 -map a {AUDIO_PATH} -y"
subprocess.run(cmd_audio, shell=True)

#=============
# 音声認識
#=============
logging.info("[START]: 音声認識")
recognizer = sr.Recognizer()
audio = AudioSegment.from_file(AUDIO_PATH)

# 母音を検出するフレーム単位の処理
vowel_detection = []
frame_rate = 30  # フレームレート (ビデオと同期する必要がある)
frame_duration = 1000  # 1フレームの時間(ms)

for i in range(0, len(audio), int(frame_duration)):
    start_time = i  # フレームの開始時間
    end_time = max(i + int(frame_duration), len(audio))  # フレームの終了時間
    segment = audio[start_time:end_time]  # フレームに対応する音声を取得

    try:
        # pydubのAudioSegmentを一時ファイルとして保存
        segment.export("temp.wav", format="wav")
        with sr.AudioFile("temp.wav") as source:
            audio_data = recognizer.record(source)
            detected_text = recognizer.recognize_google(audio_data, language='ja-JP')  # 日本語の場合
    except sr.UnknownValueError:
        detected_text = ""  # 音声が認識できない場合は空文字

    if detected_text == "":
        for i in range(frame_rate):
            vowel_detection.append("-")
    else:
        for i in range(frame_rate):
            vowel_detection.append("ok")

# フレームごとの結果を保存
vowel_df = pd.DataFrame({
    'Frame': range(len(vowel_detection)),
    'Vowel': vowel_detection
})
vowel_df.to_csv(AUDIO_CSV_PATH, index=False)
logging.info("[END]: 音声認識")


#######################################
#######################################
## 音声認識とActionUnitから母音を推定
## - 音声の認識結果を反映
## - ActionUnitから母音を推定
## - 動画に母音検出結果を反映
#===================
# 母音の発音を検出
#===================
logging.info("[START]: 母音の推定")
vowel_detection_result = []
for i in range(min(len(vowel_detection), 279)):
    if vowel_df['Vowel'][i] == "ok":
        if AU23[i] > 0.0:
            vowel_detection_result.append((frame[i], "-"))
        elif AU10[i] > 0.4:
            vowel_detection_result.append((frame[i], "a"))
        elif AU25[i] > 1.5 or AU20[i] > 0.1:
            if AU12[i] > 1.0 or AU14[i] > 0.3:
                vowel_detection_result.append((frame[i], "i"))
            else:
                vowel_detection_result.append((frame[i], "e"))
        elif AU26[i] > 1.3 or AU17[i] > 0.3:
            vowel_detection_result.append((frame[i], "u"))
        else:
            vowel_detection_result.append((frame[i], "o"))
    else:
        vowel_detection_result.append((frame[i], "-"))
# 検出結果をDataFrameとして保存
vowel_df_result = pd.DataFrame(vowel_detection_result, columns=['Frame', 'Vowel'])
vowel_df_result.to_csv(RESULT_VOWEL_PATH, index=False)
logging.info("[END]: 母音の推定")

#==========================
# 動画に母音検出結果を反映
#==========================
logging.info("[START]: 動画に母音検出結果を反映")
# 入力ビデオ
input_video = cv2.VideoCapture(DATA_PATH)

width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)

print(f"Input FPS: {fps}")

# 出力ビデオ
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_video = cv2.VideoWriter(RESULT_TEXTED_PATH, fmt, fps, (width, height))

# 母音検出結果の読み込み
vowel_data = pd.read_csv(RESULT_VOWEL_PATH)

cnt = 1

# 1フレームずつ処理
while True:
    ret, frame = input_video.read()  # ret...画像を読み込めたかどうか, frame...読み込んだ画像
    if not ret:  # 画像を読み込めなかった場合
        break

    # フレームに対応する母音検出結果を取得
    frame_vowel = vowel_data[vowel_data['Frame'] == cnt]

    if not frame_vowel.empty:
        vowel = frame_vowel['Vowel'].values[0]
        txt = f"Detected Vowel: {vowel}"
        pos = (50, 50)  # テキストの表示位置
        col = (255, 255, 255)  # 白色

        # フレームにテキストを追加
        frame = cv2.putText(frame, org=pos, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=col, thickness=2)

    # 出力動画に画像を追加
    output_video.write(frame)
    cnt += 1

input_video.release()
output_video.release()
logging.info("[END]: 動画に母音検出結果を反映")

#=================
# 結果動画の作成
#=================
cmd_speak = f'ffmpeg -i {RESULT_TEXTED_PATH} -i {DATA_PATH} -c:v copy -c:a aac -strict experimental {RESULT_AUDIO_PATH}'
subprocess.run(cmd_speak, shell=True)


###################################
###################################
## 状況からの発話内容の推定
##
#=================
# LLMの呼び出し
#=================
llm = ChatOpenAI(model_name="gpt-4o",
                 temperature=0,
                 openai_api_key=openai_api_key
                 )

#===============
# LLMの実行
#===============
logging.info("[START]: 状況から発話内容の推定")
# CSVファイルの読み込み
data = pd.read_csv(RESULT_VOWEL_PATH)
# プロンプト
prompt = PromptTemplate(
    input_variables = ['user_prompt'],
    template = (
        '{user_prompt}' +
        data.to_string(index=False) +
        'そのまま使いたいので説明はなしで推測文だけを返して'
    )
)
# Chainの定義
chain = (
    prompt
    | llm
)
# Chainの実行
result = chain.invoke(input = {"user_prompt": 'これはアナウンサーがシートベルトの喚起と経済情報ならテレ東Vizという宣伝をしている動画である。以下のCSVデータは母音のみを示しているが、この背景を基に何をしゃべっているか考察し、このCSVの各要素に日本語をあてはめよ'})

print(result)
logging.info("[END]: 状況から発話内容の推定")

#=======================
# 推論結果を動画に反映
#=======================
logging.info("[START]: 推論結果を動画に反映")
# 入力ビデオ
input_video = cv2.VideoCapture(DATA_PATH)

width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)

print(f"Input FPS: {fps}")

# 出力ビデオ
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_video = cv2.VideoWriter(RESULT_CONSIDER_PATH, fmt, fps, (width, height))

# 推論結果の読み込み
result_text = result.content.split('\n\n')[1]

cnt = 1

# 1フレームずつ処理
while True:
    ret, frame = input_video.read()  # ret...画像を読み込めたかどうか, frame...読み込んだ画像
    if not ret:  # 画像を読み込めなかった場合
        break

    txt = result_text
    pos = (50, 50)  # テキストの表示位置
    col = (255, 255, 255)  # 白色

    # 日本語テキストを画像に変換
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font_path = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
    font = ImageFont.truetype(font_path, 24)
    draw.text(pos, txt, font=font, fill=col)


    # フレームにテキストを追加
    frame = np.array(img_pil)

    # 出力動画に画像を追加
    output_video.write(frame)
    cnt += 1

input_video.release()
output_video.release()
logging.info("[END]: 推論結果を動画に反映")


#=================
# 最終動画の作成
#=================
cmd_speak = f'ffmpeg -i {RESULT_CONSIDER_PATH} -i {DATA_PATH} -c:v copy -c:a aac -strict experimental {RESULT_FINAL_PATH}'
subprocess.run(cmd_speak, shell=True)

import subprocess

DATA_PATH = './Data/speak1.mp4'
AUDIO_PATH = './DataResult_Speak/speak1.wav'
RESULT_PATH = './DataResult_Speak'
RESULT_AVI_PATH = './DataResult_Speak/speak1.avi'
RESULT_MP4_PATH = './DataResult_Speak/speak1.mp4'
RESULT_TEXT_PATH = './DataResult_Speak/speak1_text.mp4'
RESULT_AUDIO_PATH = './DataResult_Speak/speak1_audio.mp4'

# # OpenFaceを実行
# cmd_1 = './OpenFace/build/bin/FaceLandmarkVidMulti -f DATA_PATH -out_dir RESULT_AVI_PATH'
# cmd_1 = cmd_1.replace('DATA_PATH', DATA_PATH)
# cmd_1 = cmd_1.replace('RESULT_AVI_PATH', RESULT_PATH)
# subprocess.run(cmd_1, shell=True)

# # AVIファイルをMP4ファイルに変換
# cmd_2 = 'ffmpeg -i RESULT_AVI_PATH -vcodec h264 RESULT_MP4_PATH'
# cmd_2 = cmd_2.replace('RESULT_AVI_PATH', RESULT_AVI_PATH)
# cmd_2 = cmd_2.replace('RESULT_MP4_PATH', RESULT_MP4_PATH)
# subprocess.run(cmd_2, shell=True)

# #=====================
# # 動画から音声を抽出
# #=====================
# cmd_audio = f"ffmpeg -i DATA_PATH -q:a 0 -map a AUDIO_PATH -y"
# cmd_audio = cmd_audio.replace('DATA_PATH', DATA_PATH)
# cmd_audio = cmd_audio.replace('AUDIO_PATH', AUDIO_PATH)
# subprocess.run(cmd_audio, shell=True)


cmd_speak = 'ffmpeg -i RESULT_TEXT_PATH -i DATA_PATH -c:v copy -c:a aac -strict experimental RESULT_AUDIO_PATH'
cmd_speak = cmd_speak.replace('RESULT_TEXT_PATH', RESULT_TEXT_PATH)
cmd_speak = cmd_speak.replace('DATA_PATH', DATA_PATH)
cmd_speak = cmd_speak.replace('RESULT_AUDIO_PATH', RESULT_AUDIO_PATH)
subprocess.run(cmd_speak, shell=True)
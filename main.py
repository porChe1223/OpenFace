import subprocess

DATA_PATH = './Data/speak1.mp4'
RESULT_PATH = './DataResult_Speak'
RESULT_AVI_PATH = './DataResult_Speak/speak1.avi'
RESULT_MP4_PATH = './DataResult_Speak/speak1.mp4'

# OpenFaceを実行
cmd_1 = './OpenFace/build/bin/FaceLandmarkVidMulti -f DATA_PATH -out_dir RESULT_AVI_PATH'
cmd_1 = cmd_1.replace('DATA_PATH', DATA_PATH)
cmd_1 = cmd_1.replace('RESULT_AVI_PATH', RESULT_PATH)
subprocess.run(cmd_1, shell=True)

# AVIファイルをMP4ファイルに変換
cmd_2 = 'ffmpeg -i RESULT_AVI_PATH -vcodec h264 RESULT_MP4_PATH'
cmd_2 = cmd_2.replace('RESULT_AVI_PATH', RESULT_AVI_PATH)
cmd_2 = cmd_2.replace('RESULT_MP4_PATH', RESULT_MP4_PATH)
subprocess.run(cmd_2, shell=True)
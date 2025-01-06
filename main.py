import subprocess

DATA_PATH = './Data/speak1.mp4'
RESULT_PATH = './DataResult_Speak'
RESULT_AVI_PATH = './DataResult_Speak/speak1.avi'
RESULT_MP4_PATH = './DataResult_Speak/speak1.mp4'

# # OpenFaceを実行
# cmd = './OpenFace/build/bin/FaceLandmarkVidMulti -f DATA_PATH -out_dir RESULT_AVI_PATH'
# cmd = cmd.replace('DATA_PATH', DATA_PATH)
# cmd = cmd.replace('RESULT_AVI_PATH', RESULT_PATH)
# subprocess.run(cmd, shell=True)

# AVIファイルをMP4ファイルに変換
cmd = 'ffmpeg -i RESULT_AVI_PATH -vcodec h264 RESULT_MP4_PATH'
cmd = cmd.replace('RESULT_AVI_PATH', RESULT_AVI_PATH)
cmd = cmd.replace('RESULT_MP4_PATH', RESULT_MP4_PATH)
subprocess.run(cmd, shell=True)
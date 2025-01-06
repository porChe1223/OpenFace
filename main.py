import subprocess

DATA_PATH = './Data/speak1.mp4'
RESULT_PATH = './DataResult_Speak'

# OpenFaceを実行
cmd = './OpenFace/build/bin/FaceLandmarkVidMulti -f DATA_PATH -out_dir RESULT_AVI_PATH'
cmd = cmd.replace('DATA_PATH', DATA_PATH)
cmd = cmd.replace('RESULT_AVI_PATH', RESULT_PATH)
subprocess.run(cmd, shell=True)
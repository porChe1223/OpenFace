# OpenFace を用いた様々な検出

## 環境構築

### OpenFace をダウンロード

git clone --depth 1 'https://github.com/TadasBaltrusaitis/OpenFace.git'

### 中身の install.sh を入れ替える

### OpenFace をインストール

cd OpenFace  
bash ./download_models.sh && sudo bash ./install.sh

### OpenFace の実行権限を付与

cd ..  
chmod 755 OpenFace/build/bin/FaceLandmarkVidMulti

### requirements.txt をインストール

pip install -r requirements.txt

### OpenFace の実行結果動画を mp4 で取得したい場合

sudo apt update  
sudo apt install ffmpeg

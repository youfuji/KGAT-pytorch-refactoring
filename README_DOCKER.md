# Docker実行ガイド

このプロジェクトをDocker環境（GPU対応）で実行するための手順書です。
ホスト環境を汚さずに、隔離された環境でKGATやAKDNを実行できます。

## 前提条件

*   Linux環境 (Ubuntuなど)
*   Docker がインストールされていること
*   NVIDIA Driver および [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) がインストールされていること（GPUを使用するため）

## ファイル構成

*   `Dockerfile`: コンテナイメージの定義（PyTorch 2.9.0, CUDA 13.0ベース）
*   `requirements.txt`: 必要なPythonライブラリ一覧
*   `build_docker.sh`: Dockerイメージを作成するスクリプト
*   `run_docker.sh`: コンテナを起動して中に入るスクリプト（GPU有効化済み）

## 1. 環境構築 (初回のみ)

まず、各種スクリプトに実行権限を与え、Dockerイメージをビルドします。
ターミナルで以下のコマンドを実行してください。

```bash
# スクリプトに実行権限を付与
chmod +x build_docker.sh run_docker.sh

# Dockerイメージのビルド
./build_docker.sh
```

時間がかかりますが、完了するまで待機してください。

## 2. コンテナの起動

ビルドが完了したら、以下のコマンドでコンテナを起動し、シェルに入ります。

```bash
./run_docker.sh
```

実行するとプロンプトが変わり（例：`root@<container_id>:/workspace#`）、コンテナ内部の操作モードになります。
現在のディレクトリ（`/workspace`）には、ホスト側のファイルがマウントされています。

## 3. プログラムの実行

コンテナ内部で、通常通りPythonコマンドを実行してください。

**KGATの実行例:**
```bash
python main_kgat.py --data_name amazon-book
```

**AKDNの実行例:**
```bash
python main_akdn.py
```

**その他のモデル:**
```bash
python main_nfm.py --model_type nfm --data_name amazon-book
python main_bprmf.py --data_name amazon-book
```

## 4. 終了方法

*   **実行中のプログラムを停止する**: `Ctrl + C` を押してください。
*   **コンテナ環境から抜ける**: コマンドラインで `exit` と入力して Enter を押してください。

```bash
exit
```

コンテナは `--rm` オプション付きで起動しているため、終了と同時に自動的に削除されます。
ただし、カレントディレクトリ（`/workspace`）はホスト側のフォルダがマウントされているため、ここに保存されたファイル（学習済みモデルやログなど）は**消えずに残ります**。

## 5. 再起動・再実行

一度コンテナから抜けた後、再度作業を始めたい場合は、もう一度起動スクリプトを実行してください。

```bash
./run_docker.sh
```

これにより、新しいコンテナが立ち上がり、再び同じ環境で作業できます。
※ コンテナのファイルシステム（`/workspace` 以外）への変更はリセットされます。

## トラブルシューティング

**Q. `nvidia-smi` がコンテナ内で見つからない / GPUが使われていない**
A. ホスト側で `nvidia-smi` が実行できるか確認してください。また、`./run_docker.sh` スクリプト内で `--gpus all` オプションが指定されているか確認してください。

**Q. Permission denied でスクリプトが実行できない**
A. `chmod +x build_docker.sh run_docker.sh` を再実行してください。

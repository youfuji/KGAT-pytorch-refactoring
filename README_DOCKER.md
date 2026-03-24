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

## 6. 実運用チートシート（起動・実行・再起動・停止・監視）

ここだけ見れば、もう一度実行できます。

### 6.1 起動（対話モード）

コンテナ内に入って手動で実行したい場合:

```bash
./run_docker.sh
```

コンテナ内で学習を開始:

```bash
python main_akdn.py --data_name yelp2018 --n_epoch 1000 --cf_print_every 100 --stopping_steps 3
```

### 6.2 起動（バックグラウンド運用）

再接続しても処理を継続したい場合:

```bash
# 既存コンテナが残っていれば削除
docker rm -f kgat-runner 2>/dev/null || true

# バックグラウンド起動
docker run --gpus all -d --name kgat-runner -v "$(pwd)":/workspace kgat-pytorch sleep infinity

# 学習開始（ログをファイル保存）
docker exec -d kgat-runner bash -lc 'cd /workspace && nohup python main_akdn.py --data_name yelp2018 --n_epoch 1000 --cf_print_every 100 --stopping_steps 3 > /workspace/retrain_gpu.log 2>&1 &'
```

### 6.3 再起動

```bash
docker rm -f kgat-runner
docker run --gpus all -d --name kgat-runner -v "$(pwd)":/workspace kgat-pytorch sleep infinity
```

### 6.4 停止

学習プロセスだけ止める:

```bash
docker exec kgat-runner pkill -f 'python main_akdn.py'
```

コンテナごと止める（削除もする）:

```bash
docker rm -f kgat-runner
```

### 6.5 監視

コンテナ状態:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

コンテナ内プロセス:

```bash
docker top kgat-runner -eo pid,pcpu,pmem,args
```

学習ログ（リアルタイム）:

```bash
tail -f retrain_gpu.log
```

GPU使用率（ホスト）:

```bash
nvidia-smi
```

GPU使用プロセス（ホスト）:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

### 6.6 GPU確認（必須）

学習前に以下2つを確認してください。

```bash
docker exec kgat-runner python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
docker exec kgat-runner nvidia-smi
```

期待値:
- `True 1` など、True と GPU台数が出る
- `nvidia-smi` で RTX 4090 が表示される

## トラブルシューティング

**Q. `nvidia-smi` がコンテナ内で見つからない / GPUが使われていない**
A. ホスト側で `nvidia-smi` が実行できるか確認してください。また、`./run_docker.sh` スクリプト内で `--gpus all` オプションが指定されているか確認してください。

**Q. Permission denied でスクリプトが実行できない**
A. `chmod +x build_docker.sh run_docker.sh` を再実行してください。

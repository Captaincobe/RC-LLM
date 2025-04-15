#!/bin/bash

REMOTE_USER=uestc_zhou       # ← 远程服务器用户名
REMOTE_HOST=1.192.212.113  # 如 aiwen2.xx.com
REMOTE_PORT=11822
REMOTE_PATH=/home/uestc_zhou/zmh/RCML-main

echo "同步本地文件到远程服务器..."
rsync -avz -e "ssh -p $REMOTE_PORT" --exclude='.git' --exclude='__pycache__' ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo "同步完成 ✅"
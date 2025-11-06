#!/bin/bash
set -e  # 出错即停止

# 拉取 Gitee 最新代码
echo "=== 拉取 Gitee 最新代码 ==="
git pull gitee master

# 拉取 GitHub 最新代码（避免遗漏）
echo "=== 拉取 GitHub 最新代码 ==="
git pull github master

# 推送代码到 Gitee
echo "=== 推送到 Gitee ==="
git push gitee master

# 推送代码到 GitHub
echo "=== 推送到 GitHub ==="
git push github master

echo "=== 同步完成！==="
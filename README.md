# 航空术语智能问答系统

大学生创新创业项目：基于 RAG（检索增强生成）+ 通义千问的航空术语智能问答 Web 系统。

## 项目架构
- 后端：FastAPI（main.py），提供网页与 /qa 问答接口
- 数据源：阿里云 MySQL（队友 RDS），表 term_main（9746 条航空术语）
- 向量库：ChromaDB（build_kb.py 从 MySQL 重建，本地运行一次即可）
- 大模型：阿里云百炼通义千问 qwen-plus（调用云端 API，非本地部署）

## 云端部署（GitHub Codespaces）
1. 在仓库打开 Codespace
2. 配置环境变量（Secrets）：
   - DASHSCOPE_API_KEY
   - MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PWD / MYSQL_DB
3. 安装依赖：
   pip install -r requirements.txt
4. 首次构建向量库（约几分钟）：
   python build_kb.py
5. 启动服务：
   uvicorn main:app --host 0.0.0.0 --port 8000
6. Ports 面板将 8000 端口设为 Public，复制公网链接即可访问

## 说明
- 密钥全部通过环境变量注入，不写入代码仓库
- 免费 Codespaces 每月约 60 小时，长时间无访问会休眠，重新启动服务即可恢复

# etf-qim
ETF quantitative investment management



### 1. 安装 Poetry

Poetry 是一个用于 Python 依赖管理和打包的工具。

**Windows (PowerShell):**

```powershell
(Invoke-WebRequest -Uri [https://install.python-poetry.org](https://install.python-poetry.org) -UseBasicParsing).Content | py -
```

**Unix/macOS:**

```bash
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
```

### 2. 安装项目依赖

使用 Poetry 安装项目所需的依赖包：

```bash
poetry lock --no-update
```

```bash
poetry install
```

### 3. 配置环境变量

环境变量用于存储 API 密钥等敏感信息。

首先，复制示例环境变量文件：

```bash
# Create .env file for your API keys
cp .env.example .env
```

**基本运行 (只显示关键决策信息):**
```bash
poetry run python src/main.py
```

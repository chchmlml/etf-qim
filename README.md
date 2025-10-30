etf-qim
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


trading_strategy
完成项目的交易策略模块
1、整体配置文件
strategy_config.py
定义了总投入、最大买入基金数量、每支占比、基金分位线和回撤线
2、买入策略：读取基金的pe数据文件 #File:
050001_pe_data.csv
计算基金的低分位线、高分位线，当天处于低分位线进行买入一个头寸操作；注意每支最大占比限制；
3、卖出操作同买入操作计算基金的低分位线、高分位线，买入后基金已经突破高分位线，当天回撤达到1回撤线标准，进行清仓操作；
4、低分位线、高分位线处于之间不操作，持有策略
5、每个买卖策略单独一个文件，代码设计上方便后期扩展更多策略
6、每次交易写入文件文件格式trade_transactions.csv
trade_id,date,symbol,trade_type,price,quantity,commission,notes
1,2023-01-10,050020,BUY,1.300,1000,0.0013,Initial purchase based on low PE
2,2023-03-15,050020,SELL,1.450,500,0.0007,Partial sell due to PE rebound
3,2023-05-20,050020,BUY,1.400,200,0.0003,Add to position after minor correction
7、交易逻辑在workflow.py实现，要基于langgraph实现流​

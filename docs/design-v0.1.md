# Stock V0.1 — 数据采集系统设计文档

> 版本: V0.1 | 日期: 2026-07-06 | 状态: 待确认

---

## 1. 概述

### 1.1 目标

搭建股票数据采集底层，为后续策略研究提供统一的数据访问接口。

### 1.2 数据源

| 数据源 | 采集内容 | 用途 |
|:---|:---|:---|
| **Tencent** `qt.gtimg.cn` | 盘中实时快照（30s/轮） | 实时日K更新、盘中监控 |
| **Baostock** | 日/周/月K线（不复权）、行业分类、复权因子、交易日历 | 历史数据、补全、校准 |
| **Akshare** | 财务摘要（同花顺）、业绩快报（东方财富）、指数成分股（中证） | 基本面分析、成分筛选 |

### 1.3 股票范围

上证主板（60xxxx）+ 深证主板（00xxxx），约 3000 只。

---

## 2. 项目结构

```
stock_V0.1/
├── src/
│   ├── __init__.py
│   ├── config.py                # 全部配置常量
│   ├── collectors/              # 各数据源采集器
│   │   ├── __init__.py
│   │   ├── baostock.py          # K线 / 行业分类 / 复权因子 / 交易日历
│   │   ├── akshare.py           # 财务摘要 / 业绩快报 / 指数成分股
│   │   └── tencent.py           # 盘中实时快照
│   ├── storage/                 # 存储层
│   │   ├── __init__.py
│   │   ├── database.py          # SQLite 连接管理、建表、CRUD
│   │   └── archiver.py          # 快照归档（→ Parquet）
│   ├── scheduler/               # 调度与流程控制
│   │   ├── __init__.py
│   │   ├── calendar.py          # 交易日历
│   │   └── runner.py            # 快照主循环
│   └── validate.py              # 数据校验（daily_kline vs Baostock）
├── scripts/                     # 入口脚本（开发用）
│   ├── snapshot.py              # 启动盘中快照
│   ├── backfill.py              # Baostock 补全日K
│   ├── validate.py              # 触发数据校验
│   └── init_db.py               # 初始化数据库
├── data/                        # 本地数据（不入 git）
│   ├── stock.db                 # SQLite 数据库
│   └── archive/                 # 快照归档 Parquet 文件
├── logs/                        # 日志文件（不入 git）
├── docs/                        # 文档
│   └── design-v0.1.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. 数据库设计

### 3.1 ER 概要

```
stock_basic (股票基础信息)
    │
    ├── daily_kline (日K线)
    │       │
    │       ├── weekly_kline (周K线，从日K聚合)
    │       └── monthly_kline (月K线，从日K聚合)
    │
    ├── adjust_factor (复权因子)
    ├── industry_class (申万一级行业分类)
    ├── financial_summary (财务摘要，按需)
    ├── performance_express (业绩快报，按需)
    ├── index_constituent (指数成分股，按需)
    └── snapshot_raw (快照原始数据，仅保留5个交易日)
```

### 3.2 表结构

#### stock_basic

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `code` | TEXT PK | 统一格式：`sh.600000` / `sz.000001` |
| `name` | TEXT | 股票名称 |
| `ipo_date` | TEXT | 上市日期 YYYY-MM-DD |
| `delist_date` | TEXT | 退市日期，NULL=正常交易 |
| `board` | TEXT | 板块：`sh_main` / `sz_main` |
| `updated_at` | TEXT | 最后更新时间 |

#### daily_kline

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `code` | TEXT PK (复合) | 股票代码 `sh.600000` |
| `trade_date` | TEXT PK (复合) | 交易日期 YYYY-MM-DD |
| `open` | REAL | 开盘价 |
| `high` | REAL | 最高价 |
| `low` | REAL | 最低价 |
| `close` | REAL | 收盘价（盘中为最新价） |
| `preclose` | REAL | 昨日收盘价 |
| `volume` | REAL | 成交量（股） |
| `amount` | REAL | 成交额（元） |
| `turn` | REAL | 换手率 % |
| `pct_chg` | REAL | 涨跌幅 % |
| `pe_ttm` | REAL | 滚动市盈率 |
| `pb_mrq` | REAL | 市净率 |
| `ps_ttm` | REAL | 滚动市销率 |
| `pcf_ttm` | REAL | 滚动市现率 |
| `total_mv` | REAL | 总市值（元） |
| `circ_mv` | REAL | 流通市值（元） |
| `amplitude` | REAL | 振幅 % |
| `vol_ratio` | REAL | 量比 |
| `avg_price` | REAL | 均价 |
| `limit_up` | REAL | 涨停价 |
| `limit_down` | REAL | 跌停价 |
| `is_st` | INTEGER | 是否ST：1=是 0=否 |
| `data_source` | TEXT | 来源：`snapshot` / `baostock` |
| `updated_at` | TEXT | 最后更新时间 |

> 主键: `(code, trade_date)`

#### weekly_kline / monthly_kline

与 daily_kline 字段相同，`trade_date` 存周期首日（周K=周一日期，月K=1日日期），从 daily_kline 收盘后自动聚合。

#### adjust_factor

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `code` | TEXT PK (复合) | 股票代码 |
| `trade_date` | TEXT PK (复合) | 交易日期 |
| `adj_factor` | REAL | 复权因子 |
| `adj_factor_after` | REAL | 后复权因子 |

#### industry_class

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `code` | TEXT PK (复合) | 股票代码 |
| `level` | TEXT PK (复合) | `一级` |
| `industry` | TEXT | 申万行业名称 |
| `updated_at` | TEXT | 更新时间 |

#### financial_summary（同花顺，按需拉取）

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `code` | TEXT PK (复合) | 股票代码 |
| `report_date` | TEXT PK (复合) | 报告期 YYYY-MM-DD |
| ... | | Akshare 返回的全部字段 |

#### performance_express（东方财富，按需拉取）

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `code` | TEXT PK (复合) | 股票代码 |
| `report_date` | TEXT PK (复合) | 报告期 |
| ... | | Akshare 返回的全部字段 |

#### index_constituent（中证，手动触发）

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `index_code` | TEXT PK (复合) | 指数代码，如 `000300`（沪深300） |
| `code` | TEXT PK (复合) | 成分股代码 |
| `in_date` | TEXT | 纳入日期 |
| `out_date` | TEXT | 剔除日期，NULL=仍在成分中 |

#### snapshot_raw

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `id` | INTEGER PK | 自增 |
| `code` | TEXT | 股票代码 |
| `snap_time` | TEXT | 快照时间 YYYY-MM-DD HH:MM:SS |
| `trade_date` | TEXT | 交易日期 YYYY-MM-DD（索引） |
| `price` | REAL | 最新价 |
| `open` | REAL | 今开 |
| `high` | REAL | 日内最高（累计） |
| `low` | REAL | 日内最低（累计） |
| `preclose` | REAL | 昨收 |
| `volume` | REAL | 成交量（手） |
| `amount` | REAL | 成交额（万） |
| `turn` | REAL | 换手率 % |
| `pct_chg` | REAL | 涨跌幅 % |
| `pe_ttm` | REAL | 市盈率 |
| `pb` | REAL | 市净率 |
| `total_mv` | REAL | 总市值 |
| `circ_mv` | REAL | 流通市值 |
| `vol_ratio` | REAL | 量比 |
| `amplitude` | REAL | 振幅 % |
| `avg_price` | REAL | 均价 |
| `limit_up` | REAL | 涨停价 |
| `limit_down` | REAL | 跌停价 |
| `bid1_price` ~ `bid5_price` | REAL | 买一~买五价 |
| `bid1_vol` ~ `bid5_vol` | REAL | 买一~买五量 |
| `ask1_price` ~ `ask5_price` | REAL | 卖一~卖五价 |
| `ask1_vol` ~ `ask5_vol` | REAL | 卖一~卖五量 |
| `outside` | REAL | 外盘（手） |
| `inside` | REAL | 内盘（手） |
| `trade_status` | TEXT | 交易状态 |

> 每天约 144 万行（3000 只 × 480 轮）。保留最近 5 个交易日，超出归档为 Parquet。

#### run_log

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `id` | INTEGER PK | 自增 |
| `run_type` | TEXT | `snapshot` / `backfill` / `validate` |
| `start_time` | TEXT | 开始时间 |
| `end_time` | TEXT | 结束时间 |
| `total_stocks` | INTEGER | 处理股票数 |
| `total_batches` | INTEGER | 总批次数 |
| `failed_batches` | INTEGER | 失败批次数 |
| `rows_inserted` | INTEGER | 插入行数 |
| `status` | TEXT | `success` / `warning` / `error` |
| `message` | TEXT | 摘要 |

---

## 4. 数据流

### 4.1 盘中实时快照 → 日K (字段级)

#### 快照更新 daily_kline 字段对照

| daily_kline 字段 | 快照来源 (Tencent index) | 更新方式 | 说明 |
|:---|:---|:---|:---|
| `open` | index 5 (今开) | 覆盖写 | 当日不变 |
| `high` | index 33 (最高价) | 覆盖写 | 日内累计值，最后一轮 = 当日最高 |
| `low` | index 34 (最低价) | 覆盖写 | 日内累计值，最后一轮 = 当日最低 |
| `close` | index 3 (最新价) | 覆盖写 | 盘中随最新价实时变化 |
| `preclose` | index 4 (昨收) | 覆盖写 | 当日不变 |
| `volume` | index 6 × 100 | 覆盖写 | 腾讯单位"手"，×100 转为"股" |
| `amount` | index 37 × 10000 | 覆盖写 | 腾讯单位"万"，×10000 转为"元" |
| `turn` | index 38 | 覆盖写 | 换手率 % |
| `pct_chg` | index 32 | 覆盖写 | 涨跌幅 % |
| `pe_ttm` | index 39 | 覆盖写 | 滚动市盈率 |
| `pb_mrq` | index 46 | 覆盖写 | 市净率 |
| `total_mv` | index 45 | 覆盖写 | 总市值（元） |
| `circ_mv` | index 44 | 覆盖写 | 流通市值（元） |
| `amplitude` | index 43 | 覆盖写 | 振幅 % |
| `vol_ratio` | index 49 | 覆盖写 | 量比 |
| `avg_price` | index 51 | 覆盖写 | 均价 |
| `limit_up` | index 47 | 覆盖写 | 涨停价 |
| `limit_down` | index 48 | 覆盖写 | 跌停价 |
| `data_source` | — | 写 `snapshot` | 标记数据来源 |
| `updated_at` | — | 写当前时间 | |

#### Baostock 独有字段 (快照不更新)

| daily_kline 字段 | 说明 |
|:---|:---|
| `ps_ttm` | 滚动市销率，仅 Baostock 提供 |
| `pcf_ttm` | 滚动市现率，仅 Baostock 提供 |
| `is_st` | 是否 ST 股，仅 Baostock 提供 |

> 这三个字段在快照写入时保留原值（如有），Baostock 补全时填入。

#### 执行流程

```
┌──────────────────────────────────────────────────┐
│              snapshot.py 手动启动                  │
│                                                   │
│  while 交易时段:                                   │
│    1. 从 stock_basic 获取股票池                    │
│    2. 分批请求 qt.gtimg.cn (400只/批)              │
│    3. 解析响应 → snapshot_raw 批量 INSERT           │
│    4. 对每只股票 UPDATE OR INSERT daily_kline:      │
│       - 按上表字段对照写入（覆盖策略）                │
│    5. 等待至下一轮 (30秒间隔)                        │
│                                                   │
│  启动检查: 跳过非交易日, 仅在 9:30-11:30 &          │
│            13:00-15:00 执行                        │
│  保证采集 11:30 和 15:00 后的最后一轮数据             │
└──────────────────────────────────────────────────┘
```

### 4.2 Baostock 补全日K

```
backfill.py 手动触发
    1. 遍历 stock_basic 中的股票
    2. 对每只股票，查询 daily_kline 中的最大 trade_date
    3. 若缺失日期 > 1天，调用 Baostock 拉取缺失区间
    4. 写入 daily_kline (data_source = 'baostock')
       - 采用断点续跑，每只股票间隔请求避免限流
       - 连接断开时等待后重连
```

### 4.3 Baostock 批量导入（首次）

```
init_db.py 或 backfill.py --full
    1. 首次安装时拉取最近 500 个交易日全量日K
    2. 拉取全部周K/月K
    3. 拉取全部复权因子
    4. 拉取全部行业分类
    5. 拉取交易日历
    6. 初始化 stock_basic 表
```

### 4.4 周K/月K 聚合

```
从 daily_kline 聚合:
    - 周K: GROUP BY strftime('%Y-%W', trade_date)
    - 月K: GROUP BY strftime('%Y-%m', trade_date)
    - open  = 周期第一天 open
    - high   = MAX(high)
    - low    = MIN(low)
    - close  = 周期最后一天 close
    - volume = SUM(volume)
    - amount = SUM(amount)
```

### 4.5 快照归档（收盘后自动执行）

```
每个交易日 15:00 后自动触发:
    1. 查询 snapshot_raw 中 trade_date < date('now', '-5 days') 的数据
    2. 导出为 Parquet → data/archive/snapshot_YYYYMMDD.parquet
    3. 验证 Parquet 完整性后，从 SQLite 删除旧数据
```

### 4.6 数据校验（手动触发）

```
validate.py 手动触发
    1. 选定日期范围（默认最近 5 个交易日）
    2. 对范围每只股票：
       - 从 daily_kline 读取 snapshot 来源的 OHLCV
       - 从 Baostock 拉取同日官方日K
       - 对比 OHLCV，差异 > 0.5% 写入日志 warning
    3. 生成纯文本校验日志（不自动覆盖）
```

---

## 5. 配置项 (src/config.py)

```python
# 数据库
DB_PATH = "data/stock.db"

# 快照采集
SNAPSHOT_BATCH_SIZE = 400          # 每批股票数
SNAPSHOT_INTERVAL = 30             # 轮询间隔（秒）
SNAPSHOT_MARKET_OPEN = "09:30"
SNAPSHOT_MARKET_CLOSE_AM = "11:30"
SNAPSHOT_MARKET_OPEN_PM = "13:00"
SNAPSHOT_MARKET_CLOSE_PM = "15:00"

# 数据保留
SNAPSHOT_RETAIN_DAYS = 5           # 快照在 DB 中保留天数
SNAPSHOT_ARCHIVE_DIR = "data/archive"

# Baostock
BAOSTOCK_RETRY_COUNT = 3
BAOSTOCK_RETRY_INTERVAL = 2        # 秒
BAOSTOCK_REQUEST_GAP = 0.5         # 请求间隔（秒），避免限流

# 校验
VALIDATE_DIFF_THRESHOLD = 0.005    # OHLCV 差异阈值 (0.5%)
VALIDATE_DEFAULT_DAYS = 5          # 默认校验最近 N 日

# 日志
LOG_DIR = "logs"
LOG_FILE_LEVEL = "WARNING"
LOG_CONSOLE_LEVEL = "INFO"
LOG_RETENTION_DAYS = 30
```

---

## 6. 模块接口概要

### 6.1 collectors/tencent.py

```python
class TencentSnapshot:
    def __init__(self, db: Database, config: Config)
    def run(self, stock_codes: list[str] | None = None) -> None
    """启动快照采集主循环，阻塞执行。需手动 Ctrl+C 停止。"""
    def stop(self) -> None
    """优雅停止（完成当前轮后退出）"""
```

### 6.2 collectors/baostock.py

```python
class BaostockCollector:
    def __init__(self, db: Database, config: Config)
    def backfill_daily(self, code: str, start_date: str, end_date: str) -> int
    def backfill_all_daily(self, start_date: str) -> None       # 断点续跑
    def fetch_industry(self) -> None                              # 申万一级行业
    def fetch_adjust_factors(self, code: str) -> None
    def fetch_trade_calendar(self) -> None
    def init_stock_basic(self) -> None                            # 初始化股票池
```

### 6.3 collectors/akshare.py

```python
class AkshareCollector:
    def __init__(self, db: Database, config: Config)
    def fetch_financial_summary(self, code: str) -> dict
    """按需拉取单只股票财务摘要，返回值全字段入库"""
    def fetch_performance_express(self, code: str) -> dict
    """按需拉取单只股票业绩快报"""
    def fetch_index_constituents(self, index_code: str) -> None
    """拉取指数成分股列表"""
```

### 6.4 validate.py

```python
def validate_daily_kline(
    db: Database,
    codes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    auto_fix: bool = False
) -> None
"""结果写入纯文本日志文件"""
```

---

## 7. 设计决策汇总

| 决策项 | 结论 |
|:---|:---|
| 周K/月K trade_date | 存周期首日（周一 / 1日） |
| 快照归档时机 | 每个交易日收盘后自动执行 |
| 股票池刷新 | 手动触发 |
| 校验报告格式 | 纯日志文件 |
| 股票代码格式 | `sh.600000` / `sz.000001`（存储层带前缀，UI 层去前缀） |
| 快照与 Baostock 分工 | 快照更新 18 个字段（见 4.1）；ps_ttm / pcf_ttm / is_st 仅由 Baostock 提供 |

---

## 8. 不在 V0.1 范围内

- 前复权/后复权 K 线的自动生成（复权因子已存，计算交由上层策略模块）
- 分钟线 / 5 分钟线
- 策略回测框架
- Web UI / 前端展示
- 自动交易 / 信号推送

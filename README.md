# 基于开放接口的共享单车停放热点准实时预测与调度优化系统

本项目是一个面向本科毕业设计、论文截图和本地答辩演示的 Streamlit Web 系统。系统使用山东公共数据开放平台梁山县哈啰共享单车订单接口作为数据来源，实现订单准实时采集、脱敏清洗、SQLite 增量存储、网格热点识别、短时预测、调度优化、GIS 可视化和简易 AI 助手。

## 运行方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

## 数据库自动创建

SQLite 数据库不需要手动创建。程序启动时会自动检测：

```text
data/database/bike_hotspot.db
```

如果文件不存在，系统会自动创建数据库文件和所需数据表。PyCharm 中可直接运行：

```text
init_database.py
```

用于单独初始化数据库；运行 Streamlit 页面时也会自动执行同样的初始化逻辑。

## 环境变量

复制 `.env.example` 为 `.env`，填写开放平台接口凭据：

```text
SD_CLIENT_ID=你的ClientId
SD_CLIENT_SECRET=你的ClientSecret
```

可选大模型助手配置：

```text
LLM_API_KEY=你的大模型密钥
LLM_BASE_URL=大模型接口地址
LLM_MODEL=模型名称
```

未配置大模型时，系统自动使用本地规则助手。

也可以直接编辑项目根目录的 `llm_config.toml`：

```toml
[llm]
api_key = "你的 Coding Plan 专属 API Key 或百炼 API Key"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen-plus"
```

说明：Coding Plan 应使用 `sk-sp-xxxxx` 格式的套餐专属 API Key；普通百炼 API Key 通常是 `sk-xxxxx`。AI 助手会动态读取该文件，未配置时自动回退到本地规则助手。

## 功能模块

- 数据采集：POST 调用开放接口，参数放在 URL 查询字符串中，支持手动采集最近三天、今日数据和演示型定时采集。
- 增量入库：以 `orderGuid` 为唯一键，重复订单自动跳过。
- 隐私保护：手机号不入库，`userNewId` 不保存原文，仅可在原始脱敏表中保存 SHA256 哈希。
- 数据清洗：过滤异常经纬度、无效距离和无效时长，提取日期、小时、星期和周末字段。
- 网格统计：默认使用 0.005 度网格，按网格小时统计借车量和还车量。
- 热点识别：按终点订单量分位数划分低、中、高热点。
- 短时预测：样本较少时使用历史均值模型，样本充足时训练随机森林回归模型。
- 气象与节假日融合：可采集 Open-Meteo 日尺度气象数据，并生成本地节假日日历，作为预测模型的外部特征。
- 调度优化：根据预测供需差识别过剩区和短缺区，使用贪心算法生成调度任务。
- GIS 可视化：使用 Folium 展示起终点热力图、热点网格、预测风险点、调度路线和梁山县研究区域边界。
- 梁山县边界：优先读取项目根目录 `ls.geoJson`，用于在地图中绘制真实行政区边界。
- 区域命名：根据热点网格中心点匹配附近地标，将经纬度网格转换为更容易理解的区域名称。
- AI 助手：解释热点、预测结果、调度方案，生成运营建议和论文分析段落。
- 用户管理：系统启动后必须登录，支持管理员、运营人员、普通用户三类角色；不同角色访问权限不同。

## 数据库

默认数据库文件：

```text
data/database/bike_hotspot.db
```

主要表：

- `raw_orders`：原始脱敏订单表。
- `clean_orders`：清洗后订单表。
- `grid_hour_stats`：网格小时统计表。
- `hotspot_results`：热点识别结果表。
- `prediction_results`：预测结果表。
- `dispatch_plans`：调度方案表。
- `crawl_logs`：采集日志表。
- `model_logs`：模型训练日志表。
- `weather_daily`：气象日数据表。
- `holiday_calendar`：节假日日历表。
- `system_users`：本地系统用户表。

## 默认用户与权限

首次启动时会自动创建默认管理员：

```text
用户名：admin
密码：admin123
```

角色权限：

- 管理员：可访问全部页面，包括用户管理。
- 运营人员：可访问数据采集、热点预测、调度优化、报表、AI 助手等运营页面。
- 普通用户：可查看数据管理、热点监测、统计报表和 AI 助手，不允许执行采集、训练、调度生成和用户管理。

系统使用 SQLAlchemy 创建连接，后续可通过修改 `DATABASE_URL` 预留切换 MySQL 的可能。

## API 说明

接口地址：

```text
http://data.sd.gov.cn/gateway/api/1/publicBikeOrder/ls
```

请求方式：`POST`

参数放在 URL 查询字符串：

```text
pageNo=1&startDate=2026-05-04 00:00:00&endDate=2026-05-06 00:00:00
```

请求头：

- `X-Client-Id`
- `X-Timestamp`
- `X-Nonce`
- `X-Signature`

签名规则：

```text
Base64(HmacSHA256(ClientSecret, X-Client-Id + X-Timestamp + X-Nonce))
```

接口外层 JSON 的 `data` 字段中包含字符串形式的内层 JSON，系统会自动解码并从内层 `data -> orderList` 提取订单。

## 论文可用描述

本系统采用本地 Web 化方式实现，基于 Streamlit 构建交互式界面，使用 SQLite 数据库存储共享单车订单数据。系统通过山东公共数据开放平台接口获取梁山县哈啰共享单车订单数据，在数据入库前对手机号和用户标识等敏感字段进行脱敏处理。系统支持订单数据的增量更新，能够随着接口持续调用不断积累历史样本。在此基础上，系统以网格小时为分析单元，完成停放热点识别、短时需求预测和车辆调度方案生成，并通过 GIS 地图进行可视化展示。该系统主要面向共享单车短时运维调度场景，可为县域共享单车停放治理提供辅助决策参考。

## 演示建议

1. 在“数据采集”页面采集最近三天数据，确认新增订单数和重复订单数。
2. 在“数据管理”页面展示脱敏后的订单和字段说明。
3. 在“热点监测”页面展示终点热力图和热点排行榜。
4. 在“热点预测”页面重新训练模型并展示预测地图。
5. 在“调度优化”页面生成调度方案和路线图。
6. 在“AI 助手”页面生成热点解释、运营建议或论文分析段落。

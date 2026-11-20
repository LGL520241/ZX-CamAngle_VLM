# CamAngle 摄像头角度巡检系统

面向「定时抓图 → 自动判断摄像头角度/画面是否合格 → 低置信度转人工复核」的一体化 Web 系统。

摄像头长期运行后会出现两类问题：一是物理姿态变化（被撞歪、支架松动导致倾斜或移位），二是画面质量异常（遮挡、全黑、过曝、模糊、对墙或对天花板）。靠人工盯几千路摄像头不现实，本系统负责自动抓图、自动判定，只把低置信度和边界情况推给人工，形成可解释、可闭环的巡检流程。

当前仓库提供可运行的 MVP：演示模式开箱即用，接入海康云眸开放平台后即可跑真实抓图。判定层是可解释的规则引擎（不是深度学习模型），适合先跑通业务流程。

## 功能特性

- 海康云眸接口封装：获取 token、设备/通道列表、远程抓图、图片下载。
- 四层自动判定：画面质量、旋转/倾斜、基准图比对、SOP 基础规则。
- 决策输出：合格 / 不合格 / 待复核 / 无信号，均带原因与置信度。
- 人工复核：当前图与基准图并排查看，支持「合格 / 不合格 / 合格并设为新基准」。
- 可视化 Web：总览看板、相机列表、抓图记录、人工复核队列、阈值设置。

## 整体架构

```text
海康云眸（或演示模式）
   │  1. 发现设备与通道
   ▼
Pipeline 抓图调度（并发控制 + 同通道限流）
   │  2. 远程抓图、下载落盘、生成缩略图
   ▼
AnalysisEngine 四层判定
   │  3. 质量 → 旋转 → 基准比对 → SOP
   ▼
决策融合（合格/不合格/待复核/无信号 + 置信度 + 原因）
   │  4. 写入 SQLite
   ▼
Web 看板 + 人工复核队列（可设新基准图，形成闭环）
```

## 判定算法说明

判定引擎在 `cam_angle/analysis.py`，由四个独立模块组成，最后融合出结论。

**第一层：画面质量** —— 转灰度后计算亮度均值、灰度标准差、暗/亮像素占比、拉普拉斯清晰度方差、Canny 边缘密度，按级联顺序判断：

- 全黑/过暗（暗像素 ≥95% 或均值 <10）→ 不合格；
- 全白/过曝（亮像素 ≥95% 或均值 >245）→ 不合格；
- 模糊（拉普拉斯方差 <60）→ 不合格；
- 低纹理（标准差 <18 或边缘密度 <0.004）→ 警告，疑似对墙/天花板；
- 其余 → 画面可用。

**第二层：旋转/倾斜** —— Canny 提取边缘后用 HoughLinesP 检测直线，把直线分成「接近水平」和「接近垂直」两组，取数量占优的一组用中位数估计滚转角，避免离群直线干扰。倾斜 >20° 判不合格，>8° 判警告。

**第三层：基准图比对** —— 先用全局灰度相关系数快速排除大面积场景变化（<0.7 判移位），再用 ORB 特征 + 比值测试 + RANSAC 单应矩阵做几何对齐，从单应矩阵解出内点率、平移量、旋转角：内点率 <35%、平移 >18%、旋转 >18° 任一超阈值判不合格。无基准图时该层返回 unknown。

**第四层：SOP 规则** —— 按门店模板（default / indoor / outdoor）读取阈值，当前只实现「大面积纯色/低纹理」基础规则，业务 SOP 需按门店模板扩展。

**决策融合** —— 有 fail 看最高置信度（≥0.78 判不合格，否则转复核）；有 unknown 一律转复核；有 warn 且置信度 <0.98 转复核；全通过判合格。低置信度统一转人工，是「漏报代价高于误报」的巡检场景下的有意设计。

关键阈值默认值（见 `cam_angle/config.py`，均为经验值，需用真实图片重新标定）：

| 阈值 | 默认值 | 含义 |
|---|---|---|
| `dark_ratio` / `bright_ratio` | 0.95 | 全黑/全白判定占比 |
| `blur_var` | 60.0 | 拉普拉斯清晰度方差下限 |
| `std_low` / `edge_density_low` | 18.0 / 0.004 | 低纹理判定 |
| `roll_fail_deg` / `roll_review_deg` | 20.0 / 8.0 | 倾斜不合格/转复核角度 |
| `baseline_min_good` | 8 | 基准比对最少匹配点数 |
| `baseline_inlier_ratio` | 0.35 | 单应矩阵内点率下限 |
| `baseline_translation` | 0.18 | 相对图宽的平移上限 |
| `baseline_rotation_deg` | 18.0 | 基准比对旋转上限 |
| `confidence_review` | 0.78 | 低置信度转人工阈值 |

## 目录结构

```text
CamAngle/
  cam_angle/
    main.py        FastAPI 入口与 REST 接口
    pipeline.py    相机发现、抓图调度、分析入库
    analysis.py    质量/旋转/基准/SOP/决策融合算法
    hikvision.py   海康云眸开放平台客户端
    database.py    SQLite 存储
    demo.py        演示数据生成
    config.py      配置与阈值加载
  static/          前端页面（原生 HTML/JS/CSS）
  run.py           启动脚本（自动挑选空闲端口）
  requirements.txt
  .env.example     配置模板
```

## 快速开始

```powershell
# 1. 创建独立环境（conda 网络受限时，可直接使用已有 Python 3.9+ 环境）
conda create -n CamAngle python=3.10 -y
conda activate CamAngle
pip install -r requirements.txt

# 2. 启动（默认演示模式）
python run.py
```

`run.py` 会在 `20350–20399` 之间自动挑选一个空闲端口，启动后会打印实际访问地址（形如 `http://127.0.0.1:2035x`），以终端输出为准。

演示模式会自动生成 12 路相机，覆盖正常、全黑、全白、模糊、对天花板、旋转、移位、无基准图等样例，并在总览页展示判定分布。

## 接入海康云眸（真实模式）

复制 `.env.example` 为 `.env`，填写真实参数：

```env
DEMO_MODE=false
HIK_CLIENT_ID=你的client_id
HIK_CLIENT_SECRET=你的client_secret
HIK_BASE_URL=https://api2.hik-cloud.com
HIK_GROUP_NOS=1
```

> `.env` 包含密钥，已被 `.gitignore` 忽略，切勿提交到版本库。

真实模式的关键约束（来自官方文档）：

- 同一通道抓图间隔不低于 5 秒，频繁调用会进入限流黑名单。
- 同一 NVR 多通道并发会共享带宽，影响成功率。
- 返回的 `picUrl` 有效期 2 小时，系统会立即下载落盘。
- 抓图仅支持 IPC 或关联 IPC 的 NVR 通道，部分海康设备可能不支持。

## 重要：先建基准图

基准比对是判断「摄像头是否被移动」的核心能力，但新建相机**没有基准图**时，该层会返回 unknown，融合层会把 unknown 一律转复核。也就是说，如果不先为每路相机建立基准图，绝大部分抓图都会被归入「待复核」，系统无法真正自动判定移位。

初始化建议：在确认摄像头角度正确的状态下，先抓一张标准图入库，或通过人工复核队列的「合格并设为基准」逐路建立基准图。

## REST 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/stats` | 总览统计 |
| GET | `/api/cameras` | 相机列表 |
| GET | `/api/cameras/{id}/baseline` | 相机基准图 |
| GET | `/api/captures` | 抓图记录，支持 `decision` 过滤 |
| GET | `/api/captures/{id}` | 单条判定详情 |
| GET | `/api/captures/{id}/image` | 当前抓图原图 |
| GET | `/api/captures/{id}/thumb` | 当前抓图缩略图 |
| POST | `/api/captures/{id}/review` | 提交人工复核 |
| POST | `/api/cycle/start` | 启动一轮巡检 |
| GET | `/api/cycle/status` | 巡检进度 |
| GET | `/api/cycle/progress` | 巡检进度（含速度/ETA/进行中任务） |
| GET/PUT | `/api/settings` | 读取/更新判定阈值 |
| GET | `/api/health` | 健康检查 |

## 配置项

完整配置见 `.env.example`，常用项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEMO_MODE` | `true` | 演示模式开关 |
| `HIK_CLIENT_ID` / `HIK_CLIENT_SECRET` | 空 | 海康云眸凭证 |
| `HIK_BASE_URL` | `https://api2.hik-cloud.com` | 云眸接口地址 |
| `HIK_GROUP_NOS` | `1` | 门店分组编号 |
| `REGION_FILTER` | 空 | 按门店名过滤 |
| `SAMPLE_SIZE` | `0` | 每轮随机抽检数量，0 表示全量 |
| `DATA_DIR` | `./data` | 本地数据目录 |
| `DB_PATH` | `./data/cam_angle.db` | SQLite 数据库路径 |
| `CONCURRENCY` | `4` | 抓图并发数 |
| `CAPTURE_INTERVAL` | `5.0` | 同通道抓图最小间隔（秒） |

## 数据存储

SQLite 存五类数据：`cameras`（相机）、`baselines`（基准图）、`captures`（抓图与判定结果）、`reviews`（人工复核）、`settings`/`cycle_status`（配置与巡检状态）。图片文件落在 `data/images`、`data/thumbs`、`data/baselines` 下，整个 `data/` 目录已被 `.gitignore` 忽略。

## 注意事项与生产化路线

- 当前阈值为经验值，必须用真实图片重新标定。
- 旋转检测假设场景主直线水平/垂直，倾斜货架等场景可能误判。
- ORB 特征在低纹理、夜间场景下鲁棒性有限。
- SOP 层目前是基础占位，需按门店模板扩展。
- SQLite 与本地文件存储适合 MVP，2 万路规模建议替换为 PostgreSQL + 对象存储。
- 抓图并发、限流、失败重试需按云眸账号实际配额压测。

生产化建议：基准比对升级为 SuperPoint + LightGlue + DINOv2 全局特征，SOP 层接入 YOLO 目标检测与语义分割，决策层用带置信度校准的分类器，并用人工复核数据做主动学习闭环。

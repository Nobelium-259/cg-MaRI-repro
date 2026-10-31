# MaRI 推理复现与补充实验代码

本仓库整理 CVPR 2025 论文 **MaRI: Material Retrieval Integration across Domains** 的推理适配、检索界面和补充实验代码。它是课程复现项目，作者官方实现见 [MaRI 官方仓库](https://github.com/jianhuiwemi/Material-Retrieval-Integration-across-Domains)，论文见 [arXiv](https://arxiv.org/abs/2503.08111)。

**仓库只发布代码、运行说明和测试素材的来源元数据。** 离线回放、原始日志、图片数据、模型权重和虚拟环境均保留在本地，不随本仓库上传。测试素材由下载工具按需准备。




## 已完成的复现范围

- 官方 MaRI 权重校验、双编码器严格加载、CPU／CUDA推理入口与Top-k检索。
- 同规格 DINOv2 基线与颜色／梯度直方图教学基线；没有模型失败后的静默替换。
- Streamlit 内置查询和图片上传，模型切换与候选缓存管理。
- 13项核心边界测试；固定输入条件、重复计时和显存分配记录工具。
- 12项公开CC0材质的平面纹理到材质球检索，按同一资产ID确定相关项。

没有完成原论文全量评测或两阶段训练。早先12项素材是平面纹理；新版另提供96组官方真实照片配对、295候选的冻结诊断配方。官方配对可能参与权重训练，不能称为独立留出测试，也不能称为论文的Trained或Unseen集合。余弦分数不是材质成分概率。

## 目录

```text
source/course_demo/        推理适配、界面、测试和输入实验
source/dinov2/             本地DINOv2代码快照及许可证，不含权重
source/MaRI_official/      作者推理脚本节选与原始许可证
tools/prepare_demo_data.py 公共CC0演示数据准备工具
data/public_demo_sources.json  固定12项资产的URL、ID和校验值
THIRD_PARTY.md             来源与许可证说明
```

作者完整的训练和数据构建代码请参阅官方仓库。本项目实际运行入口是`source/course_demo`。本地交接训练快照存在未验证及截断内容，没有纳入本次代码发布。

## 安装环境

已验证的运行组合为 Python 3.11.7、PyTorch 2.2.1、torchvision 0.17.1、CUDA 11.8、NumPy 1.26.4、Pillow 10.2.0、Streamlit 1.30.0。设备采用单卡10GB显存、FP32、batch=1；这不是最低硬件要求。新机器与其他版本组合需另行验证。

Windows PowerShell示例：

```powershell
git clone https://github.com/Nobelium-259/cg-MaRI-repro.git
cd cg-MaRI-repro
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python.exe -m pip install -r source\course_demo\requirements.txt
```

PyTorch安装命令来自[官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v221)。CPU机器可按该页面选择CPU构建；教学基线只需要NumPy和Pillow。Linux可使用同一Python代码，将解释器路径换为`.venv/bin/python`。

已有项目环境可直接复用。`setup_windows.bat`会保留可用环境；其全新安装备用组合是torch 2.5.1／torchvision 0.20.1／cu121，与上述实测组合不同，未作为本项目已验证环境。

## 准备公共演示数据

```powershell
.venv\Scripts\python.exe tools\prepare_demo_data.py
```

工具从ambientCG官方资源下载固定12项材质的预览和颜色贴图，网络传输约16MB，解码后本地图片约71MB。数据受[CC0许可](https://docs.ambientcg.com/license/)覆盖；来源、下载哈希与实际记录均保留。

文件放入`data/public_demo/`，首次会生成`data/demo_manifest.json`。如果已有默认清单，工具会保留它；可显式使用`--manifest data/public_demo/manifest.json`。现存图片不匹配时会报错，不覆盖用户文件。

离线复用相同12项已校验PNG：

```powershell
.venv\Scripts\python.exe tools\prepare_demo_data.py --from-local <已有资产图片目录>
```

## 先运行无需权重的基线

```powershell
.venv\Scripts\python.exe source\course_demo\run.py --mode baseline --manifest data\public_demo\manifest.json --output results\baseline.json
```

12项公共素材的既有实测如下，仅描述这套补充任务：

| 方法 | 首位命中 | 前5命中 | mAP |
|---|---|---|---|
| 颜色纹理基线 | 4/12 | 11/12 | 0.565 |
| DINOv2 ViT-L/14 | 10/12 | 12/12 | 0.917 |
| MaRI官方权重 | 11/12 | 12/12 | 0.958 |

MaRI与DINOv2仅相差一个查询，两者都会混淆一对相近砖材质。小集合结果不能证明总体优势，也不能与原论文完整指标直接比较。

## 准备权重与神经网络推理

MaRI官方权重：[epoch_25.pth下载页](https://drive.google.com/file/d/1rlHUrPDShA_w1OmJF8Eeo-SMX_-CJaGD/view)。下载后存为`weights/mari.pth`，约2.6GB。本仓库不包含或重新分发权重。

固定快照SHA256：

```text
6d7bf5bee85ab3e113ad729f7dfe374285e818693da24e8276c72dd0bd64fb0f
```

也可运行`download_weights.bat`或`source/course_demo/download_weights.py`。下载脚本不覆盖已有权重；下载失败时可从官方页面人工获取。MaRI适配会校验上述哈希，以`weights_only=True`读取，并严格检查两个分支。

```powershell
.venv\Scripts\python.exe source\course_demo\run.py --mode mari --device cuda --manifest data\public_demo\manifest.json --output results\mari.json
.venv\Scripts\python.exe source\course_demo\run.py --mode dinov2 --device cuda --manifest data\public_demo\manifest.json --output results\dinov2.json
```

CPU运行将`--device cuda`改为`--device cpu`。DINOv2基线首次会下载Meta预训练权重至本地缓存；MaRI权重与DINOv2基础权重不同，不能相互替代。MaRI完整任务权重加载不需要另行下载DINOv2初始化权重。

## 实时界面与测试

先准备默认演示清单，再运行：

```powershell
.venv\Scripts\python.exe -m streamlit run source\course_demo\app.py --server.address 127.0.0.1
.venv\Scripts\python.exe source\course_demo\test_core.py
.venv\Scripts\python.exe source\course_demo\test_app.py
```

界面中选择模型、设备和清单；MaRI需要先放置官方权重。`test_app.py`对当前清单逐项执行教学基线，并使用明确不存在的临时权重路径检查报错，已有MaRI权重也不会改变该失败路径。

## 输入条件与重复计时

```powershell
.venv\Scripts\python.exe source\course_demo\supplement_experiments.py --run-id public_cpu_check --manifest data\public_demo\manifest.json --modes baseline --repeats 5
```

准备神经网络权重后，省略`--modes baseline`可运行全部三种方法，默认神经网络使用CUDA。输出目录必须是新的运行ID，旧记录不覆盖。

工具固定12种输入条件，记录每次排名、参数和来源哈希。公共原图分辨率较高，首次会生成数百MB派生图片，应预留本地磁盘空间；扩边掩码只去除人工添加的背景，不等同于物体分割。派生图片不增加独立样本数。计时按实际查询数计算：12查询×30轮为每模型360次，6查询×30轮为180次。CUDA计时显式同步，包含读取、预处理、传输、前向、归一化与排序，模型加载和建库另计。

## 来源与许可

MaRI和DINOv2上游许可证随代码保留，详见[THIRD_PARTY.md](THIRD_PARTY.md)。本仓库未为课程适配代码另行新增开源许可证，不将上游代码或模型权重统一重授权。报告与PPT仍仅保留在本地。

## 新版 MaRI 与 DINOv2 同屏对照

`launch_demo.bat` 现在打开新版比较界面：默认回放同一查询、同一候选库的两个模型结果，支持按 MaRI 单独命中、DINOv2 单独命中和共同失败筛选；全量统计始终保留。原单模型界面改由 `launch_single_demo.bat` 打开。

代码仓库不附带照片、结果或特征库。请先准备权重，按冻结配方获取子集，再依次计算三种方法并汇总：

```powershell
.venv\Scripts\python.exe tools\prepare_comparison_data.py
.venv\Scripts\python.exe source\course_demo\compare_models.py --mode baseline --run-id my_comparison
.venv\Scripts\python.exe source\course_demo\compare_models.py --mode dinov2 --run-id my_comparison --device cpu --threads 2
.venv\Scripts\python.exe source\course_demo\compare_models.py --mode mari --run-id my_comparison --device cpu --threads 2
.venv\Scripts\python.exe source\course_demo\compare_models.py --run-id my_comparison --summarize
.\launch_demo.bat
```

数据获取仅分段读取作者约21GB的Real-world2.zip，不下载整包；解码后子集约406MiB。配方固定在 `source/course_demo/comparison_recipe.json`，包含来源提交、ZIP条目偏移和CRC、素材ID及公开预览URL。96组真实配对按固定哈希规则选取，199项ambientCG预览是干扰项；一个持续404的资产已在推理前冻结剔除，没有替换。照片与掩码尺寸可不同，允许最多0.1%的比例取整偏差。

三个协议分别为真实照片加掩码、同批照片不加掩码，以及199项基础颜色贴图到材质球。后者不是物体照片。官方配对与权重训练数据的交叉未知；相关性来自原配对或资产ID，未完整重标所有外观近似材质。不得把结果写成独立泛化或完整论文指标。

评测拒绝覆盖已有模式结果，另存完整排名及可复用特征库。新版实时上传使用CPU，两模型依次加载并释放，不启动CUDA。照片上传没有人工标签，仅作定性检索。旧入口保留CUDA选项；本机出现重启后，本轮未继续CUDA负载测试。

新增校验（需完成上述数据及结果准备）：

```powershell
.venv\Scripts\python.exe source\course_demo\test_comparison.py
```

该检查重算全部排名和指标，核对成功/失败分组，并验证缓存损坏及缺权重时明确失败。

# 代码来源与许可证范围

## MaRI

- 官方仓库：https://github.com/jianhuiwemi/Material-Retrieval-Integration-across-Domains
- 本地快照依据提交536154fc7396b4f12f725885d53017f887ace8db。
- 发布节选位于source/MaRI_official，保留其MIT许可证与作者署名。完整数据构建代码请参阅上游仓库。
- source/course_demo是课程推理适配，与作者原始训练脚本区分；没有执行完整训练。

## DINOv2

- 上游：https://github.com/facebookresearch/dinov2
- 本地快照依据提交7764ea0f912e53c92e82eb78a2a1631e92725fc8。
- source/dinov2保留原代码和许可证文件。DINOv2主许可为Apache-2.0；嵌入组件及其他模型对应的专用许可也按原文件保留，不统一重授权。
- 模型权重不包含在本仓库，获取与使用应遵循其原发布条款。

## 公共演示素材

ambientCG素材及材质预览由其官方声明采用CC0 1.0：https://docs.ambientcg.com/license/ 。本仓库发布来源元数据和下载工具；demo/DEMO-MaRI.html另含已授权发布的演示缩略图，不包含独立原图文件。资产ID及校验值见data/public_demo_sources.json。

## 本项目适配代码

未为课程原创适配代码另行新增开源许可证。上游许可证适用于相应上游代码，不能据此认定整个仓库、模型权重或本地课程文档适用同一许可。


## 官方真实配对诊断子集

冻结配方来自作者公开的 Hugging Face 数据集：
https://huggingface.co/datasets/jianhuiwemi/MaRI ，提交 d9018eca4a7833c1e3c5e1867b7418aceff3faec。
只按HTTP Range读取Real-world2.zip的96组照片、掩码及外观迁移材质球。单条目按ZIP大小/CRC及SHA256校验，没有计算整个21GB归档的SHA256。原图、掩码、生成球、完整结果和特征库保留本地；除取样及获取配方外，demo/DEMO-MaRI.html按用户要求包含用于回放的缩略图及前五排名。使用原始数据时应遵循作者数据源的条款。

新增199项ambientCG预览继续使用官方CC0资源；配方包含素材ID与来源URL。一个失效预览在推理前剔除且不替换。官方配对训练重叠未知，不能据此宣称独立泛化。

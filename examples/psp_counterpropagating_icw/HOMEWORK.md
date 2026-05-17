# 本科生第十一讲作业：HybridVPIC 模拟对向传播离子回旋波

## 题目

根据第十一讲讲义第二部分“模拟案例-1：近日太阳风中双向传播离子回旋波（ICWs）与质子散射的动力学混合模拟”，在 HybridVPIC 框架下运行或分析一维 core+beam 双质子混合模拟，理解温度各向异性如何激发对向传播 ICWs，以及 ICWs 如何通过回旋共振散射质子。

## 作业仓库与下载方式

本作业所需的完整 HybridVPIC 仓库在 GitHub：

```text
https://github.com/PKU-Heliosphere-org/HybridVPIC
```

学生可以用 `git clone` 下载完整仓库：

```bash
git clone https://github.com/PKU-Heliosphere-org/HybridVPIC.git
cd HybridVPIC
```

本作业对应的案例目录为：

```bash
cd examples/psp_counterpropagating_icw
```

后续所有编译、运行、参数复算和绘图命令，如果没有特别说明，都在这个目录下执行。不要只下载单个 `.cxx` 文件；HybridVPIC 的编译和运行依赖仓库中的 `bin/`、`src/`、`config/`、`scripts/` 等完整目录结构。

## 物理模型

模拟采用一维周期盒：

- 背景磁场：\(\boldsymbol B_0=B_0\hat{\boldsymbol z}\)
- 空间方向：只沿 \(z\) 方向展开
- 离子：核质子 `ion_c` 与束流质子 `ion_b`，均为漂移双麦氏分布
- 电子：无惯性流体，满足准中性
- 边界条件：场和粒子均为周期边界

核心自由能来自：

1. 核质子与束流质子的相对漂移；
2. 两个质子成分的垂直温度各向异性 \(T_\perp>T_\parallel\)。

## 目录内容

- `psp_counterpropagating_icw.cxx`：HybridVPIC 输入 deck
- `config/psp_case1_parameters.json`：讲义参数与归一化参数
- `scripts/derive_parameters.py`：从物理量复算归一化参数
- `scripts/visualize_counterpropagating_icw.py`：生成图2、图3、图4风格可视化
- `scripts/convert_particle_dump_to_csv.py`：将 `particle/T.*/ion_c` 和 `particle/T.*/ion_b` 转为 `x,y,z,ux,uy,uz,w` CSV
- `scripts/plot_particle_vdf.py`：粒子快照转为 CSV 后绘制真实 VDF
- `translate_psp_icw.f90`：将 VPIC 原生输出翻译为 `data/*.gda`
- `run_example.sh`：编译、运行和演示图生成脚本
- `README.md`：使用说明

## 基础任务

1. 阅读 `README.md` 与 `config/psp_case1_parameters.json`，说明为什么本算例适合用 1D 沿磁场方向的模拟。
2. 运行参数复算脚本：

   ```bash
   python3 scripts/derive_parameters.py
   ```

   在报告中列出 \(v_A\)、两个质子成分的密度占比、漂移速度 \(U_c/v_A\)、\(U_b/v_A\)、\(\beta_{\parallel c}\)、\(\beta_{\parallel b}\)。

3. 编译并运行模拟。若本机没有 HybridVPIC 编译器，可只完成 demo 可视化与参数分析。

   ```bash
   make PROJECTDIR=/path/to/hybridVPIC/bin
   ./run_example.sh run 1
   ```

4. 运行场数据翻译，生成后处理所需的 `Bx.gda`、`By.gda`、`Bz.gda`：

   ```bash
   make translate
   ./run_example.sh translate
   ```

5. 若已有翻译后的 `Bx.gda`、`By.gda`、`Bz.gda`，运行：

   ```bash
   python3 scripts/visualize_counterpropagating_icw.py --data data --out figures
   ```

   若暂时没有真实输出，先运行：

   ```bash
   python3 scripts/visualize_counterpropagating_icw.py --demo --out figures
   ```

6. 若要用真实粒子数据绘制图4风格 VDF，先转换 `particle/` 快照，再绘图：

   ```bash
   ./run_example.sh particle-csv 0 200000
   ./run_example.sh particle-vdf 0
   ```

## 分析任务

报告中至少回答以下问题：

1. 图2风格结果中，\(\langle \delta B_\perp^2\rangle\) 是否经历线性增长和饱和？饱和时间约为多少？
2. \(\omega-k\) 图或 \(k\) 谱中是否同时存在 \(k>0\) 与 \(k<0\) 的增强？这如何对应对向传播 ICWs？
3. 虚拟航天器图中，为什么等离子体参考系中的对向传播波会在航天器参考系表现为两个频带？
4. 图4风格的速度空间示意中，哪一支波主要散射核质子，哪一支波主要散射束流质子？
5. 如果把 \(T_\perp\) 的 1.4 倍放大去掉，你预期增长率和饱和振幅如何变化？说明理由。

## 数据与诊断说明

本作业输出分三层：

1. `fields/`：VPIC 原生场数据，包含 `Bx, By, Bz, Ex, Ey, Ez` 等，支持图2和图3。
2. `hydro/`：`ion_c` 与 `ion_b` 的密度、电流和应力张量，支持温度各向异性分析。
3. `particle/`：稀疏粒子快照，支持真实 VDF。默认只在 \(t=0\)、\(500\Omega_p^{-1}\)、\(700\Omega_p^{-1}\) 附近输出，避免数据量过大。

`translate_psp_icw.f90` 会把 `fields/` 和 `ion_c` 的部分 `hydro/` 翻译为 `data/*.gda`。图2和图3使用 `data/Bx.gda`、`data/By.gda`、`data/Bz.gda`；图4若要使用真实粒子数据，需要先用 `scripts/convert_particle_dump_to_csv.py` 把 `particle/` 快照转换为 CSV 的 `x,y,z,ux,uy,uz,w` 列，再用 `scripts/plot_particle_vdf.py` 绘制。

## 拓展任务

任选一项：

- 将 `nppc` 从 512、2048、8192 逐步增加，比较噪声水平对波增长曲线的影响。
- 将 `A_core` 或 `A_beam` 降低 30%，比较哪一支对向波更敏感。
- 改变虚拟航天器速度 `VSW_VA`，观察时频图中两个频带的位置如何变化。
- 对 `ion_c` 和 `ion_b` 的 hydro 输出分别计算温度各向异性随时间的变化。

## 提交内容

提交一个压缩包，包含：

- 一份 PDF 或 Markdown 报告；
- 关键参数表；
- 至少三张图：波能增长/谱图、虚拟航天器时频图、速度空间散射示意或真实 VDF 图；
- 修改过的代码或脚本；
- 简短说明：运行环境、编译命令、是否使用真实模拟输出或 demo 数据。


<!--
## 评分建议

- 物理模型与参数解释：30%
- HybridVPIC 设置理解：20%
- 结果图与诊断方法：30%
- 讨论与拓展分析：15%
- 报告规范和可复现性：5%
-->

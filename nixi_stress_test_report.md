# 尼西村全景实验复盘：NUFROST 端点预测、雪山状态与历史状态选择

日期：2026-06-06  
实验位置：尼西村附近 Sentinel-2 全景  
中心坐标：`lon=94.2605, lat=29.7733`  
重点异常点：`lon=94.27484, lat=29.79822`  
目标时间：`20260206T041839`  
实验分支：`.worktrees/frequency-neighborhood`

## 1. 实验背景

这一轮实验最初的目标是：

> 在 worktree 下自由尝试各种方案，尽量把尼西村全景 RMSE 降到 100 左右。

尼西村场景被选作压力测试，因为这里有明显雪山、阴影、地形照度变化和端点预测问题。目标时间 `2026-02-06` 位于当前可用 Sentinel-2 时间序列的末端附近，因此它不是普通的时序插值问题，而更接近：

```text
历史不规则观测 + 目标时间接近末端 + 雪山状态快速变化
```

从实验结果看，尼西村是一个非常极限的 case。它暴露了 NUFROST 在端点重构、雪山状态切换和局部异常光谱方面的限制，但不适合作为唯一优化目标。

## 2. 当前 NUFROST 方法简述

当前 Rust 版本里的 NUFROST 已经不是旧的单 band 独立拟合，而是向量化多光谱路径：

1. 对每个像元构造多波段观测：

   ```math
   \mathbf y_i \in \mathbb R^B
   ```

   Sentinel-2 当前使用六个 band：

   ```text
   B2, B3, B4, B8, B11, B12
   ```

2. 先做 robust band-wise 标准化：

   ```math
   z_{i,b}
   =
   \frac{y_{i,b}-\mu_b}{s_b+\epsilon}
   ```

   其中：

   ```math
   \mu_b = \operatorname{median}(y_{\cdot,b})
   ```

   ```math
   s_b =
   1.4826
   \operatorname{median}
   \left(
   |y_{\cdot,b}-\mu_b|
   \right)
   ```

3. 用向量 NUFFT 估计共享频率：

   ```math
   F_b(f_k)
   =
   \sum_i
   z_{i,b}
   \exp(-j2\pi f_k t_i)
   ```

   然后组合成联合功率：

   ```math
   P(f_k)
   =
   \sum_{b=1}^{B}
   |F_b(f_k)|^2
   ```

4. 根据共享频率构造 harmonic/trend 设计矩阵：

   ```math
   X =
   [
   1,\,
   t,\,
   \cos(2\pi f_1t),\,
   \sin(2\pi f_1t),\,
   \dots
   ]
   ```

5. 用 date-level vector Huber-IRLS + multi-output Ridge 拟合：

   ```math
   \min_{\Theta}
   \frac12
   \left\|
   W^{1/2}
   (Z-X\Theta)
   \right\|_F^2
   +
   \frac12
   \left\|
   \Lambda^{1/2}
   \Theta
   \right\|_F^2
   ```

   固定权重时，法方程为：

   ```math
   (X^\top W X+\Lambda)\Theta
   =
   X^\top WZ
   ```

   这里一次求解所有 band 的系数：

   ```math
   \Theta \in \mathbb R^{P\times B}
   ```

当前方法的主要跨 band 耦合来自：

```text
共享频率 + 共享日期权重 + 共享设计矩阵 + 多输出求解
```

但它还不是强约束的跨 band 光谱状态模型，因为没有显式约束：

```math
\Theta H
```

或低秩结构：

```math
\operatorname{rank}(\Theta)\le K
```

## 3. 原始基线结果

尼西村全景目标时间 `20260206T041839` 上，已有主要结果如下：

| 方法 | 全景 RMSE | 说明 |
|---|---:|---|
| NUFROST 当前主线 | `247.03` | vector NUFFT + vector Huber-Ridge |
| HANTS | `284.41` | 对比方法 |
| Zhu2015 | `237.32` | 对比方法 |
| band-wise local linear | `165.68` | 非论文方法，仅诊断用 |
| clean-idw3 full | `218.61` | joint valid dates + clean IDW |
| clean-linear full | `238.68` | joint valid dates + clean linear |

其中最值得注意的是：

```text
band-wise local linear 全景 RMSE = 165.68
```

这个方法不是 NUFROST，只是诊断用 baseline。它对每个 band 独立做最近时间线性插值：

```math
\hat y_b(t_*)
=
(1-\alpha)y_b(t_0)
+
\alpha y_b(t_1)
```

由于它每个 band 独立找有效观测，因此可能拼出一个真实世界中并不存在的六维光谱向量。这和“多光谱是高维向量轨迹”的设计思想不一致。

但是它比 NUFROST 主线 RMSE 更低，说明尼西村这个目标日期上，最近观测包含很强信息，而 harmonic 外推会在一些区域产生更大误差。

## 4. 初始假设：彩色噪点来自六维污染

用户提出了一个重要判断：

> 我们有六个维度，云、雪、影的变化都是六个维度的高幅度变化。

因此第一批实验尝试把污染定义为六维向量的高幅度突变：

```math
e_i
=
\left\|
\mathbf z_i
-
\tilde{\mathbf z}_i
\right\|_2
```

其中：

```math
\tilde{\mathbf z}_i
```

是时间邻域内的多光谱中位向量。

在代码中形成了几类实验模式：

```text
clean-linear
clean-idw3
clean-idw5
clean-idw3-cost
clean-idw5-cost
clean-idw3-hard
clean-idw5-hard
```

这些模式的共同点是：先把每个日期看成一个六维观测向量，再计算日期级污染分数。

## 5. 六维高幅度污染假设的失败

实验结果表明：

```text
六维高幅度变化 != 污染
```

原因是雪山真实变化本身也会造成六维高幅度变化。

例如：

```text
积雪扩张
积雪消融
山体阴影移动
坡向导致的照度变化
雪/岩石混合像元变化
```

这些都可能在六个 band 上同时产生大幅变化。

因此，如果只根据：

```math
\left\|
\mathbf z_i-\tilde{\mathbf z}_i
\right\|_2
```

判断污染，会把真实地表变化误判为异常。

典型失败结果：

| 实验模式 | 场景 | RMSE | 结论 |
|---|---|---:|---|
| clean-idw3 full | 尼西村全景 | `218.61` | 比 NUFROST 好，但远不到 100 |
| clean-linear full | 尼西村全景 | `238.68` | 接近原始 NUFROST |
| clean-idw3-hard | 64x64 试验 | 极差 | hard reject 会误杀真实变化 |
| clean-idw3-cost | 64x64 试验 | 极差 | 用污染分数排序候选不稳定 |

因此，污染判别不能只看幅度，还必须区分：

```text
真实地表状态变化
vs.
云/雪/影/大气污染造成的观测异常
```

这需要引入光谱形状、状态连续性或外部质量信息。

## 6. 光谱方向比幅度更重要

对重点像元 `lon=94.27484, lat=29.79822` 做 pixel-bench 后发现：

NUFROST 预测：

```text
[1585.40, 1537.12, 1782.27, 1948.87, 2672.82, 2163.75]
```

目标真值：

```text
[1293.00, 1307.00, 1424.00, 1978.50, 3134.00, 2517.50]
```

可以看到：

```text
可见光 B2/B3/B4 预测偏亮
SWIR B11/B12 预测偏暗
```

这不是一个简单的整体亮度偏差，而是光谱方向偏了。

所以污染或异常不能只看：

```math
\|\mathbf z_i-\tilde{\mathbf z}_i\|_2
```

还应该看光谱角：

```math
\theta
=
\arccos
\frac{
\mathbf z_i^\top \tilde{\mathbf z}_i
}{
\|\mathbf z_i\|_2
\|\tilde{\mathbf z}_i\|_2
}
```

也就是说，真正重要的是：

```text
六维向量的方向是否合理
```

而不只是：

```text
六维向量的幅度是否大
```

## 7. 端点预测是核心困难之一

目标时间 `20260206T041839` 接近可用时间序列末端。对端点预测来说，harmonic 模型容易产生外推误差。

NUFROST 的主模型是：

```math
\hat{\mathbf y}(t)
=
\boldsymbol\mu
+
\mathbf s\odot
\left(
\mathbf x(t)^\top \Theta
\right)
```

如果目标时间在观测序列内部，模型更像插值；
如果目标时间在观测序列末端，模型更像外推。

在尼西村这个 case 中，外推风险很高：

```text
雪山状态快速变化
目标日期附近有效观测少
长时序 harmonic 会平滑掉端点状态
局部最近观测反而携带更多真实信息
```

这解释了为什么简单 local baseline 能达到 `165.68`，而 NUFROST 主线是 `247.03`。

## 8. endpoint gate 实验

为了解决端点外推问题，实验中加入了 endpoint gate：

```text
--nufrost-endpoint-gate
--nufrost-endpoint-window-days
--nufrost-endpoint-angle-deg
--nufrost-endpoint-z-rms
```

基本逻辑是：

1. 如果目标日期接近时间序列末端；
2. 如果 NUFROST 预测与 local 预测在标准化六维空间中偏离太大；
3. 则回退到 local 预测。

判据包括：

```math
\theta(\hat{\mathbf z}_{nufrost},\hat{\mathbf z}_{local})
>
\tau_\theta
```

或：

```math
\left\|
\hat{\mathbf z}_{nufrost}
-
\hat{\mathbf z}_{local}
\right\|_2
>
\tau_z
```

实验结论：

```text
endpoint gate 没有带来稳定收益。
```

原因是：

```text
local 本身在一些区域也很差；
如果回退目标不可靠，gate 无法解决根本问题。
```

这说明 endpoint gate 只能作为安全机制，不能作为主要重构方法。

## 9. band-wise 与 joint-date 的差异

本轮实验中一个重要发现是：

```text
band-wise local linear 数值上较好，但数学上不干净。
```

band-wise local linear 对每个 band 独立处理：

```math
\hat y_b(t_*)
=
\operatorname{interp}
\left(
\{(t_i,y_{i,b})\}
\right)
```

这意味着：

```math
\hat{\mathbf y}(t_*)
=
[
\hat y_1(t_*),
\hat y_2(t_*),
\dots,
\hat y_B(t_*)
]
```

但每个：

```math
\hat y_b(t_*)
```

可能来自不同日期、不同观测条件。

因此它不一定对应一个真实存在过的多光谱状态。

相比之下，joint-date 方法要求六个 band 来自同一个日期：

```math
\mathbf y_i
=
[
y_{i,1},y_{i,2},\dots,y_{i,B}
]
```

这更符合“多光谱是高维向量轨迹”的思想。

但是实验显示，joint-date local 方法在尼西村全景上表现不好：

| 方法 | 全景 RMSE |
|---|---:|
| band-wise local linear | `165.68` |
| clean-idw3 | `218.61` |
| clean-linear | `238.68` |

这说明：

```text
数学更干净的方法未必在观测稀疏/污染复杂场景中直接更准。
```

## 10. 历史 oracle 实验

为了判断 RMSE 100 是否理论上可能，做了历史 oracle 诊断。

历史 oracle 的定义是：

对每个像元，假设我们可以从所有历史非目标日期中直接挑选与目标真值最接近的六维观测：

```math
i^*
=
\arg\min_i
\left\|
\mathbf y_i
-
\mathbf y_*
\right\|_2
```

然后：

```math
\hat{\mathbf y}_*
=
\mathbf y_{i^*}
```

这个方法当然不能作为真实算法，因为它使用了目标真值，只能作为理论下限。

抽样历史 oracle 结果约为：

```text
sample_historical_oracle_rmse = 54.22
```

这个结果非常重要。

它说明：

```text
尼西村并不是信息论上无法达到 RMSE 100。
```

历史中确实存在很多与目标状态相似的观测。

真正困难的是：

```text
如何在不看目标真值的情况下，为每个像元选中正确历史状态。
```

## 11. analogue transition 实验

基于历史 oracle 的启发，尝试了 analogue transition。

基本思想：

如果当前最近状态是：

```math
\mathbf y_0
```

在历史中寻找相似状态：

```math
\mathbf y_i \approx \mathbf y_0
```

并使用历史转移：

```math
\mathbf y_i \to \mathbf y_j
```

预测当前目标：

```math
\hat{\mathbf y}_*
=
\mathbf y_0
+
(\mathbf y_j-\mathbf y_i)
```

也就是：

```text
把历史相似状态的变化量迁移到当前状态上。
```

实验模式：

```text
analog-delta
```

结果：

```text
彩点 64x64 RMSE = 591.78
```

该方法失败。

失败原因可能是：

1. 雪山状态变化不是平稳可迁移的；
2. 相似状态的后续变化受天气、积雪、照度影响很大；
3. 历史中的状态距离相近，不代表变化方向也相近；
4. 增量迁移会放大误差。

因此简单的：

```math
\mathbf y_0 + (\mathbf y_j-\mathbf y_i)
```

不稳定。

## 12. 为什么 RMSE 100 没有继续追

最终没有继续追 RMSE 100，原因不是理论上完全不可能，而是：

```text
尼西村作为唯一优化目标过于极端。
```

它同时包含：

1. 端点预测；
2. 雪山状态突变；
3. 观测稀疏；
4. 云/雪/影/照度混合；
5. 高海拔地形导致的空间非平稳；
6. band-wise 方法和 joint-vector 方法之间的矛盾。

如果继续只围绕尼西村调参，很容易得到一个对该 case 过拟合、但论文方法不稳健的特殊规则。

因此更合理的判断是：

```text
尼西村应作为 stress test，而不是作为唯一优化目标。
```

## 13. 本轮实验学到的东西

### 13.1 NUFROST 的主要短板不是 NUFFT

NUFFT 本身不是当前主要问题。

主要问题发生在：

```text
频率选择之后的时域重构与端点预测
```

尤其是：

```text
harmonic basis 在雪山端点状态下可能外推出错误光谱方向。
```

### 13.2 多光谱必须被视为高维状态

band-wise 方法数值上有时更好，但从物理和数学上，它可能生成不存在的光谱状态。

长期来看，NUFROST 应坚持：

```math
\mathbf y(t)\in\mathbb R^B
```

而不是：

```math
y_b(t)
```

的独立集合。

### 13.3 污染不是高幅度变化

更准确的污染定义应综合：

```text
幅度异常
光谱角异常
时间上下文
空间上下文
质量先验
```

而不是单独依赖：

```math
\|\mathbf z_i-\tilde{\mathbf z}_i\|_2
```

### 13.4 端点预测需要单独建模

NUFROST 当前是统一模型：

```text
内部插值和端点外推使用同一套 harmonic reconstruction
```

但实验显示，端点目标需要额外策略：

```text
endpoint confidence
local observation anchoring
state selection fallback
```

### 13.5 历史状态选择值得研究

历史 oracle 结果说明，历史库里有足够信息。

真正值得研究的是：

```math
\operatorname{select}
\left(
\mathbf y_i
\mid
\mathbf y_{1:n}, t_*
\right)
```

也就是无监督地选择目标状态，而不是强行拟合一个光滑 harmonic 轨迹。

## 14. 对论文写作的影响

这轮实验不应该写成：

```text
NUFROST 已经解决所有雪山异常。
```

更合理的写法是：

```text
NUFROST 在多数时序重构任务上提供稳定的向量频谱重构；
但在观测末端、雪山快速状态切换、有效观测不足的极限场景中，harmonic 重构可能退化。
```

可以把尼西村作为限制性讨论：

```text
Nixi Village stress case indicates that endpoint reconstruction over snow-covered mountains requires additional state-selection or quality-aware fallback mechanisms.
```

也可以作为未来工作：

```text
Future work may combine NUFROST with a quality-aware analogue state selection module for endpoint snow-cover transitions.
```

## 15. 后续建议

### 15.1 不建议继续只调尼西村

继续围绕尼西村追 RMSE 100，风险是：

```text
过拟合 stress case
牺牲方法统一性
引入太多经验规则
论文叙事变复杂
```

### 15.2 建议建立多场景评估集

应至少包含：

```text
普通植被区
农田区
城市/裸地区
雪山区
云污染严重区
端点预测区
内部插值区
```

然后比较：

```text
平均 RMSE
端点 RMSE
雪区 RMSE
空间粗糙度
光谱角误差
```

### 15.3 为 NUFROST 增加诊断指标

建议输出：

```text
target 是否为端点
nearest observation gap
effective observation count
Huber outlier ratio
prediction leverage
spectral angle to nearest clean observation
temporal roughness
selected frequency count
```

尤其是 prediction leverage：

```math
h_*
=
\mathbf x_*^\top
A^{-1}
\mathbf x_*
```

如果：

```math
h_*
```

过大，说明目标时间预测高度不稳定，应触发 fallback 或降低模型阶数。

### 15.4 未来可研究的数学方向

#### 方向 A：quality-aware vector Huber

加入先验质量权重：

```math
\Omega
=
\operatorname{diag}(q_i w_i)
```

目标函数：

```math
\min_\Theta
\frac12
\left\|
\Omega^{1/2}
(Z-X\Theta)
\right\|_F^2
+
\frac12
\left\|
\Lambda^{1/2}\Theta
\right\|_F^2
```

#### 方向 B：endpoint leverage fallback

如果：

```math
h_*>\tau_h
```

则降低模型阶数或回退到状态选择模块。

#### 方向 C：spectral-angle pollution score

污染分数可定义为：

```math
s_i
=
\alpha
\left\|
\mathbf z_i-\tilde{\mathbf z}_i
\right\|_2
+
(1-\alpha)
\theta
(\mathbf z_i,\tilde{\mathbf z}_i)
```

#### 方向 D：analogue state selection

从历史中选择目标状态：

```math
i^*
=
\arg\min_i
\mathcal L
(
\mathbf y_i,
\mathcal C(t_*),
\mathcal N_p
)
```

其中：

```text
\mathcal C(t_*) 是目标时间上下文
\mathcal N_p 是空间邻域
```

#### 方向 E：latent state model

把六维光谱向量投影到低维状态空间：

```math
\mathbf z(t)
\approx
C\mathbf a(t)
```

然后在 latent space 中做频谱重构或状态选择。

## 16. 当前建议结论

本轮实验的最终判断是：

```text
NUFROST 的主方向仍然成立；
尼西村暴露的是极限端点雪山场景下的状态选择问题；
继续把尼西村 RMSE 100 作为单一目标不合理；
应该把尼西村保留为 stress test，并在论文中作为方法限制或未来工作讨论。
```

一句话总结：

```text
尼西村不是证明 NUFROST 失败的 case，而是提醒我们：
多光谱重构不能只拟合平滑轨迹，还需要判断目标时刻处于哪一种真实地表状态。
```


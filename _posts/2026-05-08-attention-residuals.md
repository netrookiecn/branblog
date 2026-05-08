---
layout: post
title: "Attention Residuals：Kimi 用深度注意力干掉了 PreNorm 的稀释问题"
date: 2026-05-08
tags: [LLM, Architecture, Kimi, Residual]
---

Kimi 团队这篇 Attention Residuals 让我重新审视了一个我原以为"解决了就不用管"的问题：**残差连接**。

论文的核心发现：PreNorm + 标准残差在深层模型上有结构性缺陷——随着层数增加，hidden state 的 magnitude 是 O(L) 增长的，**深层的输出会被浅层的累积稀释**。Kimi 在 48B MoE 模型上观察到，Baseline 的前 10 层梯度是后 10 层的 3-4 倍（Figure 5c）。

他们的解法出乎我意料：**把残差连接从固定加权换成深度维度的 softmax attention**。

这篇文章我想讲三件事：
1. **PreNorm 的稀释问题到底有多严重**——不是"可能有影响"，是"深层 40% 的层可以直接剪掉"
2. **Attention Residuals 的核心 insight**——时间维度用 attention 替代 RNN 成功了，深度维度为什么不行？
3. **Block AttnRes 的工程落地**——怎么在 pipeline 并行下把 O(Ld) 通信开销压到可接受

---

## 一、残差连接的隐藏债务：PreNorm 稀释

我原来以为残差连接是个"解决了就不用管"的问题——梯度高速公路嘛，He Kaiming 2015 年就搞定了。

翻完 Kimi 这篇 report，我的判断变了：**残差连接解决了梯度消失，但制造了信息稀释**。

### 问题出在哪？

标准残差的更新公式是 `h_l = h_{l-1} + f_{l-1}(h_{l-1})`。展开这个递归：

```
h_L = h_1 + f_1(h_1) + f_2(h_2) + ... + f_{L-1}(h_{L-1})
```

每一层的输出权重都是 1。听起来很公平？

问题是：**随着深度增加，hidden state 的 magnitude 是 O(L) 增长的**。第 100 层想要影响最终输出，它的贡献会被前 99 层的累积稀释到 1/100。

Kimi 的实验数据很直观（Figure 5b）：Baseline 模型的输出 magnitude 从第 1 层到第 27 层**单调递增**，深层不得不学习越来越大的输出才能"被听见"。

### 这不只是理论问题

论文引用了一个让我震惊的结论：**"a significant fraction of layers can be pruned with minimal loss"**（引自 [11]）。

我原来以为层剪枝是"牺牲一点性能换效率"的 trade-off。现在看来，很多深层压根就没学到有用的东西——它们的信号被稀释了。

> 这一节讲的是"大家都在承受但没人解决的问题"；下一节看 Kimi 怎么从时间-深度对偶性找到解法。

---

## 二、核心 Insight：深度维度也可以做 Attention

### 时间-深度对偶

Kimi 的核心观察是一个类比：

| 维度 | 传统方案 | 问题 | 解法 |
|------|----------|------|------|
| **时间（序列）** | RNN 递归 | 长距离依赖丢失 | Transformer Attention |
| **深度（层）** | 残差累加 | 深层贡献被稀释 | **Attention Residuals** |

RNN 把所有历史信息压缩到一个 hidden state，Transformer 用 attention 让每个位置直接访问所有历史位置。

残差连接把所有浅层输出压缩到一个累加态，Attention Residuals 让每一层直接访问所有浅层输出，**用 learned, input-dependent 的权重做选择性聚合**。

### Full AttnRes 的公式

```
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
```

其中：
- `v_0 = h_1`（token embedding）
- `v_i = f_i(h_i)` for i ≥ 1（各层输出）
- `α_{i→l} = softmax(w_l^T · RMSNorm(k_i))`（注意力权重）
- `w_l` 是每层一个**可学习的 pseudo-query 向量**

关键设计：**query 是 learned parameter，不是 input-dependent projection**。

我一开始觉得这是偷懒——为什么不用 `W_q · h_l` 做 query？

论文的 ablation（Table 4）给出了答案：input-dependent query 确实能把 loss 从 1.737 降到 1.731，但代价是每层多一个 d×d 的投影矩阵，而且**推理时必须串行计算**，没法用后面要讲的 two-phase 优化。

这是个典型的"工程约束反过来塑造算法设计"的例子。

> 这一节讲的是算法原理；下一节看怎么在大规模训练中落地。

---

## 三、Block AttnRes：从 O(Ld) 到 O(Nd) 的工程妥协

![Standard Residual vs Full AttnRes vs Block AttnRes](/assets/images/01-attnres-comparison.svg)

### Full AttnRes 的问题

在小规模训练中，Full AttnRes 几乎没有额外开销——反正 backward 要保存所有层的 activation。

但在大规模训练中，两个因素改变了一切：
1. **Activation recomputation**：为了省显存，forward 时不保存中间 activation，backward 时重算
2. **Pipeline parallelism**：模型切成多个 stage 放在不同 GPU 上

这两个优化一开，Full AttnRes 的 O(Ld) 通信开销就爆炸了：每一层的输出都要跨 stage 传输。

### Block AttnRes 的设计

核心思路：**把 L 层分成 N 个 block，block 内用标准残差，block 间用 attention**。

```python
# Block 内：标准残差累加
b_n = Σ_{j ∈ B_n} f_j(h_j)

# Block 间：attention over block representations
h_l = Σ_{i=0}^{n-1} α_{i→l} · b_i + α_{partial} · b_n^{j}
```

这把存储和通信都从 O(Ld) 压到 O(Nd)。论文用 N≈8，实测 **end-to-end training overhead < 4%**。

### Scaling Law 验证

这是让我信服的关键实验（Figure 4）：

| 方案 | Scaling Law 拟合 | 等效计算优势 |
|------|-----------------|-------------|
| Baseline | L = 1.891 × C^{-0.057} | — |
| Block AttnRes | L = 1.870 × C^{-0.058} | **1.25× compute** |
| Full AttnRes | L = 1.865 × C^{-0.057} | — |

Block AttnRes 在最大规模（528M 参数）上只比 Full AttnRes 差 0.001 loss，但通信开销低一个数量级。

### Two-Phase 推理优化

推理时还有一个巧妙的优化（Algorithm 1）：

**Phase 1**：所有 S 层的 inter-block attention 可以**并行计算**（因为 pseudo-query 是 learned parameter，不依赖 forward 结果）

**Phase 2**：intra-block attention 串行计算，用 online softmax merge 和 Phase 1 结果合并

这把 per-layer memory access 从 O(Ld) 压到 O((N/S + 5)d)，实测 **inference latency overhead < 2%**。

> 到这里"怎么做"讲完了；下一节看实际效果和我的一些思考。

---

## 四、效果与暗线

### 下游任务提升

48B MoE 模型在 1.4T tokens 上的结果（Table 3）：

| 任务类型 | 代表 benchmark | 提升 |
|----------|---------------|------|
| 多步推理 | GPQA-Diamond | **+7.5** |
| 数学 | Minerva Math | +3.6 |
| 代码 | HumanEval | +3.1 |
| 知识 | MMLU | +1.1 |

**推理任务提升最大**——这和"深层可以选择性访问浅层表示"的 hypothesis 一致：compositional reasoning 需要后面的层"回头看"前面的中间结果。

### 训练动态的变化

Figure 5 展示了两个关键变化：

1. **Output magnitude bounded**：Block AttnRes 的输出 magnitude 呈周期性模式，不再单调增长
2. **Gradient distribution uniform**：Baseline 的早期层梯度是深层的 3-4 倍，AttnRes 后分布均匀得多

这解释了为什么 AttnRes 在深模型上效果更好（Figure 7）：**它让深度真正可用了**。

### 我归纳的暗线

翻完整篇 report，我发现一个没人直说的规律：**对"均匀"的追求在往更多维度扩散**。

- 2017：Attention 让序列维度的信息访问从"最近优先"变成"按需选择"
- 2020：MoE 让参数维度的激活从"全量"变成"按需路由"
- 2026：AttnRes 让深度维度的聚合从"固定权重"变成"按需 attention"

这三个方向的共同点：**打破固定的、uniform 的聚合方式，换成 learned、input-dependent 的选择**。

如果这个模式继续，下一个被"attention 化"的维度会是什么？Batch 维度？Head 维度？

*（本文归纳，非原论文措辞）*

---

## 五、我的判断和待验证问题

### 我现在的选择

如果我明天要训一个 >50 层的模型：

1. **一定会用 Block AttnRes**（N=8），overhead 可接受，收益确定
2. **不会用 Full AttnRes**，除非 interconnect 带宽再涨 10 倍
3. **会重新审视"深度 vs 宽度"的 trade-off**——AttnRes 让深模型的边际收益变高了

### 待验证的问题

1. **和 DeepNorm 的关系**：DeepNorm 也是解决深度训练问题的，两者能叠加吗？
2. **在 dense model 上的效果**：论文只验证了 MoE，dense Transformer 呢？
3. **长上下文场景**：prefilling 128K tokens 时 block representations 的显存开销（论文说 ~1.9GB/device），在 8 卡机上可能还是 bottleneck

---

## 附：关键数字速查

| 指标 | 数值 | 来源 |
|------|------|------|
| Block 数量 | N ≈ 8 | §5, 经验最优 |
| Training overhead | < 4% | §4.1 |
| Inference latency overhead | < 2% | §4.2 |
| Compute equivalence | 1.25× | Figure 4 |
| GPQA-Diamond 提升 | +7.5 | Table 3 |

> 代码已开源：[github.com/MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)

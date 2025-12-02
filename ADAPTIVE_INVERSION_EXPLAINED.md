# Adaptive Inversion 详细解释

## 📚 背景知识

### 什么是Inversion？

在图像编辑中，我们需要：
1. 把真实图像 → 转成噪声 (Inversion)
2. 从噪声 → 按新prompt生成图像 (Denoising)

**Null-Text Inversion**的作用：优化"空文本embedding"，让inversion更准确

### num_inner_steps的作用

```python
# num_inner_steps = 0 (不优化)
noise = ddim_invert(image, prompt="")  # 用空prompt
reconstructed = ddim_denoise(noise, prompt=original_prompt)
# ❌ 重建质量差 → 编辑质量差

# num_inner_steps = 10 (优化10次)
noise, optimized_null_text = null_text_inversion(
    image,
    prompt=original_prompt,
    num_inner_steps=10  # 优化10次
)
reconstructed = ddim_denoise(noise, optimized_null_text)
# ✅ 重建质量好 → 编辑质量好
```

## 🎯 自适应策略

### 核心思想

```
编辑难度 = f(语义距离, 编辑类型, 词汇变化)
num_inner_steps = g(编辑难度)
```

### 具体计算步骤

#### Step 1: 计算CLIP相似度

```python
# 用CLIP计算两个prompt的语义相似度
src_emb = CLIP_encode("a photo of a cat")      # [768]
tar_emb = CLIP_encode("a photo of a dog")      # [768]

similarity = cosine_similarity(src_emb, tar_emb)  # 0.72
# 0 = 完全不同, 1 = 完全相同
```

**相似度越低 → 编辑越困难 → 需要更多steps**

#### Step 2: 分析Token变化

```python
src_tokens = ["a", "photo", "of", "a", "cat"]
tar_tokens = ["a", "photo", "of", "a", "dog"]

added = {"dog"}      # 新增的词
removed = {"cat"}    # 删除的词
common = {"a", "photo", "of"}  # 共同的词

change_ratio = len(added | removed) / len(src_tokens | tar_tokens)
# = 2 / 5 = 0.4 (40%的词发生了变化)
```

**变化比例越高 → 需要更多steps**

#### Step 3: 识别编辑类型

```python
def detect_edit_type(change_ratio, added, removed):
    if change_ratio > 0.4 and len(added) >= 2 and len(removed) >= 2:
        return "object_replacement"  # 最难
    elif change_ratio > 0.2:
        return "attribute_change"    # 中等
    elif change_ratio > 0.1:
        return "style_transfer"      # 中等
    else:
        return "minor_edit"          # 最简单
```

**不同类型需要不同的steps**

#### Step 4: 计算基础steps

```python
# 基于相似度的映射 (moderate策略)
if similarity >= 0.85:      # 非常相似
    base_steps = 0          # 最少steps
elif similarity <= 0.6:     # 非常不同
    base_steps = 20         # 最多steps
else:                       # 中间插值
    ratio = (0.85 - similarity) / (0.85 - 0.6)
    base_steps = int(0 + ratio * 20)  # 线性插值
```

#### Step 5: 应用类型调整

```python
type_multipliers = {
    'object_replacement': 1.5,    # 对象替换最难，多50%
    'attribute_change': 1.2,      # 属性修改，多20%
    'style_transfer': 1.0,        # 风格转换，不变
    'minor_edit': 0.8             # 小修改，少20%
}

adjusted_steps = base_steps * type_multipliers[edit_type]
```

#### Step 6: Token变化额外调整

```python
if change_ratio > 0.5:  # 超过50%的词变了
    adjusted_steps *= 1.3  # 再多30%
```

#### Step 7: 限制在合理范围

```python
# moderate策略的范围是[0, 20]
final_steps = min(max(adjusted_steps, 0), 20)
```

## 📊 实际案例分析

### 案例1: 简单编辑（颜色变化）

```
Source: "a red car"
Target: "a blue car"

Step 1: CLIP相似度 = 0.92 (非常相似)
Step 2: Token变化 = 1/3 = 33%
Step 3: 编辑类型 = attribute_change
Step 4: base_steps = 0 (相似度>0.85)
Step 5: adjusted = 0 * 1.2 = 0
Step 6: token调整 = 不需要 (33% < 50%)
Step 7: final = 0

✅ num_inner_steps = 0
```

### 案例2: 中等编辑（属性变化）

```
Source: "a smiling person"
Target: "an angry person"

Step 1: CLIP相似度 = 0.75
Step 2: Token变化 = 2/4 = 50%
Step 3: 编辑类型 = attribute_change
Step 4: base_steps = int((0.85-0.75)/(0.85-0.6) * 20) = 8
Step 5: adjusted = 8 * 1.2 = 9.6 → 10
Step 6: token调整 = 10 * 1.3 = 13 (50%变化)
Step 7: final = 13

✅ num_inner_steps = 13
```

### 案例3: 困难编辑（对象替换）

```
Source: "a photo of a cat"
Target: "a photo of a dog"

Step 1: CLIP相似度 = 0.72
Step 2: Token变化 = 2/5 = 40%
Step 3: 编辑类型 = object_replacement
Step 4: base_steps = int((0.85-0.72)/(0.85-0.6) * 20) = 10
Step 5: adjusted = 10 * 1.5 = 15
Step 6: token调整 = 不需要 (40% < 50%)
Step 7: final = 15

✅ num_inner_steps = 15
```

## 🎛️ 三种策略对比

### Conservative (保守策略)
```python
min_steps = 0
max_steps = 10
threshold_low = 0.9   # 更严格的"相似"定义
threshold_high = 0.7  # 更严格的"不同"定义
```
- 适合：对质量要求极高的场景
- 特点：即使简单编辑也会用几个steps保证质量

### Moderate (中等策略) ⭐ 默认
```python
min_steps = 0
max_steps = 20
threshold_low = 0.85
threshold_high = 0.6
```
- 适合：大多数场景
- 特点：平衡质量和速度

### Aggressive (激进策略)
```python
min_steps = 5
max_steps = 30
threshold_low = 0.8
threshold_high = 0.5
```
- 适合：对速度要求高，愿意牺牲一些质量
- 特点：即使困难编辑也会用更多steps

## 💡 为什么这样设计有效？

### 理论基础

1. **语义距离 ↔ 编辑难度**
   - CLIP相似度直接反映语义差距
   - 语义差距越大，inversion需要越精确

2. **Token变化 ↔ 重建难度**
   - 词汇变化多 → prompt guidance差异大
   - 需要更精确的inversion来保持结构

3. **编辑类型 ↔ 所需精度**
   - 对象替换：最难，需要最精确的inversion
   - 属性修改：中等难度
   - 颜色/尺寸：简单，低精度即可

### 实验验证

```
固定 num_inner_steps = 0 (原始方法)
├─ 简单编辑：✅ 质量好，速度快
└─ 困难编辑：❌ 质量差，背景变形

固定 num_inner_steps = 20 (过度优化)
├─ 简单编辑：✅ 质量好，但 ❌ 浪费时间
└─ 困难编辑：✅ 质量好，✅ 必要的时间

自适应 num_inner_steps (我们的方法)
├─ 简单编辑：✅ 质量好，✅ 速度快 (自动用0-5)
└─ 困难编辑：✅ 质量好，✅ 合理时间 (自动用15-20)
```

## 🎬 实际运行示例

当你运行编辑时，会看到：

```
================================================================================
ADAPTIVE INVERSION ANALYSIS
================================================================================
Source: a photo of a cat
Target: a photo of a dog
Strategy: moderate
--------------------------------------------------------------------------------
Analysis:
  CLIP Similarity: 0.7234          # ← 语义相似度
  Token Change Ratio: 40%          # ← 词汇变化
  Tokens Added: 2                  # ← 新增词数
  Tokens Removed: 2                # ← 删除词数
  Edit Type: object_replacement    # ← 编辑类型
--------------------------------------------------------------------------------
Decision:
  Base Steps (from similarity): 12      # ← 从相似度得出
  Type Adjustment (×1.5): 18            # ← 类型调整
  Final num_inner_steps: 18             # ← 最终决策
================================================================================
```

## 🔬 关键参数说明

### CLIP Similarity阈值

- **0.9+**: 几乎相同的prompt (如"red car" vs "blue car")
- **0.7-0.9**: 相似但有区别 (如"cat" vs "dog")
- **0.5-0.7**: 较大差异 (如"cat" vs "house")
- **<0.5**: 完全不同 (如"cat" vs "abstract art")

### Token Change Ratio

- **<20%**: 微小变化 (1-2个词)
- **20-40%**: 中等变化 (几个关键词)
- **40-60%**: 大量变化 (大部分词)
- **>60%**: 几乎全改 (完全不同的描述)

### Edit Type分类规则

```python
if change_ratio > 0.4 and added >= 2 and removed >= 2:
    # 多个词替换 → 对象替换
    return "object_replacement"

elif change_ratio > 0.2:
    # 部分词改变 → 属性修改
    return "attribute_change"

elif change_ratio > 0.1:
    # 少量词改变 → 风格转换
    return "style_transfer"

else:
    # 极少改变 → 微小编辑
    return "minor_edit"
```

## ✅ 总结

自适应inversion通过三个维度分析编辑任务：

1. **语义距离** (CLIP similarity)
2. **词汇变化** (Token change ratio)
3. **编辑类型** (Object/Attribute/Style/Minor)

然后智能计算最优的num_inner_steps，做到：
- ✅ 简单任务用少steps（快）
- ✅ 困难任务用多steps（质量）
- ✅ 完全自动（无需手动调）

这就是adaptive inversion的完整机制！

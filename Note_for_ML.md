# **Intermediate ML**

## **Aug 27 Course Overview**

### **Vocabulary**

| voca           | meaning                          | voca       | meaning  |      |
| -------------- | -------------------------------- | ---------- | -------- | ---- |
| Semitic        | 语意                             | Symbolic   | 符号     |      |
| neocortex      | 新皮质（与知觉、意识、语言有关） | bogus      | 假冒的   |      |
| modus operandi | 拉丁语，作案手法                 | heredity   | 遗传的   |      |
| reciprocal     | 倒数                             | Derivative | 导数     |      |
| systolic       | 心脏压缩的                       | diastolic  | 心脏舒张 |      |
| intercept      | 截距                             | hr         | 心率     |      |
|                |                                  |            |          |      |
|                |                                  |            |          |      |

### **Content**

Two types of Intelligence:

1. sensory：需要寓意和流程知识。学的慢，应用快；需要很多知识；Deep learning就属于这种。高效函数，可以用于识别。
2. relational：创新、关联。学得快，应用慢；只需要很少明确的知识；符号处理和概念。
3. Can both types be supported in a single architecture? 即未来AI的发展是想要统一sensory和relation。比如transformer，仍然不能像人类一样从有限的例子里举一反三，不能快速抽象、联想和泛化。

relations are essential for reasoning

Shortcomings are masked:

1. 过度自信和虚假推理：bogus reasoning
2. 进来的AI隐藏起来了这个缺陷
3. systems是为了方便人类

Make assumptions by knowledge --> discover patterns by datas --> predict&explore --> criticize model --> revise assumptions

#### **Supervised learning**

1. **Sparse regression** 
   1. 稀疏回归是一种特殊的回归方法，它希望模型在学习过程中能自动**“选择”最重要的特征**。简单来说，就是让那些不重要的特征的权重变为零。这有助于简化模型、防止过拟合，并能帮助我们理解数据中哪些因素最关键。这就像一个医生在诊断疾病时，忽略那些不相关的症状，只关注少数几个关键指标。
2. **Smoothing and kernels**
   - **平滑（Smoothing）** 是一种处理数据噪音或不规则性的技术。它通过对数据进行平滑处理来揭示其潜在的模式或趋势。
   - **核（Kernels）** 是一种数学技巧，它能将复杂、非线性的数据从低维空间映射到高维空间，从而让数据在高维空间中变得**线性可分**。这使得一些原本无法用简单直线（或平面）分割的数据，能用更简单的模型来处理。这就像把一张揉皱的纸展开，让它更容易被分割。
3. **(Convolutional) neural networks**
   - **神经网络（Neural Networks）** 是一类模仿人脑神经元结构的模型。它由多层节点组成，通过学习输入数据来调整节点之间的连接权重，从而做出预测。
   - **卷积神经网络（CNNs）** 是一种特别适合处理**图像、视频**等网格状数据的神经网络。它通过使用“卷积层”来自动提取图像中的特征（比如边缘、形状和纹理）。CNNs 在图像识别、目标检测等领域取得了巨大成功。
4. **Risk bounds and generalization error**
   - **风险界限（Risk Bounds）** 是一个理论概念，<u>它为模型的**泛化能力**提供了一个数学上的保证</u>。它帮助我们估算模型在未见过的数据上表现有多好。
   - **泛化误差（Generalization Error）** 指的是模型在**新数据**（即未用于训练的数据）上的表现与在训练数据上的表现之间的差异。如果泛化误差很小，说明模型学习到了数据的普遍规律，而不是仅仅记住了训练集上的例子（过拟合）。这衡量了模型的“举一反三”能力。

#### Unsupervised (and self-supervised) learning (no Label)

1. **Nonparametric Bayes**
   - **贝叶斯方法**是一种基于概率的统计推理方法。它利用先验知识来推断未知事件的概率。
   - **非参数**在这里指的是模型结构的灵活性。传统的“参数化”模型有固定的参数数量，而**非参数贝叶斯**<u>模型可以根据数据量自动调整模型复杂度</u>，参数数量可以无限增长。这使得它能更好地适应复杂的数据结构，而不是被预设的规则所限制。它在聚类、主题建模等领域很常见。
2. **Approximate inference（近似推断）**
   - 在复杂的概率模型中，精确计算某些推断（比如后验概率）可能在计算上非常困难甚至不可行。
   - **近似推断**就是用来解决这个问题的。它使用一些巧妙的近似方法（比如变分推断或<u>马尔可夫链蒙特卡洛方法</u>），来快速、有效地得到一个足够好的近似结果，即使不能得到精确解。这就像在复杂的数学问题中，我们不追求精确解，而是找到一个非常接近的近似值，来解决实际问题。
3. **Approaches to generative models**
   - **生成模型**是一类能够学习数据的内在分布，并能**生成**新数据的模型。例如，训练一个生成模型来学习猫的图片，它就能生成全新的、不存在的猫图片。
   - 这与**判别模型**（如分类器）不同，判别模型只是学习如何区分不同类别的数据。生成模型在图像生成（如GANs、Diffusion Models）、文本生成等领域扮演着核心角色。
4. **Structure learning**
   - **结构学习**是贝叶斯网络（一种特殊的概率图模型）中的一个重要概念。它不仅仅是学习模型中的参数，更重要的是**学习变量之间的依赖关系**，即网络中的**“连接结构”**。
   - 例如，在医疗数据中，结构学习可以帮助我们发现疾病A与症状B、C之间的因果关系，并绘制出这种关系图。这能让模型更加透明和可解释，因为它揭示了数据背后的逻辑结构。

#### Reinforcement learning (no label)

强化学习与监督学习和无监督学习都不同，它更像是一种“试错学习”。在强化学习中，一个**智能体（agent）** 在一个**环境（environment）** 中行动。它的目标是通过一系列行动来最大化一个累积的奖励。它没有标签数据，也没有要发现的隐藏模式，它通过执行某个动作后得到的“奖励”或“惩罚”来学习。这就像教小狗坐下，它做对了就给零食（奖励），做错了就得不到。

1. **Deep Q-Learning（深度Q学习）** 这是强化学习中的一个经典方法。**Q学习**的目的是学习一个“Q值”，它代表在特定状态下采取某个行动后能获得的**最大未来奖励**。这个Q值可以帮助智能体决定下一步该怎么走。**深度Q学习**则是在此基础上，用**深度神经网络**来近似这个Q值函数。这让<u>模型能够处理非常复杂、高维度的状态空间（比如视频游戏画面）</u>，而不是简单的表格数据。

2. **Policy gradient methods（策略梯度方法）** 与学习Q值不同，策略梯度方法直接学习一个“策略”（policy）。这个**策略**是一个函数，它告诉智能体在任何给定状态下，应该采取什么行动。<u>策略梯度方法通过调整策略，使得那些能够获得更高奖励的行动出现的概率更大</u>。这就像调整一个指南针，让它更频繁地指向能带来宝藏的方向。

3. **Actor-Critic approaches（演员-评论家方法）** 这是结合了前面两种方法思想的一类算法。它有两个核心部分：

   - **演员（Actor）**：这个部分就像一个“策略”，负责根据当前状态决定采取什么行动。

   - **评论家（Critic）**：这个部分负责对演员的行动进行“评价”，告诉它这个行动是好还是坏。 演员根据评论家的反馈来调整自己的策略，以做出更好的决策。这种双管齐下的方法，通常比单独使用其中一种方法更稳定、更高效。

#### Representation/Sequence learning

1. **Classical techniques (Kalman filters, HMMs)** 这些是处理序列数据的**经典方法**。

   - **卡尔曼滤波器（Kalman Filters）** 是一种强大的算法，特别擅长处理带有**不确定性**的动态系统。它通过一系列的测量数据，来**估计**系统随时间变化的状态。这就像一个GPS导航系统，它通过接收卫星信号来不断修正你的位置和速度，即使信号不稳定，它也能给出最佳的估计。

   - **隐马尔可夫模型（HMMs）** 是一种概率模型，它假设系统有一个我们无法直接观察到的“隐藏”状态，而我们能观察到的数据是这些隐藏状态的**结果**。HMMs 被广泛应用于语音识别和生物信息学中，用来推断隐藏的语言模式或DNA序列。

2. **Recurrent neural networks（循环神经网络，RNNs）** RNNs 是一类专门处理序列数据的**神经网络**。它有一个**“记忆”**功能，可以将先前的信息传递给当前步骤。这使得它能够理解和处理序列中的上下文关系。例如，在处理一句话时，RNNs 会记住前面单词的信息，从而更好地理解后面的单词。不过，传统的 RNNs 在处理长序列时会遇到“梯度消失”的问题，难以捕捉长距离的依赖关系。

3. **Attention and language models（注意力机制与语言模型）**

   - **注意力机制（Attention Mechanism）** 是一种革命性的技术。它允许模型在处理序列数据时，将注意力集中在序列中的**特定部分**，而不是平均地关注所有信息。这就像你在读一篇文章时，某些关键词或短语会让你特别注意，而其他部分则相对次要。注意力机制极大地提高了模型处理长序列的能力，解决了 RNNs 遇到的问题。

   - **语言模型（Language Models）** 旨在学习语言的统计规律，能够预测下一个单词或生成连贯的文本。注意力机制的出现，让大型语言模型（LLMs）能够高效地处理和生成长篇文本。

4. **Transformers** Transformer 模型是**基于注意力机制**的神经网络架构。它彻底抛弃了传统的 RNNs 结构，完全依赖于注意力机制来处理序列数据。Transformer 在自然语言处理（NLP）领域带来了巨大的突破，是目前最先进的语言模型（如GPT、BERT）的基础。它能够并行处理整个序列，这比 RNNs 的顺序处理快得多，并且能更好地捕捉长距离的依赖关系。

## Aug 29 Sparse Regression

**Ridge regression**: a technique used to address **overfitting** in linear regression models. 

#### Regression

#### High dimensional regression

#### Sparsity and lasso

## **Sep 3 Lasso, smoothing and kernels**

#### Continuation of lasso

Select $\lambda$ by risk estimation

#### A simple algorithm for the lasso

#### Nonparametric regression

#### Smoothing methods

#### Bias, variance, and curse of dimensionality
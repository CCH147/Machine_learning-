
## 1. SVM (Support Vector Machine) 的訓練過程

SVM 的核心目標是找到一個「超平面」（hyperplane），這個超平面能將不同類別的數據點分隔開，並且擁有「最大邊界」（maximum margin）。

### 核心概念

1.  **超平面 (Hyperplane):** 在 $N$ 維空間中，這是一個 $N-1$ 維的子空間。在 2D 空間中它是一條線，在 3D 空間中它是一個平面。其數學表示為: 
    
   $\vec{w} \cdot \vec{x} + b = 0$
    
   $\vec{w}$ 是法向量（normal vector），決定了超平面的方向。
   $b$ 是偏置（bias），決定了超平面的位置。

2.  **邊界 (Margin):** 指的是超平面與「最接近」它的數據點（稱為**支持向量 (Support Vectors)**）之間的距離。SVM 的目標就是最大化這個距離。

### 訓練過程（優化問題）

SVM 的訓練過程本質上是在求解一個**有條件的優化問題**。

#### a. 硬邊界 (Hard Margin) - 假設數據線性可分

目標： 最大化邊界。最大化邊界 $\frac{2}{||\vec{w}||}$ 等價於最小化 $||\vec{w}||^2$（為了數學計算方便，通常最小化 $\frac{1}{2}||\vec{w}||^2$）。

* **優化目標 (Minimize):**
    
    $\min_{\vec{w}, b} \frac{1}{2} ||\vec{w}||^2$
    
* **條件 (Subject to):** 確保所有數據點都被正確分類。
    
    $y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1 \quad \text{for all } i$
    
    * $y_i$ 是第 $i$ 個數據點的類別（+1 或 -1）。
    * $\vec{x}_i$ 是第 $i$ 個數據點的特徵向量。

#### b. 軟邊界 (Soft Margin) - 允許部分數據點被錯分

在現實世界中，數據很難完全線性可分。因此，我們引入「鬆弛變量」($\xi_i$)，允許一些點跑到邊界內甚至被錯分。

* **優化目標 (Minimize):**
    
    $\min_{\vec{w}, b, \xi} \frac{1}{2} ||\vec{w}||^2 + C \sum_{i=1}^{n} \xi_i$
    
* **條件 (Subject to):**
    
    $y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1 - \xi_i \quad \text{and} \quad \xi_i \ge 0 \quad \text{for all } i$
    
    * $C$ 是一個超參數，用來平衡「最大化邊界」（小 $\frac{1}{2}||\vec{w}||^2$）和「最小化分類錯誤」（小 $\sum \xi_i$）之間的權重。

#### c. 核技巧 (Kernel Trick) - 處理非線性數據

如果數據在原始空間中無法線性分離，SVM 會使用「核函數」（如 RBF 核、多項式核）將數據映射到更高維度的空間，在那個高維空間中尋找線性超平面。

### 求解方法

這個優化問題是一個**凸二次規劃 (Convex Quadratic Programming, QP)** 問題。
1.  **拉格朗日對偶 (Lagrangian Duality):** 傳統上，會將上述問題轉換為其「對偶問題」來求解，這樣更容易引入核函數。
2.  **SMO (Sequential Minimal Optimization):** 一種高效的算法，專門用於訓練 SVM。它通過一次只優化兩個拉格朗日乘數（對應兩個數據點）來迭代求解。
3.  **梯度下降 (SGD):** 對於大型數據集（如線性 SVM），也可以使用隨機梯度下降 (Stochastic Gradient Descent) 來直接優化原始問題（的另一種形式，稱為 Hinge Loss）。

---

## 2. MLP (Multilayer Perceptron) 的訓練過程

MLP 是一種前饋神經網路 (Feedforward Neural Network)，由輸入層、一個或多個隱藏層和一個輸出層組成。其訓練過程的核心是**反向傳播 (Backpropagation)** 演算法。

### 訓練步驟

1.  **初始化 (Initialization):**
    * 隨機初始化網路中所有的權重 ($\vec{W}$) 和偏置 ($\vec{b}$)。

2.  **前向傳播 (Forward Propagation):**
    * 將一批（batch）訓練數據 $\vec{x}$ 輸入網路。
    * 數據從輸入層開始，逐層向前傳遞：
        * 在第 $l$ 層，計算加權總和:  $\vec{z}^{(l)} = W^{(l)} \vec{a}^{(l-1)} + \vec{b}^{(l)}$
        * 通過「激活函數」(Activation Function，如 ReLU, Sigmoid) 得到該層的輸出： $\vec{a}^{(l)} = g(\vec{z}^{(l)})$
    * 重複此過程，直到得到輸出層的預測值 $\hat{y}$（即 $\vec{a}^{(L)}$）。

3.  **計算損失 (Compute Loss):**
    * 使用「損失函数」(Loss Function) 來衡量預測值 $\hat{y}$ 與真實標籤 $y$ 之間的差距。
    * 例如：
        * 迴歸問題： 均方誤差 (MSE) $L = \frac{1}{n} \sum (\hat{y}_i - y_i)^2$
        * 分類問題： 交叉熵 (Cross-Entropy) $L = -\sum y_i \log(\hat{y}_i)$

4.  **反向傳播 (Backpropagation):**
    * 這是訓練的核心。目標是計算損失函數 $L$ 對於網路中**每一個**權重 $w_{ij}$ 和偏置 $b_i$ 的**梯度**（gradient），即偏微分 ($\frac{\partial L}{\partial w_{ij}}$)。
    * 此過程利用**鏈式法則 (Chain Rule)**，從輸出層開始，將「誤差」逐層反向傳播回輸入層。
    * $\frac{\partial L}{\partial w^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial w^{(l)}}$

5.  **權重更新 (Weight Update):**
    * 有了梯度 ($\nabla L$) 之後，使用「優化器」（Optimizer，最基本的是梯度下降）來更新權重和偏置，以試圖最小化損失 $L$。
    * 公式:         
        $W_{\text{new}} = W_{\text{old}} - \eta \cdot \nabla_{W} L$
        
        ($\eta$ 是學習率 Learning Rate)。

6.  **迭代 (Iterate):**
    * 重複步驟 2 到 5，處理多批數據。
    * 當所有數據都處理過一遍時，稱為一個「epoch」。
    * 持續訓練多個 epoch，直到損失收斂或達到預設的 epoch 數量。

---

## 3. 找到最佳 $W$ 的數學推導 (以梯度下降為例)

在機器學習中，"找到最佳 $\vec{W}$" 通常是指找到一組 $\vec{W}$，使得損失函數 $L(\vec{W})$ 最小。

除了 SVM 這種有特定解法（QP）的問題外，對於像 MLP 這樣複雜的模型，我們無法一次性解出「最佳解」。我們使用迭代的方法，例如**梯度下降 (Gradient Descent)**，來**逼近**最佳解。

### 推導過程：

1.  **目標:** 最小化損失函數 $L(\vec{W})$。

2.  **核心概念：梯度 (Gradient)**
    * 梯度 $\nabla L(\vec{W})$ 是一個向量，它指向 $L(\vec{W})$ 在 $\vec{W}$ 當前位置**上升最快**的方向。
    * $\nabla L(\vec{W}) = \left[ \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \dots \right]$

3.  **如何更新 $\vec{W}$？**
    * 為了「最小化」 $L$，我們應該朝著 $L$ **下降最快**的方向移動。
    * 這個方向就是梯度的反方向: $-\nabla L(\vec{W})$。

4.  **定義更新規則：**
    * 我們從當前的權重 $\vec{W}^{(t)}$ 開始，朝著負梯度方向移動一小步。
    * 這一步的大小由**學習率 (Learning Rate)** $\eta$（一個很小的正數）來控制。
    * 因此，我們定義權重的「變化量」($\Delta \vec{W}$)，如下：
        
        $\Delta \vec{W} = - \eta \cdot \nabla L(\vec{W}^{(t)})$
        

5.  **推導出更新公式：**
    * 新的權重 $\vec{W}^{(t+1)}$（您稱為 $W^{*}$）等於舊的權重 $\vec{W}^{(t)}$ 加上這個變化量。
    * 
        $\vec{W}^{(t+1)} = \vec{W}^{(t)} + \Delta \vec{W}$
        
    * 將 $\Delta \vec{W}$ 代入，即得到梯度下降的最終公式：
        
        $\vec{W}^{(t+1)} = \vec{W}^{(t)} - \eta \nabla L(\vec{W}^{(t)})$
        

**總結：** MLP 的反向傳播（Backpropagation）演算法，其目的就是為了高效地計算出 $\nabla L(\vec{W})$，然後利用上述的梯度下降公式來更新 $\vec{W}$。

---

## 4. 程式中如何實現： $w^{*} = w + \Delta w$

$w^{*}$ 指的是「更新後的 $w$」。

將以上述推導為基礎: $\Delta w = - \eta \cdot \text{gradient}$

假設使用的是 Python 和 NumPy 函式庫：

```python
import numpy as np

# 1. 初始化權重 (w) 和學習率 (eta)
# 假設 w 是一個 3x1 的權重向量
w = np.array([0.5, -0.2, 1.0]) 
learning_rate = 0.01  # 學習率 (eta)

# 2. 假設我們已經通過反向傳播計算出梯度 (gradient)
# 梯度的維度必須與 w 相同
gradient = np.array([2.5, -0.8, 0.4])

# 3. 計算權重變化量 (delta_w)
# 根據推導： delta_w = - eta * gradient
delta_w = -learning_rate * gradient

# 4. 實現 w* = w + delta_w
# 這一步就是將舊的 w 加上 delta_w，得到新的 w*
# 在程式中，我們通常直接更新 w 變數
w_star = w + delta_w

# 在一個完整的訓練迴圈中，會寫成：
# w = w + delta_w
# 或者更常見的寫法（合併 3 和 4）：
# w = w - learning_rate * gradient

# --- 打印結果 ---
print(f"原始權重 (w):    {w}")
print(f"計算出的梯度:      {gradient}")
print(f"學習率 (eta):    {learning_rate}")
print(f"權重變化量 (delta_w): {delta_w}")
print(f"更新後權重 (w*): {w_star}")

# 範例輸出：
# 原始權重 (w):    [ 0.5 -0.2  1. ]
# 計算出的梯度:      [ 2.5 -0.8  0.4]
# 學習率 (eta):    0.01
# 權重變化量 (delta_w): [-0.025  0.008 -0.004]
# 更新後權重 (w*): [ 0.475 -0.192  0.996]

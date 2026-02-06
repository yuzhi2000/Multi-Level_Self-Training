# Multi-Level Self-Training for Detecting Aspect Relevance in Social Media Posts
# 多層次自訓練策略用於社群文本之面向指標偵測

---

## 📖 專案簡介 (Introduction)

本專案為碩士論文實作程式碼。本研究針對社群媒體文本（Social Media Posts）中正類別標註資料稀缺特性，提出了一種結合 **監督式對比學習 (Supervised Contrastive Learning, SupCon)** 與 **多層次自訓練 (Multi-Level Self-Training, MST)** 的半監督式學習架構。

### 貢獻
1.  **文本分段策略**：針對社群貼文冗長且夾雜無關生活敘事的特性，改採「段落」為單位。
2.  **初始特徵優化**：在監督式階段引入 SupCon Loss，優化特徵空間幾何結構，提升初始偽標籤品質。
2.  **多層次自訓練策略**：* 提出一種多層次的偽標籤篩選架構，將未標示樣本依信賴度劃分為三層：
        **低可信偽標籤資料集合**、**不明確偽標籤資料集合**、 **可信偽標籤資料集合**。
        結合動態門檻調整機制，讓樣本能隨著迭代過程依序晉升。此設計允許模型利用N池進行淺層邊界探索，同時利用G池穩定特徵核心，實現了從「廣度探索」到「深度確立」的漸進式學習，有效解決傳統自訓練中雜訊容忍度不足的問題。

---

## 🛠️ 環境建置 (Installation)

本專案基於 Python 3.8+ 與 PyTorch。建議使用 Conda 進行環境管理。

### 1. 建立虛擬環境
```bash
conda create -n mst_env python=3.8
conda activate mst_env
```

### 2. 專案結構
```
├── MSL.py                  # 主程式：包含模型定義、訓練迴圈 (SupCon + MST)
├── checkpoints/            # 模型權重儲存區 (自動建立)
│   ├── supcon_classifier.pt  # 監督式階段訓練完成的模型
│   └── mst_final.pt          # 自訓練結束後的最終模型
├── data/                   # 資料集放置區 (需自行準備 CSV)
│   ├── mental_health/
│   │   ├── train_segments.csv             # 標註訓練集
│   │   ├── test_segments.csv              # 段落測試集
│   │   ├── unlabeled_segments.csv         # 未標註資料集
│   │   └── test_text.csv                  # 文本測試集
└── function/               # 輔助函式庫
    ├── __init__.py
    └── preprocess.py       # 包含 infer_probs, evaluate_posts 等評估工具

```

## -----執行指令-----

### 監督式
    python MSL.py \
        --do_supervised 

### 半監督
    python MSL.py --do_msl \
        --load_supervised_checkpoint checkpoints/supcon_classifier.pt \

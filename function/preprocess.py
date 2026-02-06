import re
import math
from typing import List, Tuple, Dict
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# --- 你給的切段函式（我稍微修正了一些細節） ---
def split_content(text: str, min_seg_len: int = 100) -> List[str]:
    """
    依據空格及中文標點符號將文本切分成片段，
    再累積片段直到累積長度達到 min_seg_len，作為一個分段輸出。
    - min_seg_len 以字元數計（中文一字一碼）
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    # 移除英文字母（如你原本的需求）
    text = re.sub(r'[A-Za-z]', '', text)
    # 切分標點（保留分隔符）
    pattern = r'([。！？；，、,.\s])'
    parts = re.split(pattern, text)
    # 合併分隔符回前一片段
    fragments = []
    current_frag = ""
    for part in parts:
        if part == "":
            continue
        if re.match(pattern, part):
            current_frag += part
            if current_frag.strip():
                fragments.append(current_frag.strip())
            current_frag = ""
        else:
            current_frag += part
    if current_frag.strip():
        fragments.append(current_frag.strip())

    # 累積 fragments 直到長度達到 min_seg_len
    segments = []
    current_segment = ""
    for frag in fragments:
        candidate = (current_segment + " " + frag).strip() if current_segment else frag
        if len(candidate) < min_seg_len:
            current_segment = candidate
        else:
            segments.append(candidate.strip())
            current_segment = ""
    # 若最後 remainder 不為空，也當作一個 segment（避免遺漏）
    if current_segment.strip():
        segments.append(current_segment.strip())
    return segments

# --- 批次化 segment 進行推論（使用你的 model 與 tokenizer） ---
@torch.no_grad()
def predict_segments(model, tokenizer, segments: List[str], device: str, batch_size: int = 64, max_length: int = 512) -> np.ndarray:
    """
    回傳每個 segment 的正類機率（shape = (len(segments),)）
    """
    model.eval()
    probs = []
    # simple batching
    for i in range(0, len(segments), batch_size):
        batch_texts = segments[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        

        # 兼容三種返回格式
        if isinstance(out, dict):
            # ContrastiveClassifier 返回字典 {"logits": ..., "features": ...}
            logits = out["logits"]
        elif isinstance(out, tuple):
            # EncoderHead (MSL.py) 返回 tuple (logits, z)
            logits = out[0]
        else:
            # BCEClassifier 直接返回張量
            logits = out

        # logits = out["logits"]

        # For binary classification, get probability of positive class (index 1)
        # p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        
        # 🔍 根據 logits 維度自動判斷分類類型
        if logits.dim() == 1 or logits.shape[-1] == 1:
            # 二分類 BCE 格式: (batch_size,) 或 (batch_size, 1)
            if logits.dim() == 2:
                logits = logits.squeeze(-1)
            p = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            # 多分類 CE 格式: (batch_size, num_classes)
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        probs.append(p)
    if len(probs) == 0:
        return np.array([])
    return np.concatenate(probs, axis=0)

# --- 從 segments 的機率聚合到 post-level ---
def aggregate_post_from_segment_probs(seg_probs: np.ndarray, threshold: float = 0.5, agg_mode: str = "any") -> int:
    """
    agg_mode:
      - "any": 若任一 segment prob >= threshold -> post positive
      - "max": 使用 max(seg_probs) >= threshold（等同 any）
      - "topk": 若 top-k 中有 >= threshold（可擴充）
    回傳 1 (positive) 或 0 (negative)
    """
    if seg_probs.size == 0:
        return 0
    if agg_mode in ("any", "max"):
        return int(np.max(seg_probs) >= threshold)
    # 可擴充其他聚合策略
    raise ValueError("Unknown agg_mode")

# --- 針對 test dataframe 的 batch processing（每篇貼文） ---
def evaluate_posts(model, tokenizer, df_test, text_col="segment", label_col="label",
                   min_seg_len: int = 18, threshold: float = 0.5,
                   device: str = "cuda", seg_batch_size: int = 64, max_length: int = 512,
                   verbose: bool = False) -> Dict:
    """
    df_test: DataFrame with one row per post, columns: text_col (full post), optional label_col (0/1)
    回傳 dict 包含 per-post predictions 與評估指標（若有 label）
    """
    posts = df_test[text_col].astype(str).tolist()
    golds = df_test[label_col].tolist() if (label_col in df_test.columns) else None

    all_post_preds = []
    all_post_probs = []  # store the max segment prob per post (for analysis)
    for idx, post in enumerate(posts):
        segments = split_content(post, min_seg_len=min_seg_len)
        if len(segments) == 0:
            # empty post -> predict negative
            all_post_preds.append(0)
            all_post_probs.append(0.0)
            continue
        seg_probs = predict_segments(model, tokenizer, segments, device=device,
                                     batch_size=seg_batch_size, max_length=max_length)
        post_pred = aggregate_post_from_segment_probs(seg_probs, threshold=threshold, agg_mode="any")
        all_post_preds.append(post_pred)
        all_post_probs.append(float(np.max(seg_probs) if seg_probs.size>0 else 0.0))

        if verbose and (idx % 100 == 0):
            print(f"Processed {idx}/{len(posts)} posts; segments={len(segments)}; top_prob={all_post_probs[-1]:.4f}")

    results = {"preds": all_post_preds, "probs": all_post_probs}

    if golds is not None:
        y_true = np.array(golds, dtype=int)
        y_pred = np.array(all_post_preds, dtype=int)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1], average=None)
        macro = precision_recall_fscore_support(y_true, y_pred, average="macro")
        rpt = classification_report(y_true, y_pred, digits=4)
        results.update({
            "accuracy": acc,
            "precision_per_class": p.tolist(),
            "recall_per_class": r.tolist(),
            "f1_per_class": f1.tolist(),
            "macro": macro,
            "report": rpt
        })
    return results

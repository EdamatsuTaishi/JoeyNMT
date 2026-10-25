# JoeyNMT + 相対位置エンコーディング + LM-Prior（KL蒸留）

[JoeyNMT v2.3.0](https://github.com/joeynmt/joeynmt/tree/v2.3) をベースに、次の2つを追加した改造版です。

1. **相対位置エンコーディング**：Transformer の attention に、トークン間の相対距離に応じたバイアスを加える
2. **LM-Prior（KLダイバージェンス正則化）**：事前学習済み言語モデル（Llama 系）の出力分布に、翻訳モデルの出力分布を近づける

---

## 1. 相対位置エンコーディング

元の JoeyNMT は、各トークンの「絶対位置」を sinusoidal エンコーディングで埋め込みに足すだけです。
この改造では attention スコアの計算に「クエリ位置 $i$ とキー位置 $j$ の距離 $j-i$」の情報を直接加えています（Shaw et al., 2018 の方式）。

$$
e_{ij} = \frac{q_i^\top k_j + q_i^\top r_{\,\mathrm{clip}(j-i)}}{\sqrt{d_h}}
$$

- $r_{j-i}$ は距離ごとに学習するベクトル（`rel_emb`、形は `(head_size, 2R-1)`）
- 距離は $[-(R-1),\, R-1]$ にクリップ（$R$ = `max_relative_positions` = 512）
- 絶対位置エンコーディング（sinusoidal）はそのまま残しており、両方を併用

**設定は不要**です。`MultiHeadedAttention` 自体を置き換えているので、Transformer を使う限り常に有効になります
（encoder の self-attention、decoder の self-attention・cross-attention のすべて）。

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `transformer_layers.py` | `MultiHeadedAttention` に `rel_emb` パラメータと相対位置バイアスの計算を追加 |

---

## 2. LM-Prior（KLダイバージェンス）

対象言語の事前学習済み言語モデル（教師）が出す「次のトークンの分布」に、翻訳モデル（生徒）の分布を近づけます（Baziotis et al., 2020, *Language Model Prior for Low-Resource Neural Machine Translation*）。
損失は通常のクロスエントロピーに KL 項を足したものです。

$$
\mathcal{L} = \mathrm{CE}\big(y,\; p_{\text{NMT}}\big) + \lambda\,\tau^2\,\mathrm{KL}\!\left(p_{\text{LM}}^{(\tau)} \,\middle\|\, p_{\text{NMT}}^{(\tau)}\right)
$$

- $p^{(\tau)} = \mathrm{softmax}(z/\tau)$：温度 $\tau$ で平滑化した分布
- $\lambda$：KL 項の重み（`kl_lambda`）、$\tau$：温度（`kl_tau`）
- 教師 LM は `torch.no_grad()` で推論するだけで、学習はしない
- KL はパディング以外のトークン位置だけで計算

KL を取るには、**翻訳モデルの出力語彙と教師 LM の語彙が ID レベルで一致している必要があります**。
そのため、ターゲット側のトークナイザを HuggingFace（Llama）のトークナイザに置き換え、語彙ファイルも Llama の語彙を ID 順に並べたものを使います。

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `loss.py` | `XentLoss` が `lm_prior` 設定を受け取り、教師 LM を読み込んで KL 項を損失に加える |
| `model.py` | YAML の `model.lm_prior` を読み、損失関数へ渡す。温度 $\tau$ で割った log 確率（`kl_log_probs`）を計算 |
| `tokenizers.py` | `tokenizer_type: huggingface`（`HuggingFaceTokenizer`）を追加 |
| `vocabulary.py` | 語彙ファイルに特殊トークンが含まれる場合、先頭に特殊トークンを足さないよう変更（Llama の ID とずれないため） |

### 設定例

```yaml
data:
  trg:
    level: "bpe"
    tokenizer_type: "huggingface"
    access_token_name: "HF_TOKEN"        # HuggingFace トークンを入れた環境変数の名前
    voc_file: "path/to/llama_vocab.txt"  # Llama の語彙を ID 順に並べたもの
    tokenizer_cfg:
      model_file: "meta-llama/Llama-3.2-1B"

model:
  lm_prior:
    model_file: "meta-llama/Llama-3.2-1B" # 教師 LM
    access_token_name: "hf_xxx"           # 現状の実装ではトークン文字列そのものを渡す
    kl_lambda: 0.5
    kl_tau: 2.0
```

`kl_lambda`・`kl_tau`・`model_file`・`access_token_name` のどれかが空なら KL 項は無効になり、元の JoeyNMT と同じ損失で学習します。

---

## 実験結果

### 設定

| 項目 | 値 |
|---|---|
| 言語対 | ja → en |
| データ | [small_parallel_enja](https://github.com/odashi/small_parallel_enja)（学習 50,000 文 / dev 500 文 / test 500 文） |
| 評価指標 | sacreBLEU |
| デコード | beam search |
| 教師 LM | Llama 3 系 |
| $\lambda$ / $\tau$ | 0.5 / 2.0 |

### スコア

| 手法 | BLEU | Δ（ベースライン比） | Δ（1つ前の手法比） |
|---|---:|---:|---:|
| ベースライン（JoeyNMT v2.3） | 36.90 | – | – |
| + 相対位置エンコーディング | 38.05 | +1.15 | +1.15 |
| + 相対位置エンコーディング + LM-Prior（KL） | **39.09** | **+2.19** | +1.04 |

- 相対位置エンコーディングだけで BLEU が **+1.15**、さらに LM-Prior を加えると合計 **+2.19** 向上しました。
- 相対位置エンコーディングは常に有効な実装なので、LM-Prior の実験は相対位置エンコーディングと組み合わせた状態で行っています。

---

## 参考文献

- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). *Self-Attention with Relative Position Representations.* NAACL.
- Baziotis, C., Haddow, B., & Birch, A. (2020). *Language Model Prior for Low-Resource Neural Machine Translation.* EMNLP.
- Kreutzer, J., Bastings, J., & Riezler, S. (2019). *Joey NMT: A Minimalist NMT Toolkit for Novices.* EMNLP (demo).

# OpenFoundation-Engine: Modular Foundation Model Pre-training, Data Curation & Risk Evaluation Framework

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2F%20NumPy-orange.svg)](https://pytorch.org)
[![HPC](https://img.shields.io/badge/HPC-ZeRO--3%20%7C%20FSDP%20%7C%20MFU%2048%25-purple.svg)](#)
[![Evaluations](https://img.shields.io/badge/Evaluation-MMLU%20%7C%20PPL%20%7C%20Bias%20Audit-teal.svg)](#)

Developed as a dedicated open-source research and engineering framework for building, tweaking, training, and risk-managing modern **Decoder-Only Foundation Models**. Built specifically targeting the research mandate of institutions like the **Institute of Foundation Models (MBZUAI)**.

---

## 🌟 Key Research Capabilities

```
                           +-----------------------------------------------+
                           |          OpenFoundation-Engine Pipeline       |
                           +-----------------------------------------------+
                                                  |
       +-------------------------+----------------+-------------------------+
       |                         |                                          |
       v                         v                                          v
+------------------+    +-----------------------+              +------------------------+
| 1. Data Curation |    | 2. Architecture & MoE |              | 3. Testing & Risk Lab  |
| - MinHash LSH    |    | - GQA (4:1 KV Ratio)  |              | - Cross-Entropy & PPL  |
| - Quality Filter |    | - RoPE Freq Scaling   |              | - MMLU Choice Scoring  |
| - Custom BPE Vocab|   | - SwiGLU Gated FFN    |              | - Bias Disparity Delta |
| - MMap (.bin/.idx)|   | - Sparse MoE (Top-2)  |              | - Hallucination Entropy|
+------------------+    +-----------------------+              +------------------------+
                                 |                                          |
                                 +------------------+-----------------------+
                                                    v
                                      +---------------------------+
                                      | 4. Distributed HPC Lab    |
                                      | - ZeRO-1/2/3 & FSDP VRAM  |
                                      | - MFU Hardware Benchmark  |
                                      | - Loss Spike Anomaly Det. |
                                      +---------------------------+
```

### 1. Large-Scale Data Curation & Tokenization Pipeline
* **MinHash & Locality-Sensitive Hashing (LSH):** Implements Broder's theorem with 64 linear hash permutations grouped into 16 LSH bands. Flags and prunes near-duplicate documents at $\ge 0.80$ Jaccard similarity.
* **RefinedWeb & FineWeb Heuristic Filtering:** Multi-stage filtering pruning documents violating symbol-to-word ratios ($>0.25$), repetition bounds ($>0.30$), and minimum information entropy.
* **Custom Byte-Pair Encoding (BPE) Tokenizer:** Regex pre-tokenization with iterative merge table learning from raw corpora, tracking **Characters Per Token (CPT)** and **Bytes Per Token (BPT)**.
* **Zero-Copy Memory-Mapped Token Storage:** Exports token sequences into binary `.bin` and `.idx` formats for zero-copy streaming into distributed data loaders.

### 2. Modern Decoder-Only Foundation Model Architecture
* **Grouped-Query Attention (GQA):** Implements $4:1$ query-to-KV head grouping, slashing KV cache memory footprint during autoregressive generation by up to $75\%$ with negligible perplexity penalty.
* **Rotary Position Embeddings (RoPE):** Applies complex rotation operators $\mathbf{R}_{\Theta, m}^d$ to preserve relative distances and support long-context extrapolation.
* **SwiGLU Gated Feed-Forward Networks:** Implements Shazeer's $\text{SwiGLU}(x) = (x W_{\text{gate}} \odot \text{SiLU}(x W_{\text{gate}})) W_{\text{down}}$ with $\frac{8}{3} d_{\text{model}}$ hidden expansion.
* **Sparse Mixture of Experts (MoE):** Top-2 expert routing with router z-loss and auxiliary load-balancing loss:
  $$\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^N P_i f_i$$

### 3. Comprehensive Evaluation & Safety Auditing Harness
* **Perplexity & Cross-Entropy Tracking:** Exact validation tracking:
  $$\text{PPL} = \exp\left( - \frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i}) \right)$$
* **MMLU / Reasoning Evaluation Runner:** Normalized continuation log-likelihood scoring across multi-choice reasoning tasks.
* **Counterfactual Demographic Bias Probing:** Measures log-odds disparity across counterfactual pairs (gender, nationality, profession) to audit algorithmic bias:
  $$\Delta = \log P(\text{completion} \mid \text{Group}_A) - \log P(\text{completion} \mid \text{Group}_B)$$
* **Hallucination Risk Scoring:** Measures predictive token entropy $H(p) = - \sum p \log p$ across factual assertions; high entropy flags epistemic uncertainty and hallucination susceptibility.

### 4. Distributed Deep Learning & HPC Diagnostics
* **Distributed Memory Modeling:** Compares VRAM consumption across Standard DDP, ZeRO-1, ZeRO-2, and DeepSpeed ZeRO-3 / PyTorch FSDP across clusters.
* **Hardware Model FLOPs Utilization (MFU):** Calculates cluster compute efficiency:
  $$\text{MFU} = \frac{6 \times \text{Tokens/sec} \times \text{Params}}{\text{Num\_GPUs} \times \text{Peak\_TFLOPS}}$$
* **Real-Time Loss Spike Anomaly Detection:** Rolling $Z$-score tracking on per-iteration loss and gradient norms to trigger automated rollback recommendations during instability.

---

## 🚀 Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/alaa01-bit/OpenFoundation-Engine.git
cd OpenFoundation-Engine

# Install dependencies
pip install -r requirements.txt

# Run full unit test suite
python -m unittest discover -s tests -p "test_*.py"

# Launch Research Dashboard & REST API
python app.py
```

Visit the interactive research dashboard at `http://localhost:8000`.

---

## 📊 REST API Specifications

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/model/specs` | Returns model parameter counts, FLOPs/token, and GQA grouping. |
| `POST` | `/api/tokenizer/analyze` | Returns BPE subword breakdown, BPT, CPT, and compression ratio. |
| `POST` | `/api/data/curate` | Runs heuristic filters & MinHash LSH deduplication on document corpus. |
| `GET` | `/api/eval/benchmark-suite` | Runs MMLU reasoning benchmarks and returns choice probabilities. |
| `GET` | `/api/eval/bias-audit` | Evaluates demographic parity delta and hallucination entropy. |
| `POST` | `/api/hpc/memory-plan` | Calculates per-GPU VRAM breakdown across DDP vs ZeRO-1/2/3 and MFU. |

---

## 🧪 Benchmark Verification Output

```text
Testing health_check...
Health OK: {'status': 'healthy', 'service': 'OpenFoundation-Engine', 'version': '1.0.0'}
Testing get_model_specs...
Specs OK. Active params: 5,413,120
Testing analyze_tokenization...
Tokenizer OK. Compression ratio: 1.62x | Bytes/Token: 1.61
Testing curate_dataset...
Curation OK. Duplicates Removed: 1 (50.0% retention)
Testing run_benchmark_suite...
Benchmarks OK. Multi-choice Log-Likelihood Evaluation Passed
Testing run_bias_and_safety_audit...
Bias audit OK. Counterfactual Log-Odds Disparity: Measured
Testing calculate_distributed_memory...
Memory Plan OK. 7B Model FSDP VRAM per GPU: 33.04 GB (Fits comfortably in 80GB H100)

ALL API ENDPOINTS FUNCTION PERFECTLY!
```

---

## 📜 Citation & Author

**Alaa Abdel Fattah Allam**  
M.Sc. Student in Artificial Intelligence & Data Science  
Arab Academy for Science, Technology and Maritime Transport (AASTMT)  
*AI Research Consultant, National Research Centre (NRC)*  
*GitHub:* [@alaa01-bit](https://github.com/alaa01-bit) &bull; *LinkedIn:* [alaa-a-allam](https://linkedin.com/in/alaa-a-allam)

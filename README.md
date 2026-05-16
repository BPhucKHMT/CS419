# Truy Xuất Thông Tin (CS419) – Đồ án Cuối Kỳ

**Bộ dữ liệu:** Cranfield (1400 Documents, 225 Queries)

## 1. Giới thiệu
Dự án triển khai và đánh giá các mô hình Truy xuất thông tin (Information Retrieval) từ con số không (from scratch) trên bộ dữ liệu Cranfield. Các công thức toán học cho việc tính toán điểm số (Scoring) được cài đặt thủ công để hiểu rõ bản chất bên trong của từng mô hình thay vì sử dụng thư viện có sẵn.

## 2. Pipeline (Workflow)
Hệ thống được thiết kế theo dạng pipeline tuần tự:

```text
[Dataset: Cranfield] 
       ↓
1. Tiền xử lý (Preprocessing)
   - Chuyển chữ thường, mở rộng từ viết tắt (Regex)
   - Tokenization & Lọc Stopwords
   - Stemming bằng thuật toán Snowball
       ↓
2. Lập chỉ mục (Indexing)
   - Xây dựng từ điển (Vocabulary: 4452 terms)
   - Tạo Inverted Index để tăng tốc độ truy hồi
       ↓
3. Tính điểm & Xếp hạng (Scoring & Ranking)
   - Vector Space Model (TF-IDF Weighting: ntc.ntc)
   - Okapi BM25 (Manual Implementation)
       ↓
4. Đánh giá hệ thống (Evaluation)
   - Độ đo: MAP, P@20, Recall@20
   - Thực hiện trên toàn bộ 225 câu truy vấn
```

## 3. Các Mô hình Đã Cài Đặt
Toàn bộ logic tính toán được đóng gói dạng module trong thư mục `src/` và được trình diễn trực quan trong `final_project_cs419.ipynb`.

### 3.1 Vector Space Model (VSM)
Mỗi tài liệu và câu truy vấn được biểu diễn thành **vector số thực** trong không gian từ vựng. Điểm truy hồi là **Cosine Similarity** giữa vector query và vector tài liệu.

**Cách tính trọng số cho từ trong Tài liệu:**

$$w_{t,d} = \underbrace{\frac{tf_{t,d}}{|d|}}_{\text{tần suất chuẩn hóa}} \times \underbrace{\ln\!\left(\frac{N}{df_t}\right)}_{\text{IDF}}$$

- $tf_{t,d}$: số lần từ $t$ xuất hiện trong tài liệu $d$
- $|d|$: tổng số token trong tài liệu $d$ (dùng để chuẩn hóa độ dài)
- $N = 1400$: tổng số tài liệu
- $df_t$: số tài liệu chứa từ $t$ (từ càng hiếm → IDF càng cao → trọng số càng lớn)

**Cách tính trọng số cho từ trong Câu truy vấn:** tương tự như tài liệu — cũng nhân thêm IDF, không chỉ đếm số lần xuất hiện.

$$w_{t,q} = \frac{tf_{t,q}}{|q|} \times \ln\!\left(\frac{N}{df_t}\right)$$

Sau cùng, cả hai vector được **chuẩn hóa L2** (chia cho độ dài vector) để điểm số Cosine nằm trong khoảng $[0, 1]$, không bị ảnh hưởng bởi độ dài tài liệu.

### 3.2 Okapi BM25
Triển khai thủ công công thức xác suất BM25. Tham số được Tuning cẩn thận trên tập Cranfield.

$$\text{score}(d,q) = \sum_{t \in q} \ln\!\left(\frac{N - df_t + 0.5}{df_t + 0.5} + 1\right) \cdot \frac{f_{t,d} \cdot (k_1+1)}{f_{t,d} + k_1\!\left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

- Tham số tối ưu: **`k1 = 2.0`**, **`b = 0.6`**
- `avgdl ≈ 91` token/tài liệu

## 4. Kết quả Đánh giá (Final Results)
Đánh giá trên toàn bộ 225 Queries. MAP tính trên top-100, P@20 và Recall@20 trên top-20:

| Mô hình | MAP@100 | P@20 | Recall@20 |
|---------|:-------:|:----:|:---------:|
| VSM (TF-IDF) Baseline | 0.2864 | 0.1573 | 0.5048 |
| **VSM + Cluster Reranking** | **0.2973** | **0.1660** | **0.5305** |
| BM25 Baseline | 0.3060 | 0.1622 | 0.5182 |
| **BM25 + Cluster Reranking** | **0.3219** | **0.1720** | **0.5500** |

**Nhận xét:**
- Cluster Reranking cải thiện **tất cả 3 metrics** cho cả VSM và BM25.
- BM25 + Cluster đạt MAP@100 = **0.3219** (+5.2% so với BM25 baseline).
- Recall@20 tăng từ 0.5182 lên **0.5500** — tìm được thêm ~3% tài liệu liên quan trong top-20.

## 5. Cluster-based Reranking

### Ý tưởng
Sau khi retrieval trả về top-K docs, các tài liệu cùng chủ đề thường cụm lại trong cùng cluster. Ta dùng thông tin cluster để **boost nhẹ** điểm của các tài liệu thuộc cluster được top-docs vote nhiều nhất.

### Pipeline

```text
Query
  │
  ▼
[1] Retrieve top-100 docs (BM25 hoặc VSM)
  │
  ▼
[2] Xác định cluster quan trọng
    - Lấy top-20 docs từ kết quả retrieval
    - Đếm xem các docs này thuộc cluster nào (vote)
    - Normalize cluster score = count / max_count ∈ [0, 1]
  │
  ▼
[3] Hybrid reranking
    final_score = α × norm(retrieval_score) + (1-α) × cluster_score
    với α = 0.85 (giữ 85% tín hiệu retrieval gốc)
  │
  ▼
Top-20 kết quả sau reranking
```

### Cấu hình tối ưu

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `n_components` | 100 | SVD giảm chiều TF-IDF → 100D LSA space |
| `N_CLUSTERS` | 200 | ~7 docs/cluster → micro-topic precision cao |
| `retrieve_k` | 100 | Số docs retrieval ban đầu để rerank |
| `top_cluster_docs` | 20 | Số docs đầu dùng để vote cluster |
| `alpha` | 0.85 | Trọng số giữ ranking gốc (không override hoàn toàn) |

### Tại sao N_CLUSTERS = 200 hiệu quả?
Cluster nhỏ (~7 docs) tạo **micro-topic**: các tài liệu trong cùng cluster cực kỳ gần nhau về ngữ nghĩa. Khi top-docs vote vào cluster này, boost điểm cho các tài liệu cùng micro-topic → tăng topical coverage mà không làm nhiễu ranking chất lượng cao của BM25/VSM.



## 6. Các File CSV Xuất Ra
Notebook tự động sinh các file CSV để hỗ trợ phân tích và báo cáo:

| File | Nội dung |
|------|----------|
| `tfidf_terms.csv` | Toàn bộ từ vựng kèm `Total_TF`, `DF`, `IDF`, `Max_TFIDF` — dùng để kiểm tra trọng số của từng term |
| `term_bm25.csv` | Toàn bộ từ vựng kèm `DF`, `IDF_BM25`, `Max_TF`, `Max_BM25_Score` — xem mức độ phân biệt của từng term theo BM25 |
| `query_vsm.csv` | Phân rã điểm Cosine Similarity theo từng term cho Top 20 tài liệu của 225 queries — gồm `W_query`, `W_doc`, `Dot_Product_Contrib`, `Query_Norm`, `Doc_Norm` |
| `query_bm25.csv` | Phân rã điểm BM25 theo từng term cho Top 20 tài liệu của 225 queries — gồm `TF(f)`, `IDF_BM25`, `Numerator`, `Denominator`, `Term_BM25_Score` |
| `evaluation_per_query.csv` | Điểm AP, P@20, Recall@20 của **từng query riêng lẻ** cho cả VSM và BM25 — dùng để phân tích best/worst case |

## 7. Hướng dẫn sử dụng

```bash
# Cài đặt môi trường
conda activate cs419

# Mở notebook dự án
jupyter notebook final_project_cs419.ipynb

# Hoặc chạy pipeline qua terminal
python main.py
```

> **Lưu ý:** Notebook đính kèm các ô Markdown giải thích cách **tính tay chi tiết** (Preprocessing, TF-IDF, BM25, MAP) dựa trên Query 1 thực tế. Các ví dụ này có thể copy/paste trực tiếp vào báo cáo học thuật.

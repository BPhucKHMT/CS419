# BÁO CÁO DỰ ÁN HỌC THUẬT: HỆ THỐNG TRUY XUẤT THÔNG TIN TRÊN BỘ DỮ LIỆU CRANFIELD

Môn học: **Truy xuất thông tin (CS419)**  
Cơ sở dữ liệu thực nghiệm: **Cranfield Dataset** (1.400 tài liệu, 225 câu truy vấn, 225 tệp kết quả liên quan)  

---

## 1. Giới thiệu Bài toán & Bộ dữ liệu
Dự án tập trung nghiên cứu, thiết lập và đánh giá các mô hình **Truy xuất thông tin (Information Retrieval - IR)** từ con số không (*from scratch*). Mục tiêu chính là cài đặt thủ công các công thức tính điểm (Scoring) để hiểu rõ cơ chế toán học và thiết kế cấu trúc dữ liệu tối ưu cho quá trình tìm kiếm.

Bộ dữ liệu **Cranfield** được sử dụng làm ngữ liệu chuẩn để kiểm thử hệ thống. Bộ dữ liệu này gồm:
* **1.400 tài liệu** tóm tắt các nghiên cứu về khí động học.
* **225 câu truy vấn** dạng ngôn ngữ tự nhiên.
* **Tệp liên quan (Relevance Judgments)** chứa danh sách các tài liệu thực sự liên quan đến từng câu truy vấn do chuyên gia đánh giá.

Toàn bộ quá trình thực nghiệm được triển khai trực quan trong tệp [final_project_cs419.ipynb](file:///c:/Users/phucnlb/cs419/CS419/final_project_cs419.ipynb) với hai mô hình chính là: **Vector Space Model (VSM)** và **Okapi BM25**.

---

## 2. Quy trình Tiền xử lý Văn bản (Text Preprocessing)
Để chuẩn hóa văn bản trước khi lập chỉ mục và tìm kiếm, hệ thống triển khai một Pipeline tiền xử lý tuần tự gồm 5 bước nhằm làm sạch nhiễu và đồng nhất các biến thể từ:

```
Văn bản thô (Raw Text)
   │
   ▼
1. Lowercase & Loại bỏ ký tự đặc biệt (Chuyển thành chữ thường, thay thế gạch ngang '-' bằng khoảng trắng)
   │
   ▼
2. Mở rộng viết tắt (Áp dụng từ điển Regex để chuẩn hóa các từ như fig. -> figure, eq. -> equation, ...)
   │
   ▼
3. Chuyển đổi số thành chữ (Nhận diện số nguyên/thập phân và chuyển sang chữ tiếng Anh bằng thư viện num2words)
   │
   ▼
4. Tokenization & Lọc Stopwords (Tách từ bằng word_tokenize, loại bỏ từ dừng tiếng Anh tiêu chuẩn + chữ số La Mã)
   │
   ▼
5. Stemming (Đưa các từ về dạng gốc bằng thuật toán Snowball Stemmer)
   │
   ▼
Danh sách Token chuẩn hóa (Processed Tokens)
```

### Chi tiết các bước trong Pipeline:
* **Từ điển viết tắt (`ABBREVIATIONS`)**:
  ```python
  {
      r'\bfig\.?\b':    'figure',
      r'\bref\.?\b':    'reference',
      r'\bapprox\.?\b': 'approximately',
      r'\beq\.?\b':     'equation',
      r'\bsq\.?\b':     'square',
      r'\bno\.?\b':     'number',
      r'\be\.g\.?\b':  'for example',
      r'\bi\.e\.?\b':  'that is',
      r'\bsec\.?\b':    'section',
  }
  ```
* **Lọc từ dừng bổ sung (`CUSTOM_STOPWORDS`)**: Loại bỏ các chữ số La Mã thường xuất hiện trong tiêu đề chương mục như `ii, iii, iv, vi, vii, viii, ix, xi, xii`.
* **Quy tắc lọc từ ngắn**: Loại bỏ tất cả các token có chiều dài $\le 2$ ký tự nếu chúng là ký tự chữ cái (nhằm loại bỏ các từ vô nghĩa như `a`, `an`, `by`, `at`, `on`,...).

---

## 3. Lựa chọn Từ khóa (Term Selection)
Lựa chọn term quyết định trực tiếp đến chất lượng biểu diễn tài liệu. Mục tiêu của bước này là giữ lại các từ mang nội dung chính, đồng thời loại bỏ các token ít giá trị để giảm kích thước vocabulary và tăng tốc truy vấn.

### 3.1. Loại Term được chọn & Lý giải lựa chọn
Dự án chọn **từ đơn (unigram/word)** làm đơn vị term chính.

Lý do lựa chọn:

1. Bộ dữ liệu Cranfield là tiếng Anh nên ranh giới từ khá rõ, phù hợp với tách từ theo word-level.
2. VSM và BM25 trong hệ thống đều hoạt động trên mô hình Bag-of-Words, nên dùng unigram giúp biểu diễn đơn giản, dễ lập chỉ mục và dễ giải thích điểm số.
3. N-gram có thể giữ được cụm từ tốt hơn, nhưng làm vocabulary tăng mạnh; concept-level lại cần thêm xử lý ngữ nghĩa phức tạp. Vì vậy, unigram là lựa chọn cân bằng giữa độ chính xác, tốc độ và khả năng trình bày.

### 3.2. Phương pháp Chọn Term
Hệ thống chọn term theo hai nhóm tiêu chí chính:

1. **Loại nhiễu và term ít phân biệt**
   * Loại bỏ stopwords như `the`, `is`, `an`, `of` vì các từ này xuất hiện quá phổ biến và gần như không giúp phân biệt tài liệu.
   * Loại bỏ token chữ cái có độ dài $\le 2$ vì thường là từ ngắn ít thông tin hoặc nhiễu.
   * Chuẩn hóa ký tự đặc biệt, gạch nối, chữ số và viết tắt để các term có cùng cách biểu diễn.

2. **Gom các biến thể hình thái về cùng một term**
   * Các biến thể như `wing`, `wings` hoặc `obeyed`, `obeying` được đưa về cùng một dạng gốc.
   * Hệ thống dùng **Snowball Stemmer** để tạo lớp tương đương hình thái, giúp query và document khớp nhau dù dùng biến thể từ khác nhau.

---

### 3.3. Thuật toán Chọn Term
Có thể mô tả thuật toán chọn term dưới dạng tổng quát như sau:

#### Mã giả thuật toán (Pseudocode):
```text
ALGORITHM TermSelection(DocumentText D, StopwordsList S, AbbreviationsDict A, Stemmer St)
INPUT: Văn bản thô D, Danh sách từ dừng S, Từ điển viết tắt A, Thuật toán Stemmer St
OUTPUT: Danh sách term đã chuẩn hóa

1. D_norm ← NormalizeText(D, A)
       - chuyển chữ thường
       - tách gạch nối thành khoảng trắng
       - mở rộng viết tắt
       - chuyển số thành chữ
       - loại ký tự đặc biệt

2. Tokens ← Tokenize(D_norm)

3. SelectedTerms ← []
4. For each token in Tokens:
       If IsValidToken(token, S):
           term ← St.stem(token)
           Append term vào SelectedTerms

5. Return SelectedTerms

FUNCTION IsValidToken(token, S):
    Return token không thuộc S
           AND (độ dài token > 2 OR token có chứa chữ số)
```

Tóm lại, thuật toán không chọn term bằng một công thức thống kê riêng như Chi-square hay Mutual Information. Với bài toán này, term được chọn bằng **pipeline lọc nhiễu + chuẩn hóa hình thái**, sau đó các mô hình VSM/BM25 mới dùng TF, DF và IDF để tính trọng số và xếp hạng.

---

### 3.4. Hiệu quả suy giảm kích thước từ vựng (Vocabulary Reduction)
Nhờ áp dụng thuật toán chọn term chặt chẽ, không gian từ vựng và kích thước tài liệu trung bình của hệ thống được tối ưu hóa:

| Đặc trưng thống kê | Trước xử lý (Raw Tokenize) | Sau xử lý (Processed) | Tỷ lệ giảm |
|--------------------|:-------------------------:|:--------------------:|:----------:|
| **Số lượng Term độc bản (Vocabulary Size)** | **7.472** | **4.588** | **38.60%** |
| **Độ dài trung bình tài liệu** | **161.91** từ | **95.58** từ | **40.97%** |

* **Tổng số term phân tích được từ toàn bộ tài liệu (Trước xử lý)**: **7.472 terms**.
* **Tổng số term được lựa chọn đưa vào từ điển (Sau tiền xử lý)**: **4.588 terms**.
* **Tỷ lệ chọn term (Selection Ratio)**: 
  $$\text{Tỷ lệ term được chọn} = \frac{4.588}{7.472} \approx 61.40\%$$
  *(Như vậy, hệ thống loại bỏ khoảng 38.60% các từ nhiễu/từ dừng không mang thông tin và giữ lại khoảng 61.40% gốc từ mang nội dung thực thụ)*.

---

### 3.5. Danh sách các Term được chọn tiêu biểu
Dưới đây là một số term tiêu biểu trong bộ từ vựng 4.588 term được chọn từ tập tài liệu Cranfield. Bảng không liệt kê toàn bộ vocabulary, mà chỉ chọn các term có tần suất cao hoặc mang ý nghĩa chuyên ngành rõ trong miền khí động học:

| Hạng | Term được chọn | Thừa số gốc (Gợi ý từ gốc) | Tổng tần suất (Total_TF) | Số tài liệu chứa (DF) | Trọng số IDF |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | **flow** | *flowing, flows, flow* | 2082 | 730 | 0.6512 |
| 2 | **pressur** | *pressure, pressures* | 1391 | 552 | 0.9307 |
| 3 | **boundari** | *boundary, boundaries* | 1216 | 470 | 1.0915 |
| 4 | **layer** | *layer, layers* | 1164 | 414 | 1.2184 |
| 5 | **heat** | *heating, heated, heat* | 847 | 306 | 1.5206 |
| 6 | **wing** | *wings, wing* | 837 | 226 | 1.8237 |
| 7 | **mach** | *mach* | 823 | 388 | 1.2832 |
| 8 | **shock** | *shock, shocks* | 746 | 240 | 1.7636 |
| 9 | **surfac** | *surface, surfaces* | 691 | 330 | 1.4451 |
| 10 | **temperatur** | *temperature, temperatures* | 629 | 268 | 1.6532 |

---

## 4. Mô hình Truy xuất & Cơ chế tính Trọng số Term

Báo cáo phân tích chi tiết cơ sở toán học và cơ chế tính trọng số, đồng thời làm rõ sự khác biệt giữa ba loại công thức thường dễ bị nhầm lẫn trong đồ án:

* **Công thức cơ sở**: trả lời câu hỏi "mô hình hiểu độ liên quan giữa tài liệu và câu truy vấn là gì?".
* **Công thức tính trọng số term**: trả lời câu hỏi "mỗi từ khóa đóng góp bao nhiêu điểm?".
* **Công thức xếp hạng**: trả lời câu hỏi "khi có query thật, hệ thống cộng điểm và sắp xếp tài liệu như thế nào?".

Nói ngắn gọn: **công thức cơ sở** là ý tưởng đo liên quan của mô hình, **công thức trọng số term** là cách tính điểm cho từng term, còn **công thức xếp hạng** là cách tổng hợp các trọng số đó để tạo ranking cuối cùng.

### 4.1. Biểu diễn Tài liệu và Câu truy vấn (Representation)
Cách thức biểu diễn thông tin đóng vai trò là đầu vào cho quá trình so khớp độ liên quan:
* **Trong mô hình Vector (VSM)**: Cả tài liệu $d$ và câu truy vấn $q$ đều được biểu diễn dưới dạng **Vector số thực** trong không gian từ vựng đa chiều:
  $$\vec{d} = \left(w_{1,d}, w_{2,d}, ..., w_{V,d}\right)$$
  $$\vec{q} = \left(w_{1,q}, w_{2,q}, ..., w_{V,q}\right)$$
  *(Trong đó $V=4588$ là chiều từ vựng, các thành phần là trọng số TF-IDF).*
* **Trong mô hình Probabilistic (Okapi BM25)**: Tài liệu được biểu diễn dưới dạng **Túi từ (Bag-of-Words)** đi kèm với độ dài thực tế của tài liệu $|d|$ để chuẩn hóa.

---

### 4.2. Vector Space Model (VSM) với TF-IDF

#### 1. Công thức cơ sở để tính độ liên quan giữa tài liệu và câu truy vấn
Trong mô hình VSM lý thuyết, độ tương đồng hay độ liên quan giữa tài liệu $d$ và câu truy vấn $q$ được đo lường bằng **Cosine Similarity**. Ý tưởng chính là: nếu vector tài liệu và vector truy vấn càng cùng hướng, tài liệu càng được xem là liên quan đến truy vấn.

$$\text{Similarity}(q, d) = \cos(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2} = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\sqrt{\sum_{t \in q} w_{t,q}^2} \cdot \sqrt{\sum_{t \in d} w_{t,d}^2}}$$

* **Tử số** $\vec{q} \cdot \vec{d}$: đo mức độ trùng khớp có trọng số giữa query và document.
* **Mẫu số** $\|\vec{q}\|_2 \|\vec{d}\|_2$: chuẩn hóa độ dài vector, giúp tài liệu dài không tự động có lợi thế chỉ vì chứa nhiều từ hơn.
* **Cách nói trên slide**: VSM đo độ gần nhau về hướng giữa vector query và vector document; score càng cao thì tài liệu càng liên quan.

#### 2. Công thức tính trọng số term
Sau khi đã biểu diễn tài liệu và câu truy vấn thành vector, cần xác định giá trị của từng chiều trong vector. Mỗi chiều ứng với một term, và trọng số term được tính bằng TF-IDF:

* **Trọng số từ trong Tài liệu ($w_{t,d}$)**:
  $$w_{t,d} = TF(t, d) \times IDF(t) = \frac{tf_{t,d}}{|d|} \times \ln\!\left(\frac{N}{df_t}\right)$$
* **Trọng số từ trong Câu truy vấn ($w_{t,q}$)**:
  $$w_{t,q} = TF(t, q) \times IDF(t) = \frac{tf_{t,q}}{|q|} \times \ln\!\left(\frac{N}{df_t}\right)$$
* **Giải thích thành phần**:
  * $tf_{t,d}$: số lần term $t$ xuất hiện trong tài liệu $d$.
  * $|d|$: độ dài tài liệu sau tiền xử lý.
  * $N$: tổng số tài liệu trong corpus.
  * $df_t$: số tài liệu có chứa term $t$.
  * **TF** cho biết term xuất hiện dày hay thưa trong một tài liệu.
  * **IDF** cho biết term đó hiếm hay phổ biến trong toàn bộ corpus.

* **Ý nghĩa trực quan**: Một term có trọng số cao khi nó xuất hiện đủ nhiều trong tài liệu đang xét, nhưng không xuất hiện quá phổ biến trong toàn bộ tập tài liệu. Ví dụ `shock`, `wing`, `aeroelast` thường có khả năng phân biệt tốt hơn các từ chung như `result`, `method`.

#### 3. Công thức xếp hạng tài liệu
Khi người dùng nhập query, hệ thống tạo vector TF-IDF cho câu truy vấn, sau đó tính Cosine Similarity giữa vector truy vấn và vector TF-IDF của từng tài liệu ứng viên. Công thức xếp hạng thực tế trong chương trình là:

$$\text{score}_{\text{VSM}}(q,d) = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\|\vec{q}\|_2 \cdot \|\vec{d}\|_2}$$

* **Cơ chế xếp hạng**:
  * $w_{t,q}$ là trọng số TF-IDF của term trong query.
  * $w_{t,d}$ là trọng số TF-IDF của term trong document.
  * Chỉ các term thuộc $q \cap d$ mới đóng góp điểm.
  * $\|\vec{q}\|_2$ và $\|\vec{d}\|_2$ được tính bằng `np.linalg.norm` khi gọi hàm `cosine_similarity`.
  * Tài liệu có Cosine Similarity cao hơn sẽ được xếp hạng cao hơn.

* **Cách nói trên slide**: VSM xếp hạng bằng cosine similarity đầy đủ giữa vector TF-IDF của query và vector TF-IDF của document. Trong code hiện tại, cả hai norm được tính trực tiếp lúc truy vấn, không chuẩn hóa vector trước.

---

### 4.3. Okapi BM25

#### 1. Công thức cơ sở để tính độ liên quan giữa tài liệu và câu truy vấn
Với Okapi BM25, độ liên quan giữa tài liệu $d$ và câu truy vấn $q$ được tính trực tiếp bằng tổng điểm của các term truy vấn xuất hiện trong tài liệu:

$$\text{Relevance}_{\text{BM25}}(d,q) = \sum_{t \in q \cap d} w_{\text{BM25}}(t,d)$$

* $q \cap d$: các term vừa xuất hiện trong query, vừa xuất hiện trong tài liệu.
* $w_{\text{BM25}}(t,d)$: điểm đóng góp của term $t$ cho tài liệu $d$.
* Tài liệu có nhiều term truy vấn quan trọng thì tổng điểm cao hơn.

#### 2. Công thức tính trọng số term
Trong BM25, trọng số đóng góp của mỗi term $t$ trong tài liệu $d$ được tính như sau:

$$w_{\text{BM25}}(t, d) = IDF_{BM25}(t) \cdot \frac{f_{t,d} \cdot (k_1+1)}{f_{t,d} + k_1\left(1 - b + b \cdot \dfrac{|d|}{avgdl}\right)}$$

* **Hàm IDF BM25**: Được điều chỉnh để tránh điểm số IDF bị âm khi từ khóa xuất hiện trong hơn 50% tài liệu:
  $$IDF_{BM25}(t) = \ln\!\left(\frac{N - df_t + 0.5}{df_t + 0.5} + 1\right)$$
* **Giải thích thành phần**:
  * $f_{t,d}$: số lần term $t$ xuất hiện trong tài liệu $d$.
  * $|d|$: độ dài tài liệu.
  * $avgdl$: độ dài trung bình của tài liệu trong corpus, ở thực nghiệm này là `95.58`.
  * $k_1 = 2.0$: điều khiển mức bão hòa tần suất. Khi term lặp lại nhiều lần, điểm vẫn tăng nhưng chậm dần.
  * $b = 0.6$: điều khiển mức chuẩn hóa độ dài tài liệu. Nếu $b$ càng lớn, hệ thống càng phạt tài liệu dài mạnh hơn.



#### 3. Công thức xếp hạng tài liệu
Khi có câu truy vấn $q$, hệ thống xếp hạng tài liệu dựa trên tổng điểm BM25 của các term truy vấn xuất hiện trong tài liệu:

$$\text{score}_{\text{BM25}}(d,q) = \sum_{t \in q \cap d} IDF_{BM25}(t) \cdot \frac{f_{t,d} \cdot (k_1+1)}{f_{t,d} + k_1\left(1 - b + b \cdot \dfrac{|d|}{avgdl}\right)}$$

* **Cơ chế xếp hạng**:
  * Với mỗi term trong query, hệ thống lấy posting list từ chỉ mục đảo ngược.
  * Với mỗi tài liệu trong posting list, hệ thống tính điểm term theo BM25.
  * Điểm cuối cùng của tài liệu là tổng điểm của tất cả term query mà tài liệu chứa.
  * Tài liệu có score cao hơn được xếp trước.

* **Cách nói trên slide**: BM25 cộng điểm từng term trong query; term hiếm và xuất hiện hợp lý trong tài liệu sẽ đóng góp nhiều điểm hơn, nhưng tài liệu dài bị chuẩn hóa để ranking công bằng hơn.

---

## 5. Cấu trúc Chỉ mục & Quá trình Lập/Truy xuất dữ liệu

Sau khi đã chọn term và xây dựng công thức tính điểm, bước tiếp theo là tổ chức dữ liệu sao cho truy vấn có thể được xử lý nhanh. Nếu mỗi query đều phải quét lại toàn bộ 1.400 tài liệu, hệ thống sẽ tốn nhiều thời gian. Vì vậy, đồ án sử dụng **chỉ mục đảo ngược (Inverted Index)**.

Ý tưởng của chỉ mục đảo ngược rất trực quan: thay vì hỏi *"mỗi tài liệu chứa những term nào?"*, hệ thống lưu theo hướng ngược lại: *"mỗi term xuất hiện trong những tài liệu nào?"*.

---

### 5.1. Cấu trúc dữ liệu cho chỉ mục

Chỉ mục đảo ngược lưu ánh xạ từ một term đến danh sách các tài liệu chứa term đó:

$$\text{term} \rightarrow \{nDoc: df_t,\ \text{postings}: [(doc\_id_1, value_1), (doc\_id_2, value_2), ...]\}$$

Ví dụ minh họa:

$$\text{flow} \rightarrow [(12, value), (51, value), (184, value), ...]$$

Trong đó, `value` phụ thuộc vào mô hình truy xuất:

* Với **VSM**, `value` là trọng số TF-IDF của term trong tài liệu.
* Với **BM25**, `value` là tần suất thô $f_{t,d}$ của term trong tài liệu.

Các thành phần chính của chỉ mục:

* $df_t$: số tài liệu chứa term $t$.
* `postings`: danh sách các tài liệu chứa term.
* `doc_id`: mã tài liệu trong tập Cranfield.
* `value`: trọng số hoặc tần suất của term trong tài liệu.
* `doc_len`: độ dài tài liệu, cần cho BM25.
* `avgdl`: độ dài trung bình của toàn bộ tập tài liệu, cần cho BM25.

Nhờ cấu trúc này, khi query chứa các term như `flow`, `wing`, `shock`, hệ thống chỉ cần lấy postings list của các term đó thay vì quét toàn bộ corpus.

---

### 5.2. Thuật toán lập chỉ mục

Lập chỉ mục là pha xử lý offline, được thực hiện trước khi người dùng truy vấn. Mục tiêu của pha này là biến tập tài liệu ban đầu thành các cấu trúc tra cứu nhanh.

#### 5.2.1. Quy trình tổng quát

```text
Documents
   ↓
Tiền xử lý từng tài liệu
   ↓
Xây dựng vocabulary
   ↓
Tính TF, DF, IDF
   ↓
Tạo chỉ mục VSM và BM25
```

Thuật toán:

1. Mỗi tài liệu được đưa qua pipeline tiền xử lý để thu được danh sách term chuẩn hóa.
2. Từ toàn bộ tài liệu, hệ thống xây dựng vocabulary gồm các term độc bản.
3. Với từng term, hệ thống đếm:
   * term xuất hiện bao nhiêu lần trong từng tài liệu;
   * term xuất hiện trong bao nhiêu tài liệu.
4. Từ các thống kê đó, hệ thống tính các giá trị cần thiết như TF, DF, IDF, độ dài tài liệu và độ dài trung bình.
5. Cuối cùng, hệ thống tạo postings list cho từng term.

#### 5.2.2. Chỉ mục cho VSM

Với VSM, mỗi tài liệu được biểu diễn bằng vector TF-IDF. Trọng số của term $t$ trong tài liệu $d$ là:

$$w_{t,d} = \frac{tf_{t,d}}{|d|} \times \ln\left(\frac{N}{df_t}\right)$$

Chỉ mục VSM lưu các term có trọng số khác 0:

$$t \rightarrow [(doc\_id, w_{t,d}), ...]$$

Ý nghĩa: nếu term `wing` xuất hiện trong tài liệu 51 với trọng số TF-IDF là $w_{\text{wing},51}$, postings list của `wing` sẽ chứa cặp:

$$({51}, w_{\text{wing},51})$$

Khi truy vấn, các trọng số này được dùng để tính Cosine Similarity giữa query vector và document vector.

#### 5.2.3. Chỉ mục cho BM25

Với BM25, hệ thống không cần lưu trọng số TF-IDF. Thay vào đó, postings list lưu tần suất thô của term trong từng tài liệu:

$$t \rightarrow [(doc\_id, f_{t,d}), ...]$$

Ngoài postings list, BM25 cần thêm độ dài tài liệu và độ dài trung bình:

$$\text{doc\_len}[d] = |d|$$

$$avgdl = \frac{\sum_{d \in D}|d|}{N}$$

$$IDF_{BM25}(t) = \ln\left(\frac{N - df_t + 0.5}{df_t + 0.5} + 1\right)$$

Những thông tin này giúp BM25 vừa xét mức độ xuất hiện của term trong tài liệu, vừa điều chỉnh ảnh hưởng của tài liệu quá dài.

---

### 5.3. Thuật toán xử lý câu truy vấn

Xử lý truy vấn là pha online, diễn ra khi người dùng nhập câu truy vấn. Điểm quan trọng là query phải được tiền xử lý bằng cùng pipeline với tài liệu, để các term trong query và trong chỉ mục có cùng dạng biểu diễn.

#### 5.3.1. Quy trình chung

```text
Query thô
   ↓
Tiền xử lý query
   ↓
Tra cứu postings list
   ↓
Tạo tập tài liệu ứng viên
   ↓
Tính điểm VSM hoặc BM25
   ↓
Sắp xếp giảm dần theo score
   ↓
Trả về top-k tài liệu
```

Thuật toán xử lý query:

1. Tiền xử lý query để thu được các term chuẩn hóa.
2. Với mỗi term trong query, lấy postings list tương ứng từ chỉ mục đảo ngược.
3. Hợp các tài liệu trong các postings list để tạo tập tài liệu ứng viên.
4. Tính điểm cho từng tài liệu ứng viên bằng VSM hoặc BM25.
5. Sắp xếp tài liệu theo điểm giảm dần và trả về top-k.

Ví dụ, nếu query sau xử lý có các term:

$$q = \{\text{aeroelast}, \text{model}, \text{heat}, \text{aircraft}\}$$

hệ thống sẽ lấy postings list của bốn term này, tạo danh sách tài liệu ứng viên, sau đó tính điểm xếp hạng cho từng tài liệu.

#### 5.3.2. Tính điểm với VSM

Với VSM, query cũng được biểu diễn thành vector TF-IDF:

$$w_{t,q} = \frac{tf_{t,q}}{|q|} \times \ln\left(\frac{N}{df_t}\right)$$

Mỗi tài liệu ứng viên đã có vector TF-IDF từ pha lập chỉ mục. Hệ thống tính độ tương đồng bằng Cosine Similarity:

$$\text{score}_{\text{VSM}}(q,d) = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\|\vec{q}\|_2 \cdot \|\vec{d}\|_2}$$

Tài liệu nào có vector gần hướng với query vector hơn sẽ có điểm cao hơn và được xếp hạng cao hơn.

#### 5.3.3. Tính điểm với BM25

Với BM25, hệ thống không so sánh vector bằng góc như VSM. Thay vào đó, mỗi term trong query đóng góp một lượng điểm vào tài liệu chứa nó:

$$\text{score}_{\text{BM25}}(d,q) = \sum_{t \in q \cap d} IDF_{BM25}(t) \cdot \frac{f_{t,d}(k_1+1)}{f_{t,d} + k_1\left(1-b+b\cdot\frac{|d|}{avgdl}\right)}$$

Trong thực nghiệm, hệ thống sử dụng:

* $k_1 = 2.0$
* $b = 0.6$
* $avgdl = 95.58$

Tài liệu có nhiều term truy vấn quan trọng, tần suất hợp lý và độ dài không quá lệch so với trung bình sẽ có điểm BM25 cao hơn.

---

## 6. Thử nghiệm và Đánh giá Hiệu năng

Hệ thống được đánh giá bằng phương pháp kiểm thử toàn diện trên toàn bộ **225 câu truy vấn** của tập Cranfield. Các mô hình đều dùng chung pipeline tiền xử lý, cùng tập relevance judgments và cùng bộ chỉ số: **MAP**, **Precision@20 (P@20)** và **Recall@20 (R@20)**.

### 6.1. Kết quả baseline trước khi cải tiến

Bảng dưới đây là kết quả của hai mô hình chính trước khi áp dụng các phần mở rộng/cải tiến:

| Mô hình | MAP | P@20 | R@20 |
|---|:---:|:---:|:---:|
| VSM Baseline | 0.2923 | 0.1573 | 0.5048 |
| BM25 Baseline | 0.3118 | 0.1622 | 0.5182 |

**Nhận xét:** BM25 tốt hơn VSM ở cả ba metric. Điều này phù hợp với kỳ vọng vì BM25 có cơ chế bão hòa tần suất và chuẩn hóa độ dài tài liệu, trong khi VSM chủ yếu dựa trên cosine similarity của vector TF-IDF.

---

## 7. Phân tích lỗi và trường hợp truy vấn
Để phần phân tích không chỉ dừng ở nhận xét cảm tính, nhóm kiểm tra từng query bằng dữ liệu thật từ `evaluation_per_query.csv`, `query_bm25.csv`, `TEST/query.txt` và relevance judgments trong `TEST/RES`. Script kiểm chứng được lưu tại `analyze_query_cases.py`, kết quả chi tiết nằm trong `docs/query_case_evidence.md`.

Cách phân tích mỗi query:

1. Xem AP, P@20, Recall@20 để biết query tốt hay kém.
2. Kiểm tra top tài liệu hệ thống trả về có nằm trong tập relevance hay không.
3. Xem các term nào đóng góp điểm BM25 lớn nhất.
4. Từ đó xác định nguyên nhân: term đặc thù, term nhiễu, mismatch từ vựng, hay mất thông tin cụm từ.

---

### 7.1. Trường hợp truy vấn tốt

Các query tốt thường có term chuyên ngành rõ và khớp trực tiếp với tài liệu liên quan.

Lưu ý: hệ thống vẫn xếp hạng trên toàn bộ **1.400 tài liệu**. Cột "Relevant docs" là số tài liệu thật sự liên quan trong ground truth (`TEST/RES`), không phải số tài liệu hệ thống retrieve.

| Query ID | Nội dung rút gọn | Relevant docs | AP BM25 | Bằng chứng |
|---:|---|---:|---:|---|
| 119 | `axisymmetric deviations`, `load-deflection`, `hydrostatic pressure` | 1 | 1.0000 | Doc đúng đứng rank 1 |
| 150 | `wing-body interference`, `supersonic mach number` | 2 | 1.0000 | Hai doc đúng đứng rank 1 và 2 |
| 41 | `vortex wake`, `cruciform wing` | 3 | 0.8667 | Hai doc đúng nằm ngay rank 1 và 2 |

#### Case tốt: Query 150

Query:

```text
what is the magnitude of second-order wing-body interference at high supersonic mach number
```

Kết quả BM25 top đầu:

| Rank | Doc ID | BM25 score | Relevant? |
|---:|---:|---:|:---:|
| 1 | 1074 | 29.1695 | Yes |
| 2 | 1075 | 27.0222 | Yes |
| 3 | 1062 | 26.9908 | No |

Các term đóng góp mạnh ở rank 1:

| Term | TF | IDF BM25 | Term score |
|---|---:|---:|---:|
| `interfer` | 3 | 3.4055 | 6.0929 |
| `second` | 4 | 2.5399 | 5.0543 |
| `order` | 5 | 1.9592 | 4.1802 |
| `wing` | 5 | 1.8222 | 3.8878 |

Trích đoạn đối chiếu:

| Doc | Relevant? | Trích đoạn nội dung |
|---:|:---:|---|
| 1074 | Yes | theoretical and experimental investigation of **second order supersonic wing body interference** ... approximate **second order** solutions for the **supersonic** flow around **wing body** combinations ... |
| 1075 | Yes | an experimental and theoretical investigation of **second order supersonic wing body interference** ... pressure distributions on the **wing** ... at **mach numbers 3 and 4** ... |

**Phân tích:** Query 150 là case dễ cho BM25 vì cụm trong query xuất hiện gần như nguyên vẹn trong hai tài liệu đúng: **second order supersonic wing body interference**. Các term có điểm cao như `interfer`, `second`, `order`, `wing` đều nằm trong cùng ngữ cảnh, không phải chỉ khớp rời rạc. Vì vậy hai tài liệu relevant được đưa lên rank 1 và rank 2, làm AP BM25 đạt 1.0000.

---

### 7.2. Trường hợp truy vấn kém

Query kém không phải lúc nào cũng do thiếu từ khóa. Nhiều query vẫn có term rất mạnh, nhưng hệ thống có thể chọn nhầm nếu tài liệu chỉ trùng từ khóa mà không trả lời đúng ý của truy vấn.

| Query ID | Nội dung rút gọn | Relevant docs | AP BM25 | Hiện tượng |
|---:|---|---:|---:|---|
| 13 | `transonic aileron buzz` | 4 | 0.0000 | Term đúng chủ đề nhưng top docs đều không relevant |
| 87 | `boundary layer`, `inviscid flow`, `shock` | 8 | 0.0000 | Nhiều term phổ biến trong Cranfield, dễ nhiễu |
| 109 | `panels`, `aerodynamic heating` | 5 | 0.0000 | Query ngắn, ít tín hiệu phân biệt |

#### Case lỗi: Query 13

Query:

```text
what is the basic mechanism of the transonic aileron buzz
```

Kết quả BM25 top đầu:

| Rank | Doc ID | BM25 score | Relevant? |
|---:|---:|---:|:---:|
| 1 | 496 | 26.7918 | No |
| 2 | 903 | 15.7377 | No |
| 3 | 520 | 12.6100 | No |
| 4 | 643 | 11.0287 | No |
| 5 | 199 | 10.8414 | No |

Các term đóng góp mạnh ở rank 1:

| Term | TF | IDF BM25 | Term score |
|---|---:|---:|---:|
| `buzz` | 2 | 6.8395 | 11.1931 |
| `aileron` | 3 | 4.9936 | 9.6314 |
| `transon` | 3 | 3.0939 | 5.9673 |

Trích đoạn đối chiếu:

| Doc | Relevant? | Trích đoạn nội dung |
|---:|:---:|---|
| 496 | No | a theory of **transonic aileron buzz**, neglecting viscous effects ... harmonic oscillations of an **aileron** ... stability boundary for **transonic aileron buzz** ... |
| 265 | Yes | instabilities arising from the interaction between **shock waves** and **boundary layer** ... oscillatory behaviour of aerofoils and **control surfaces** ... shock induced separation in the instability of a **control surface** ... |

Doc 496 được BM25 cho điểm rất cao vì khớp trực tiếp các term mạnh: **buzz**, **aileron**, **transon**. Tuy nhiên, theo ground truth, Doc 496 không được đánh dấu relevant cho query 13.

**Phân tích:** Query 13 hỏi về **basic mechanism** của hiện tượng **transonic aileron buzz**, tức là muốn tìm tài liệu giải thích cơ chế gây ra hiện tượng này. Doc 496 lại tập trung vào việc xây dựng **a theory of transonic aileron buzz**, mô hình hóa dao động của **aileron**, và đưa ra stability boundary. Vì Doc 496 lặp lại trực tiếp các term mạnh như `buzz`, `aileron`, `transon`, BM25 cho điểm rất cao và xếp hạng 1. Tuy nhiên, theo ground truth, Doc 496 không được xem là tài liệu relevant cho query này. Đây là lỗi do BM25 ưu tiên khớp từ khóa bề mặt, trong khi query cần đúng khía cạnh **cơ chế nền** của hiện tượng.

#### Case lỗi: Query 87

Query:

```text
what effect has the boundary layer in modifying the basic inviscid flow behind the shock
```

Các term đóng góp mạnh ở rank 1:

| Term | TF | IDF BM25 | Term score |
|---|---:|---:|---:|
| `corner` | 2 | 4.3827 | 7.0281 |
| `lead` | 4 | 2.0830 | 4.3535 |
| `edg` | 4 | 2.0058 | 4.1922 |
| `shock` | 3 | 1.7622 | 3.3449 |

Trích đoạn tài liệu rank 1:

| Doc | Relevant? | Trích đoạn nội dung |
|---:|:---:|---|
| 1228 | No | **leading edge** separation of laminar boundary layers in supersonic flow ... interaction of **shock wave** and laminar boundary layer on a compression **corner** ... compression **corner** angle ... |

**Phân tích:** Query 87 có một điều kiện quan trọng: nó hỏi ảnh hưởng của **boundary layer** phía sau **shock**, nhưng đồng thời nói **neglecting effects of leading edge and corner** (bỏ qua ảnh hưởng của cạnh trước và góc). Doc 1228 lại tập trung vào **leading edge separation** và **compression corner**. Vì BM25 tính điểm theo các term riêng lẻ, những từ như `corner`, `lead`, `edg`, `shock` vẫn tạo điểm cao, mặc dù chính phần **leading edge/corner** là thứ query muốn bỏ qua. Đây là lỗi do mô hình không giữ được ràng buộc phủ định/điều kiện trong câu truy vấn.

---

### 7.3. Bài học rút ra

Từ các case trên, có thể rút ra ba nhóm lỗi chính:

* **False positive do term mạnh**: query 13 cho thấy tài liệu có thể chứa term hiếm như `buzz`, `aileron` nhưng vẫn không đúng nhu cầu truy vấn.
* **Nhiễu do term phổ biến**: query 87 cho thấy các term như `flow`, `shock`, `boundary` dễ xuất hiện trong nhiều tài liệu, làm ranking bị nhiễu.
* **Mất thông tin cụm từ**: các cụm như `boundary layer`, `skin friction`, `aileron buzz` có ý nghĩa mạnh hơn từng từ riêng lẻ, nhưng hệ thống hiện chủ yếu xử lý theo unigram.

Hướng cải thiện:

* Thêm phrase matching cho các cụm chuyên ngành.
* Dùng query expansion hoặc pseudo relevance feedback để bổ sung term đặc trưng.
* Kết hợp tín hiệu chủ đề như Cluster Reranking để giảm phụ thuộc hoàn toàn vào term matching.

---

## 8. Hai phần mở rộng

### 8.1. Phần mở rộng 1: KMeans Cluster-based Reranking

#### 9.1.1. Động lực cải tiến
VSM và BM25 xếp hạng chủ yếu dựa trên term matching. Để bổ sung tín hiệu chủ đề, hệ thống thêm bước **Cluster Reranking** sau ranking ban đầu.

Ý tưởng: nếu nhiều tài liệu top đầu cùng thuộc một cụm, cụm đó được xem là có liên quan đến query và các tài liệu trong cụm sẽ được boost nhẹ.

#### 9.1.2. Quy trình offline: xây dựng không gian cụm
Trước khi truy vấn, toàn bộ tài liệu được biểu diễn và phân cụm một lần:

```text
Processed Cranfield Documents
        ↓
TF-IDF Matrix
        ↓
TruncatedSVD
        ↓
LSA Vector Space 100D
        ↓
L2 Normalization
        ↓
KMeans Clustering với 200 clusters
        ↓
doc_to_cluster và cluster_to_docs
```

Cấu hình chính: TF-IDF được giảm chiều bằng TruncatedSVD xuống 100 chiều, sau đó chuẩn hóa L2 và phân cụm bằng KMeans với **200 cụm**. Mỗi cụm đại diện cho một nhóm tài liệu có chủ đề gần nhau.

#### 9.1.3. Quy trình online: reranking theo cụm
Khi có query, hệ thống dùng ranking ban đầu của VSM/BM25 để xác định cụm nào đang quan trọng:

```text
Query
  ↓
VSM hoặc BM25 trả về ranking ban đầu trên 1.400 docs
  ↓
Lấy top-20 documents đầu ranking
  ↓
Đếm số lượt xuất hiện của từng cluster trong top-20
  ↓
Chuẩn hóa cluster_score[c] = count[c] / max_count
  ↓
Kết hợp retrieval score và cluster score
  ↓
Sinh ranking mới sau reranking
```

Công thức kết hợp điểm:

$$\text{final\_score}[d] = \alpha \cdot \text{norm\_score}[d] + (1 - \alpha) \cdot \text{cluster\_score}[\text{cluster}(d)]$$

Trong đó:

* $\text{norm\_score}[d] = \frac{\text{retrieval\_score}[d]}{\max(\text{retrieval\_score})}$ là điểm truy hồi gốc đã chuẩn hóa.
* $\text{cluster\_score}[\text{cluster}(d)]$ là mức độ quan trọng của cụm chứa tài liệu $d$ dựa trên voting từ top-20.
* $\alpha = 0.85$ giữ ranking gốc làm tín hiệu chính, cluster chỉ đóng vai trò boost nhẹ.

Minh họa nhanh với ranking ban đầu:

| Hạng ban đầu | Tài liệu | Retrieval score chuẩn hóa | Cluster |
|---:|---:|---:|---:|
| 1 | 51 | 1.00 | C2 |
| 2 | 184 | 0.82 | C2 |
| 3 | 12 | 0.76 | C5 |
| 4 | 486 | 0.70 | C2 |
| 5 | 359 | 0.68 | C9 |

Giả sử hệ thống dùng top-3 tài liệu đầu để xác định cụm quan trọng. Trong top-3, cụm C2 xuất hiện 2 lần, cụm C5 xuất hiện 1 lần. Do đó:

$$\text{cluster\_score}[C2] = 1.0,\quad \text{cluster\_score}[C5] = 0.5$$

Nếu một tài liệu khác thuộc cụm C2 có retrieval score đã chuẩn hóa là 0.70, điểm sau reranking là:

$$\text{final\_score} = 0.85 \times 0.70 + 0.15 \times 1.0 = 0.745$$

Khi áp dụng cho thêm một vài tài liệu ứng viên:

| Tài liệu | Retrieval score chuẩn hóa | Cluster | Cluster score | Final score | Thứ hạng sau rerank |
|---:|---:|---:|---:|---:|---:|
| 51 | 1.00 | C2 | 1.00 | 1.000 | 1 |
| 184 | 0.82 | C2 | 1.00 | 0.847 | 2 |
| 12 | 0.76 | C5 | 0.50 | 0.721 | 4 |
| 486 | 0.70 | C2 | 1.00 | 0.745 | 3 |
| 359 | 0.68 | C9 | 0.00 | 0.578 | 5 |

Ở ranking gốc, tài liệu 486 đứng sau tài liệu 12. Sau reranking, 486 được tăng điểm vì thuộc cụm C2, là cụm xuất hiện nhiều trong top đầu, nên vượt lên trên tài liệu 12. Ngược lại, tài liệu 359 không thuộc cụm quan trọng nên không được boost.

#### 9.1.4. Kết quả của Cluster Reranking

| Model | MAP | P@20 | R@20 |
|---|:---:|:---:|:---:|
| VSM Baseline | 0.2923 | 0.1573 | 0.5048 |
| **VSM + Cluster Reranking** | **0.3045** | **0.1660** | **0.5305** |
| BM25 Baseline | 0.3118 | 0.1622 | 0.5182 |
| **BM25 + Cluster Reranking** | **0.3297** | **0.1720** | **0.5500** |

Mức cải thiện:

| So sánh | ΔMAP | ΔP@20 | ΔR@20 |
|---|:---:|:---:|:---:|
| VSM + Cluster so với VSM | +4.17% | +5.53% | +5.09% |
| BM25 + Cluster so với BM25 | +5.74% | +6.04% | +6.13% |

**Nhận xét:** Cluster Reranking cải thiện đồng loạt cả ba metric cho cả VSM và BM25. Kết quả tốt nhất là **BM25 + Cluster Reranking** với MAP = **0.3297** và R@20 = **0.5500**.

Phân tích kết quả:

* **MAP tăng** cho thấy reranking không chỉ đưa thêm tài liệu liên quan vào danh sách, mà còn cải thiện vị trí của chúng trong ranking. Nói cách khác, một số tài liệu đúng được đẩy lên sớm hơn.
* **P@20 tăng** cho thấy trong 20 kết quả đầu có nhiều tài liệu liên quan hơn. Điều này phù hợp với cơ chế cluster boost: các tài liệu cùng cụm với nhóm top đầu có thêm cơ hội được đưa vào top-20.
* **R@20 tăng** cho thấy hệ thống tìm được thêm tài liệu liên quan mà baseline VSM/BM25 ban đầu chưa ưu tiên đủ cao.
* **BM25 + Cluster tốt nhất** vì BM25 đã là baseline mạnh hơn VSM; khi thêm tín hiệu cụm, mô hình vừa giữ được khả năng term matching tốt, vừa bổ sung thêm tín hiệu chủ đề.

Tuy nhiên, cluster chỉ được dùng như tín hiệu phụ với $\alpha = 0.85$, nên ranking gốc vẫn chiếm vai trò chính. Điều này giúp hạn chế rủi ro đẩy quá nhiều tài liệu cùng cụm nhưng không thật sự liên quan lên đầu danh sách.

---

### 8.2. Phần mở rộng 2: So sánh với Whoosh BM25F Baseline

#### 9.2.1. Mục tiêu so sánh
Phần mở rộng thứ hai dùng thư viện **Whoosh** để xây dựng một baseline bên ngoài. Mục tiêu không phải thay thế mô hình thủ công, mà là dùng một thư viện IR có sẵn để kiểm chứng chất lượng tương đối của pipeline tự cài đặt.

Để so sánh công bằng, Whoosh được chạy trên cùng:

* Bộ tài liệu Cranfield 1.400 documents.
* 225 queries.
* 225 relevance files.
* Bộ metric MAP, P@20, R@20.
* Pipeline tiền xử lý của dự án.

#### 9.2.2. Tiền xử lý dùng cho Whoosh
Thay vì dùng analyzer mặc định của Whoosh, hệ thống đưa documents và queries qua lại **cùng hàm `process_document`** đã dùng cho VSM/BM25 thủ công:

```text
Raw text
  ↓
lowercase và thay '-' bằng khoảng trắng
  ↓
mở rộng viết tắt bằng ABBREVIATIONS
  ↓
chuyển số thành chữ bằng num2words
  ↓
loại ký tự đặc biệt
  ↓
word_tokenize
  ↓
stopword removal + bỏ token chữ có độ dài <= 2
  ↓
Snowball stemming
  ↓
chuỗi token đã chuẩn hóa dùng cho Whoosh index/search
```

Trong Whoosh, trường nội dung dùng `KeywordAnalyzer()` để tránh việc Whoosh tokenize hoặc stem lại lần nữa. Như vậy, dữ liệu đưa vào Whoosh đã ở cùng không gian term với mô hình VSM/BM25 của dự án.

#### 9.2.3. Cấu hình Whoosh

| Thành phần | Cấu hình |
|---|---|
| Thư viện | Whoosh |
| Scoring | BM25F |
| Analyzer | `KeywordAnalyzer()` sau khi đã preprocess thủ công |
| Index field | `doc_id`, `content` |
| Query parser | `QueryParser` với `OrGroup` |
| Số tài liệu retrieve | 1.400 docs/query |
| Evaluation | MAP, P@20, Recall@20 |

`OrGroup` được dùng để query hoạt động theo hướng OR giữa các term đã xử lý. Điều này gần với cách hệ thống thủ công dùng chỉ mục đảo ngược để cộng điểm các tài liệu chứa ít nhất một term của query.

#### 9.2.4. Kết quả so sánh với Whoosh

| Model | MAP | P@20 | R@20 |
|---|:---:|:---:|:---:|
| VSM Baseline | 0.2923 | 0.1573 | 0.5048 |
| VSM + Cluster Reranking | 0.3045 | 0.1660 | 0.5305 |
| BM25 Baseline | 0.3118 | 0.1622 | 0.5182 |
| **BM25 + Cluster Reranking** | **0.3297** | **0.1720** | **0.5500** |
| Whoosh BM25F Baseline | 0.3030 | 0.1607 | 0.5123 |

#### 9.2.5. Phân tích kết quả Whoosh
Kết quả Whoosh BM25F đạt **MAP = 0.3030**, cao hơn VSM Baseline và gần với VSM + Cluster Reranking, nhưng vẫn thấp hơn BM25 thủ công và BM25 + Cluster Reranking.

Điều này cho thấy:

* Pipeline thủ công không chỉ tái hiện được chất lượng của thư viện IR có sẵn, mà còn vượt Whoosh BM25F trong thực nghiệm Cranfield.
* BM25 thủ công được tuning trực tiếp cho tập dữ liệu này với $k_1 = 2.0$ và $b = 0.6$, trong khi Whoosh BM25F dùng cấu hình scoring tổng quát hơn.
* Khi kết hợp thêm cluster reranking, mô hình thủ công khai thác được tín hiệu chủ đề ngoài term matching nên đạt kết quả cao nhất.

---

### 8.3. Tổng kết phần thực nghiệm

| Hạng | Mô hình | MAP | Nhận xét ngắn |
|:---:|---|:---:|---|
| 1 | **BM25 + Cluster Reranking** | **0.3297** | Tốt nhất tổng thể, cân bằng term matching và micro-topic reranking |
| 2 | BM25 Baseline | 0.3118 | Baseline thủ công mạnh nhất trước khi rerank |
| 3 | VSM + Cluster Reranking | 0.3045 | Cluster giúp VSM vượt rõ baseline |
| 4 | Whoosh BM25F Baseline | 0.3030 | Baseline thư viện tốt, nhưng chưa vượt BM25 thủ công |
| 5 | VSM Baseline | 0.2923 | Mô hình nền đơn giản nhất |

Hai phần mở rộng cho thấy hướng cải tiến có ý nghĩa:

1. **KMeans Cluster Reranking** cải thiện chất lượng truy hồi bằng tín hiệu chủ đề ở cấp cụm.
2. **Whoosh BM25F** cung cấp baseline thư viện để đối chiếu, chứng minh pipeline thủ công đạt chất lượng cạnh tranh và có thể vượt baseline tổng quát khi được tuning theo Cranfield.

---

---


## 9. Phụ lục Optional: Minh họa Tính toán Chạy tay trên Query 1
Để làm sáng tỏ quy trình vận hành chi tiết của hệ thống, dưới đây là phần phân tích từng bước tính toán đối với **Query 1**:

### 9.1. Chạy tay Pipeline Tiền xử lý (Query 1)
* **Câu gốc**: `"what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft ."`
* **Chuyển chữ thường & Loại ký tự đặc biệt**: `"what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft"`
* **Tách từ (Tokenize)**: `['what', 'similarity', 'laws', 'must', 'be', 'obeyed', 'when', 'constructing', 'aeroelastic', 'models', 'of', 'heated', 'high', 'speed', 'aircraft']`
* **Lọc Stopword & Từ ngắn (length <= 2)**: 
  * Các từ bị loại bỏ: `what`, `must`, `be`, `when`, `of`
  * Các từ được giữ lại: `['similarity', 'laws', 'obeyed', 'constructing', 'aeroelastic', 'models', 'heated', 'high', 'speed', 'aircraft']`
* **Stemming (Thuật toán Snowball)**:
  * `similarity` $\rightarrow$ **similar**
  * `laws` $\rightarrow$ **law**
  * `obeyed` $\rightarrow$ **obey**
  * `constructing` $\rightarrow$ **construct**
  * `aeroelastic` $\rightarrow$ **aeroelast**
  * `models` $\rightarrow$ **model**
  * `heated` $\rightarrow$ **heat**
  * `high` $\rightarrow$ **high**
  * `speed` $\rightarrow$ **speed**
  * `aircraft` $\rightarrow$ **aircraft**
* **Kết quả Token cuối cùng**: `['similar', 'law', 'obey', 'construct', 'aeroelast', 'model', 'heat', 'high', 'speed', 'aircraft']`

---

### 9.2. Kết quả Xếp hạng và Điểm số Top 5 (Query 1)

#### Kết quả với mô hình Vector Space Model (VSM):
Bảng phân rã Cosine Similarity của các tài liệu hàng đầu cho thấy mức độ đóng góp điểm số chủ yếu đến từ các từ khóa hiếm có giá trị IDF cao như `aeroelast` (IDF = 4.09), `obey` (IDF = 4.85):
* **Tài liệu 51** (Score = **0.2780**)
* **Tài liệu 184** (Score = **0.2437**)
* **Tài liệu 12** (Score = **0.2185**)
* **Tài liệu 359** (Score = **0.1956**)
* **Tài liệu 746** (Score = **0.1916**)

#### Kết quả với mô hình Okapi BM25:
Mô hình BM25 đạt điểm số vượt trội do bão hòa tần suất của các từ xuất hiện nhiều lần trong tài liệu:
* **Tài liệu 51** (BM25 Score = **25.3230**)
* **Tài liệu 486** (BM25 Score = **22.3925**)
* **Tài liệu 12** (BM25 Score = **20.3612**)
* **Tài liệu 184** (BM25 Score = **19.1581**)
* **Tài liệu 878** (BM25 Score = **17.3155**)

---

### 9.3. Chạy tay đánh giá độ chính xác (Evaluation) cho Query 1 với k=5
* **Tập tài liệu thực sự liên quan (Relevance Judgments)**: $R_{q_1}$ gồm **28 tài liệu** (trong đó có các tài liệu số `12, 51, 184, ...`).
* **Danh sách tài liệu hệ thống VSM trả về (Top 5)**: `[51, 184, 12, 359, 746]`
* **Số tài liệu khớp đúng (Hits trong Top 5)**: Tài liệu `51` (đúng), `184` (đúng), `12` (đúng). Tài liệu `359` và `746` là sai. Tổng cộng có **3 tài liệu đúng**.

#### 1. Tính Precision@5:
$$P@5 = \frac{\text{Số tài liệu khớp đúng}}{\text{Số tài liệu trả về}} = \frac{3}{5} = 0.6000\ (60\%)$$

#### 2. Tính Recall@5:
$$Recall@5 = \frac{\text{Số tài liệu khớp đúng}}{\text{Tổng số tài liệu liên quan chuẩn}} = \frac{3}{28} \approx 0.1071\ (10.71\%)$$

#### 3. Tính Average Precision (AP@5) cho câu truy vấn 1:
Xét lần lượt từng vị trí từ $i=1$ đến $5$:
* Tại vị trí $i=1$ (tài liệu `51` - Đúng ✅): $\text{Precision}(1) = \frac{1}{1} = 1.0000$
* Tại vị trí $i=2$ (tài liệu `184` - Đúng ✅): $\text{Precision}(2) = \frac{2}{2} = 1.0000$
* Tại vị trí $i=3$ (tài liệu `12` - Đúng ✅): $\text{Precision}(3) = \frac{3}{3} = 1.0000$
* Tại vị trí $i=4$ (tài liệu `359` - Sai ❌): Không tính.
* Tại vị trí $i=5$ (tài liệu `746` - Sai ❌): Không tính.

$$\text{AP}(q_1) = \frac{1.0000 + 1.0000 + 1.0000}{|R_{q_1}|} = \frac{3.0000}{28} \approx 0.1071$$

---

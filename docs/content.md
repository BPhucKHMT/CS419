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
Chọn term (lựa chọn từ khóa đại diện) đóng vai trò quyết định đến hiệu năng của hệ thống IR. Nếu không lọc nhiễu, kích thước từ điển sẽ rất lớn, dẫn đến chỉ mục cồng kềnh và tốc độ truy vấn chậm. Ngược lại, nếu lọc quá đà sẽ làm mất mát thông tin ngữ nghĩa quan trọng.

### 3.1. Loại Term được chọn & Lý giải lựa chọn
Trong các phương pháp chọn term, dự án lựa chọn phương pháp:
* **Loại term được chọn**: **Từ đơn (Word)** làm đơn vị cơ bản cho không gian từ vựng.
* **Lý do lựa chọn**:
  1. *Tính chất ngôn ngữ tiếng Anh*: Bộ dữ liệu Cranfield viết bằng tiếng Anh, có ranh giới từ phân định rõ ràng qua khoảng trắng và các ký tự đặc biệt, giúp việc tách từ (Word Segmentation/Tokenize) đạt độ chính xác cao mà không cần các bộ từ điển ranh giới phức tạp.
  2. *Đơn giản và hiệu năng*: Biểu diễn tài liệu dưới dạng túi từ (Bag-of-Words) trên các từ đơn là giải pháp tối ưu và phổ biến nhất, đảm bảo cân bằng giữa hiệu năng tính toán và độ chính xác truy hồi. Việc sử dụng *N-gram* sẽ làm bùng nổ số lượng chiều của vector (kích thước từ điển tăng theo cấp số nhân), gây quá tải bộ nhớ và tính toán. Trong khi sử dụng *Khái niệm (Concept)* đòi hỏi thêm quy trình khử nhập nhằng nghĩa (Word Sense Disambiguation) phức tạp nhưng hiệu quả mang lại trên ngữ liệu kỹ thuật khí động học không thực sự vượt trội.

### 3.2. Phương pháp Chọn Term
Hệ thống áp dụng phương pháp lọc và chọn term dựa trên hai nguyên lý cốt lõi trong slide:
1. **Mức độ quan trọng và Độ phân biệt của từ (Word's Resolution Power)**:
   * Loại bỏ các từ dừng (Stopwords): Những từ xuất hiện cực kỳ phổ biến trên toàn bộ corpus như *the, is, an, of* có DF rất cao dẫn đến IDF xấp xỉ 0. Các từ này hoàn toàn không có khả năng phân biệt để tìm ra tài liệu liên quan. Việc loại bỏ chúng giúp loại trừ các chiều vector vô nghĩa.
   * Loại bỏ các từ quá ngắn ($\le 2$ ký tự): Hầu hết các từ này là từ dừng viết tắt hoặc nhiễu ký tự đơn.
2. **Xử lý các lớp tương đương (Equivalence Classes / Morphological Variations)**:
   * Để giải quyết bài toán biến thể hình thái từ (ví dụ: số nhiều/số ít như *wing* và *wings*; các thì của động từ như *obeyed* và *obeying*), hệ thống đưa chúng về một gốc từ (stem) duy nhất để biểu diễn.
   * Sử dụng thuật toán **Stemming (Snowball Stemmer)** để đồng nhất các từ thuộc cùng lớp tương đương. Nhờ đó, một câu truy vấn chứa từ *obeyed* vẫn có khả năng khớp chính xác với tài liệu chứa từ *obey*.

---

### 3.3. Thuật toán Chọn Term
Quy trình lọc và chọn term từ văn bản tài liệu/câu truy vấn được thực hiện tự động qua thuật toán sau:

#### Mã giả thuật toán (Pseudocode):
```text
ALGORITHM TermSelection(DocumentText D, StopwordsList S, AbbreviationsDict A, Stemmer St)
INPUT: Văn bản thô D, Danh sách từ dừng S, Từ điển viết tắt A, Thuật toán Stemmer St
OUTPUT: Danh sách các Term được lựa chọn đại diện

1. D_low ← Chuyển D thành chữ thường, thay thế tất cả gạch ngang '-' bằng khoảng trắng
2. For each pattern, expansion in A:
       D_low ← Thay thế tất cả các chuỗi khớp với pattern trong D_low bằng expansion
3. For each numeric_sequence in D_low:
       word_equivalent ← Chuyển số thành chữ tiếng Anh (dùng num2words)
       D_low ← Thay thế numeric_sequence trong D_low bằng word_equivalent
4. D_clean ← Loại bỏ toàn bộ các ký tự không phải chữ cái (a-z), chữ số (0-9) hoặc khoảng trắng
5. Tokens ← Tách từ D_clean bằng thuật toán phân tách từ (word_tokenize)
6. RemainingTokens ← Khởi tạo danh sách rỗng
7. For each word in Tokens:
       If (độ dài word > 2 OR word chứa ký tự số) AND word không nằm trong S:
           Append word vào RemainingTokens
8. SelectedTerms ← Khởi tạo danh sách rỗng
9. For each word in RemainingTokens:
       stemmed_word ← Áp dụng St.stem(word) (Snowball Stemmer)
       Append stemmed_word vào SelectedTerms
10. Return SelectedTerms
```

---

### 3.4. Hiệu quả suy giảm kích thước từ vựng (Vocabulary Reduction)
Nhờ áp dụng thuật toán chọn term chặt chẽ, không gian từ vựng và kích thước tài liệu trung bình của hệ thống được tối ưu hóa:

| Đặc trưng thống kê | Trước xử lý (Raw Tokenize) | Sau xử lý (Processed) | Tỷ lệ giảm |
|--------------------|:-------------------------:|:--------------------:|:----------:|
| **Số lượng Term độc bản (Vocabulary Size)** | **7.472** | **4.452** | **40.42%** |
| **Độ dài trung bình tài liệu** | **161.91** từ | **95.58** từ | **40.97%** |

* **Tổng số term phân tích được từ toàn bộ tài liệu (Trước xử lý)**: **7.472 terms**.
* **Tổng số term được lựa chọn đưa vào từ điển (Sau tiền xử lý)**: **4.452 terms**.
* **Tỷ lệ chọn term (Selection Ratio)**: 
  $$\text{Tỷ lệ term được chọn} = \frac{4.452}{7.472} \approx 59.58\%$$
  *(Như vậy, hệ thống loại bỏ 40.42% các từ nhiễu/từ dừng không mang thông tin và giữ lại 59.58% gốc từ mang nội dung thực thụ)*.

---

### 3.5. Danh sách các Term được chọn tiêu biểu
Dưới đây là danh sách 25 term xuất hiện nhiều nhất trong bộ từ vựng 4.452 term được chọn từ tập tài liệu Cranfield, kèm theo tổng tần suất xuất hiện (`Total_TF`), số tài liệu chứa term (`DF`) và giá trị trọng số nghịch đảo tài liệu (`IDF`):

| Hạng | Term được chọn | Thừa số gốc (Gợi ý từ gốc) | Tổng tần suất (Total_TF) | Số tài liệu chứa (DF) | Trọng số IDF |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | **flow** | *flowing, flows, flow* | 2082 | 730 | 0.6512 |
| 2 | **number** | *numbers, number* | 1500 | 633 | 0.7938 |
| 3 | **pressur** | *pressure, pressures* | 1391 | 552 | 0.9307 |
| 4 | **boundari** | *boundary, boundaries* | 1216 | 470 | 1.0915 |
| 5 | **layer** | *layer, layers* | 1164 | 414 | 1.2184 |
| 6 | **result** | *results, result* | 1088 | 692 | 0.7046 |
| 7 | **two** | *two* | 1071 | 602 | 0.8440 |
| 8 | **point** | *points, point* | 1065 | 476 | 1.0788 |
| 9 | **one** | *one* | 1032 | 561 | 0.9145 |
| 10 | **effect** | *effects, effect* | 996 | 540 | 0.9527 |
| 11 | **method** | *methods, method* | 887 | 455 | 1.1239 |
| 12 | **theori** | *theory, theories* | 882 | 456 | 1.1217 |
| 13 | **bodi** | *bodies, body* | 854 | 293 | 1.5641 |
| 14 | **solut** | *solution, solutions* | 849 | 407 | 1.2354 |
| 15 | **heat** | *heating, heated, heat* | 847 | 306 | 1.5206 |
| 16 | **wing** | *wings, wing* | 837 | 226 | 1.8237 |
| 17 | **mach** | *mach* | 823 | 388 | 1.2832 |
| 18 | **equat** | *equation, equations* | 781 | 402 | 1.2478 |
| 19 | **shock** | *shock, shocks* | 746 | 240 | 1.7636 |
| 20 | **use** | *used, use, using* | 733 | 513 | 1.0040 |
| 21 | **present** | *presented, present* | 698 | 507 | 1.0157 |
| 22 | **surfac** | *surface, surfaces* | 691 | 330 | 1.4451 |
| 23 | **distribut** | *distribution, distributions* | 650 | 361 | 1.3553 |
| 24 | **obtain** | *obtained, obtain* | 643 | 464 | 1.1043 |
| 25 | **temperatur** | *temperature, temperatures* | 629 | 268 | 1.6532 |

---

## 4. Mô hình Truy xuất & Cơ chế tính Trọng số Term

Báo cáo phân tích chi tiết cơ sở toán học và cơ chế tính trọng số, đồng thời làm rõ sự khác biệt giữa **Công thức cơ sở tính độ liên quan** (lý thuyết/toán học gốc) và **Công thức xếp hạng** (rút gọn/lập trình thực tế) của các mô hình trong đồ án:

### 4.1. Biểu diễn Tài liệu và Câu truy vấn (Representation)
Cách thức biểu diễn thông tin đóng vai trò là đầu vào cho quá trình so khớp độ liên quan:
* **Trong mô hình Vector (VSM)**: Cả tài liệu $d$ và câu truy vấn $q$ đều được biểu diễn dưới dạng **Vector số thực** trong không gian từ vựng đa chiều:
  $$\vec{d} = \left(w_{1,d}, w_{2,d}, ..., w_{V,d}\right)$$
  $$\vec{q} = \left(w_{1,q}, w_{2,q}, ..., w_{V,q}\right)$$
  *(Trong đó $V=4452$ là chiều từ vựng, các thành phần là trọng số TF-IDF).*
* **Trong mô hình Probabilistic (Okapi BM25)**: Tài liệu được biểu diễn dưới dạng **Túi từ (Bag-of-Words)** đi kèm với độ dài thực tế của tài liệu $|d|$ để chuẩn hóa.

---

### 4.2. Vector Space Model (VSM) với TF-IDF

#### 1. Công thức cơ sở để tính độ liên quan giữa tài liệu và câu truy vấn
Trong mô hình VSM lý thuyết, độ tương đồng hay độ liên quan giữa tài liệu $d$ và câu truy vấn $q$ được đo lường bằng **Cosine Similarity** (độ đo góc giữa hai vector):
$$\text{Similarity}(q, d) = \cos(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2} = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\sqrt{\sum_{t \in q} w_{t,q}^2} \cdot \sqrt{\sum_{t \in d} w_{t,d}^2}}$$
* **Cơ sở lý thuyết**: Mô hình VSM (do Gerard Salton đề xuất) biểu diễn tài liệu và câu truy vấn như các vector trong không gian Euclid đa chiều $\mathbb{R}^V$ ($V = 4452$ trong hệ thống này). Cosine Similarity chỉ quan tâm đến hướng góc lệch vector chứ không phụ thuộc vào độ dài vector. Bằng cách chia cho tích độ dài Euclid (L2-norm) $\|\vec{q}\|_2 \|\vec{d}\|_2$, phép đo này chuẩn hóa các văn bản về cùng một mặt cầu đơn vị, giúp loại bỏ hoàn toàn sai lệch do độ dài văn bản thô khác biệt gây ra.

#### 2. Công thức tính trọng số term
Trọng số của mỗi từ khóa trong mô hình VSM được tính toán dựa trên hệ thống TF-IDF để kết hợp tầm quan trọng nội bộ (local) và độ phân biệt toàn cục (global):
* **Trọng số từ trong Tài liệu ($w_{t,d}$)**:
  $$w_{t,d} = TF(t, d) \times IDF(t) = \frac{tf_{t,d}}{|d|} \times \ln\!\left(\frac{N}{df_t}\right)$$
* **Trọng số từ trong Câu truy vấn ($w_{t,q}$)**:
  $$w_{t,q} = TF(t, q) \times IDF(t) = \frac{tf_{t,q}}{|q|} \times \ln\!\left(\frac{N}{df_t}\right)$$
* **Giải thích thành phần**:
  * **Tần suất từ khóa ($TF$)**: Phản ánh mật độ xuất hiện của từ khóa bằng cách chia tần suất thô ($tf$) cho tổng số từ của tài liệu ($|d|$) hoặc query ($|q|$) nhằm chuẩn hóa độ dài tuyến tính.
  * **Tần suất nghịch đảo tài liệu ($IDF$)**: Phản ánh giá trị thông tin/độ phân biệt của từ khóa. Lấy Logarithm tự nhiên của tổng số tài liệu $N$ chia cho số tài liệu chứa từ $df_t$.

#### 3. Công thức xếp hạng tài liệu
Trong lập trình thực tế trực tuyến, RSV (Retrieval Status Value) được rút gọn tối đa nhằm tối ưu hiệu năng tính toán:
$$\text{score}_{\text{VSM}}(q, d) = \sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}' = \sum_{t \in q \cap d} w_{t,q} \cdot \frac{w_{t,d}}{\|\vec{d}\|_2}$$
* **Cơ chế xếp hạng**:
  * Chuẩn L2 của vector truy vấn $\|\vec{q}\|_2$ là hằng số đối với tất cả tài liệu ứng viên cho câu truy vấn đó, nên có thể lược bỏ khỏi mẫu số mà không làm thay đổi thứ tự xếp hạng.
  * Chuẩn L2 của vector tài liệu $\|\vec{d}\|_2$ được tính toán trước ở pha ngoại tuyến (Offline Indexing) để sinh ra vector tài liệu đã chuẩn hóa L2 ($w_{t,d}' = \frac{w_{t,d}}{\|\vec{d}\|_2}$).
  * RSV lúc này chỉ cần tính tích vô hướng rút gọn trên các postings list giao nhau.

---

### 4.3. Okapi BM25

#### 1. Công thức cơ sở để tính độ liên quan giữa tài liệu và câu truy vấn
Okapi BM25 có nguồn gốc từ mô hình xác suất truy xuất thông tin (BIM - Binary Independence Model). Công thức cơ sở để tính độ liên quan dựa trên tỷ lệ chênh lệch xác suất (Odds Ratio) của sự kiện tài liệu liên quan ($R=1$) so với không liên quan ($R=0$) dựa trên giả thiết độc lập giữa các từ khóa:
$$O(R=1 \mid d, q) = \frac{P(R=1 \mid d, q)}{P(R=0 \mid d, q)} = \frac{P(d \mid R=1, q) \cdot P(R=1 \mid q)}{P(d \mid R=0, q) \cdot P(R=0 \mid q)}$$
Bằng cách lấy Logarithm của tỷ số xác suất Odds, BIM định nghĩa điểm số độ liên quan cơ sở (không có tần suất và chuẩn hóa chiều dài):
$$\text{Relevance}_{\text{BIM}}(q, d) \propto \sum_{t \in q \cap d} \ln \frac{p_t(1 - u_t)}{u_t(1 - p_t)}$$
* **Cơ sở lý thuyết**: BIM biểu diễn tài liệu dưới dạng vector nhị phân đại diện cho sự xuất hiện (1) hoặc vắng mặt (0) của từ khóa ($p_t$ là xác suất xuất hiện của từ trong tập tài liệu liên quan, $u_t$ trong tập không liên quan). BIM cung cấp khung xác suất lý thuyết cơ sở vững chắc, làm bàn đạp cho BM25 tích hợp thêm tần suất từ khóa thực tế và hiệu chỉnh độ dài tài liệu.

#### 2. Công thức tính trọng số term
Trọng số của mỗi từ khóa trong BM25 tích hợp bão hòa tần suất thực tế ($tf$) và chuẩn hóa độ dài văn bản:
$$w_{\text{BM25}}(t, d) = IDF_{BM25}(t) \cdot \frac{f_{t,d} \cdot (k_1+1)}{f_{t,d} + k_1\left(1 - b + b \cdot \dfrac{|d|}{avgdl}\right)}$$
* **Hàm IDF BM25**: Được điều chỉnh để tránh điểm số IDF bị âm khi từ khóa xuất hiện trong hơn 50% tài liệu:
  $$IDF_{BM25}(t) = \ln\!\left(\frac{N - df_t + 0.5}{df_t + 0.5} + 1\right)$$
* **Bão hòa tần suất ($k_1 = 2.0$)**: Khống chế giới hạn trên của tần suất từ khóa. Khi $f_{t,d} \rightarrow \infty$, trọng số tiến dần đến giới hạn tiệm cận $k_1 + 1$ thay vì tăng tuyến tính vô hạn như VSM.
* **Chuẩn hóa độ dài tài liệu ($b = 0.6$)**: Điều khiển mức phạt độ dài tài liệu $|d|$ so với độ dài trung bình toàn corpus $avgdl$ (giá trị tính toán thực tế sau tiền xử lý là `95.58`).

#### 3. Công thức xếp hạng tài liệu
Khi có câu truy vấn $q$, hệ thống xếp hạng tài liệu dựa trên tổng điểm trọng số BM25 của các từ khóa trùng khớp:
$$\text{score}_{\text{BM25}}(d,q) = \sum_{t \in q \cap d} IDF_{BM25}(t) \cdot \frac{f_{t,d} \cdot (k_1+1)}{f_{t,d} + k_1\left(1 - b + b \cdot \dfrac{|d|}{avgdl}\right)}$$
* **Cơ chế xếp hạng**: Điểm số xếp hạng cuối cùng (RSV) là tổng điểm tích lũy của các trọng số từ khóa trong tài liệu $d$ thuộc câu truy vấn $q$. Các tài liệu không chứa bất kỳ từ khóa truy vấn nào sẽ có điểm bằng 0 và bị loại bỏ khỏi danh sách xếp hạng.

---

## 5. Cấu trúc Chỉ mục & Quá trình Lập/Truy xuất dữ liệu

Hệ thống sử dụng cấu trúc **Chỉ mục đảo ngược (Inverted Index)** để tối ưu hóa hiệu năng, giảm độ phức tạp thời gian tìm kiếm từ quét tuyến tính toàn bộ corpus $O(N \cdot |Q|)$ xuống chỉ còn $O(|Q| \cdot L)$ với $L$ là độ dài trung bình của postings list ($L \ll N$).

### 5.1. Cấu trúc Dữ liệu Chỉ mục
Cấu trúc chỉ mục đảo ngược trong đồ án lưu trữ ánh xạ từ một từ khóa (`term`) đến danh sách các tài liệu chứa từ khóa đó kèm theo thông tin tần suất hoặc trọng số (Postings List) và các siêu dữ liệu hỗ trợ tính toán:

* **Từ điển từ vựng (Vocabulary Dictionary)**:
  Lưu trữ danh sách các term độc bản cùng các thống kê toàn cục:
  * `term` $\rightarrow$ giá trị $df_t$ (Document Frequency - tần suất tài liệu chứa term) và trọng số $IDF$ đã được tính sẵn.
  * Chỉ mục từ vựng (Vocabulary Index) ánh xạ `term` sang một chỉ số nguyên $idx \in [0, V-1]$ để định vị trong ma trận đặc trưng.
* **Danh sách Postings (Postings List)**:
  Ánh xạ từ mỗi `term` đến mảng các cặp giá trị tài liệu:
  $$\text{term} \rightarrow \left\{ nDoc: df_t, \text{postings: } [(doc\_id_1, f_{t,d_1}), (doc\_id_2, f_{t,d_2}), ...] \right\}$$
  * Trong mô hình VSM: $f_{t,d}$ là trọng số TF-IDF đã được chuẩn hóa L2 ($w_{t,d}'$) của term trong tài liệu $d$.
  * Trong mô hình BM25: $f_{t,d}$ là tần suất thô (Term Frequency) xuất hiện của term trong tài liệu $d$ để phục vụ tính điểm phi tuyến động trực tuyến.
* **Bảng tra cứu độ dài văn bản (Document Length Lookup Table)**:
  Lưu trữ độ dài thực tế của từng tài liệu trong cơ sở dữ liệu:
  $$\text{doc\_id} \rightarrow \text{length}(d)$$
  Dùng để tính toán hệ số phạt độ dài phi tuyến trực tiếp cho mô hình BM25.

---

### 5.2. Thuật toán Lập Chỉ mục Đảo ngược
Quá trình xây dựng chỉ mục được thực hiện một lần duy nhất ở chế độ ngoại tuyến (Offline). Thuật toán gồm hai giai đoạn chính:

#### 5.2.1. Quy trình các bước lập chỉ mục (Indexing Steps):
* **Các bước tạo từ điển và danh sách posting**:
  1. Duyệt qua từng tài liệu $doc\_id$ từ tập văn bản gốc, áp dụng pipeline tiền xử lý `TermSelection` để nhận danh sách các token chuẩn hóa (terms).
  2. Ghi nhận độ dài (số lượng token) của tài liệu vào bảng tra cứu `DocLengths`.
  3. Sử dụng một cấu trúc đếm (ví dụ: `Counter` hoặc bảng băm) để xác định tần suất thô $f_{t,d}$ của từng term độc bản trong tài liệu.
  4. Duyệt qua các term độc bản vừa tìm được, thêm cặp dữ liệu `(doc_id, f_td)` vào cuối danh sách postings của `Index[term]`.
  
* **Các bước tính toán số liệu và lưu trữ chúng**:
  1. Tính tổng số tài liệu $N$ và tổng độ dài của toàn bộ tập dữ liệu để suy ra độ dài trung bình tài liệu:
     $$avgdl = \frac{\sum_{d \in D} |d|}{N}$$
  2. Duyệt qua mỗi term trong từ điển `Index` để tính toán giá trị Document Frequency ($df_t$), được xác định bằng chính độ dài của postings list của term đó.
  3. Tính toán và lưu trữ giá trị IDF toàn cục cho từng term:
     * Với VSM: $IDF(t) = \ln(N / df_t)$
     * Với BM25: $IDF_{BM25}(t) = \ln((N - df_t + 0.5)/(df_t + 0.5) + 1)$
  4. **Đối với mô hình VSM**: Tính toán ma trận trọng số TF-IDF cho từng tài liệu dựa trên TF chuẩn hóa và IDF toàn cục, sau đó tính chuẩn Euclid (L2-norm) của vector tài liệu $\|\vec{d}\|_2$. Thực hiện chuẩn hóa L2 trọng số của từ $w_{t,d}' = w_{t,d} / \|\vec{d}\|_2$ và lưu trữ postings list hoàn chỉnh dưới dạng cặp `(doc_id, w_td')`.
  5. Lưu trữ chỉ mục đảo ngược hoàn chỉnh vào bộ nhớ để phục vụ truy hồi.

#### 5.2.2. Lập chỉ mục đảo ngược cho VSM (TF-IDF chuẩn hóa L2)
```text
ALGORITHM BuildVSMInvertedIndex(ProcessedDocuments docs, VocabularyIndex vocab_index)
INPUT: Tập tài liệu docs, Chỉ mục từ vựng vocab_index
OUTPUT: Chỉ mục đảo ngược VSM chứa trọng số TF-IDF chuẩn hóa L2

1. N ← Số lượng tài liệu trong docs
2. DF ← Khởi tạo từ điển đếm số tài liệu chứa từ (khởi trị 0)
3. For each doc_id, tokens in docs:
       For each unique term t in tokens:
           DF[t] ← DF[t] + 1

4. DocumentVectors ← Khởi tạo từ điển rỗng
5. For each doc_id, tokens in docs:
       # Tạo vector TF thô
       tf_vec ← Khởi tạo vector kích thước |vocab_index| toàn giá trị 0
       For each token in tokens:
           If token in vocab_index:
               tf_vec[vocab_index[token]] ← tf_vec[vocab_index[token]] + 1
       
       # Chuẩn hóa TF theo độ dài tài liệu
       tf_vec ← tf_vec / length(tokens)
       
       # Nhân thêm IDF toàn cục
       DocVector ← Khởi tạo vector kích thước |vocab_index| toàn 0
       For each term in unique(tokens):
           If term in vocab_index:
               idx ← vocab_index[term]
               idf_t ← ln(N / DF[term])
               DocVector[idx] ← tf_vec[idx] * idf_t
               
       # Tính độ dài vector L2 để chuẩn hóa
       L2_norm ← sqrt(sum(DocVector^2))
       If L2_norm > 0:
           DocumentVectors[doc_id] ← DocVector / L2_norm
       Else:
           DocumentVectors[doc_id] ← DocVector

6. VSM_Index ← Khởi tạo từ điển rỗng
7. For each term, idx in vocab_index:
       postings_t ← Khởi tạo danh sách rỗng
       For each doc_id in docs:
           weight_val ← DocumentVectors[doc_id][idx]
           If weight_val > 0:
               Append (doc_id, weight_val) vào postings_t
       VSM_Index[term] ← {nDoc: DF[term], postings: postings_t}

8. Return VSM_Index
```

#### 5.2.3. Lập chỉ mục đảo ngược cho Okapi BM25 (TF thô)
```text
ALGORITHM BuildBM25InvertedIndex(ProcessedDocuments docs)
INPUT: Tập tài liệu đã qua tiền xử lý docs = {doc_id: [tokens...]}
OUTPUT: Chỉ mục đảo ngược Index, Bảng tra độ dài DocLengths, Bảng tra IDF

1. Index ← Khởi tạo từ điển rỗng (defaultdict(list))
2. DocLengths ← Khởi tạo từ điển rỗng
3. IDF ← Khởi tạo từ điển rỗng
4. TotalLength ← 0
5. N ← Số lượng tài liệu trong docs

6. For each doc_id, tokens in docs:
       DocLengths[doc_id] ← Độ dài của tokens
       TotalLength ← TotalLength + DocLengths[doc_id]
       
       TokenFrequencies ← CountFrequencies(tokens) 
       For each term, frequency in TokenFrequencies:
           Append (doc_id, frequency) vào Index[term]

7. AvgDocLength ← TotalLength / N

8. For each term, postings in Index:
       df_t ← Độ dài của postings
       IDF[term] ← ln((N - df_t + 0.5) / (df_t + 0.5) + 1.0)

9. Return Index, DocLengths, IDF, AvgDocLength
```

---

### 5.3. Thuật toán xử lý câu truy vấn (Query Processing & Scoring)
Quá trình xử lý câu truy vấn diễn ra trực tuyến (Online) khi nhận câu truy vấn từ người dùng. Thuật toán được thực hiện qua hai pha tuần tự:

#### 5.3.1. Quy trình các bước xử lý câu truy vấn:
* **Xác định tài liệu liên quan (Candidate Document Identification)**:
  1. Áp dụng tiền xử lý `TermSelection` cho câu truy vấn $Q$ để nhận được danh sách `QueryTokens`.
  2. Khởi tạo một tập hợp rỗng `CandidateDocuments` chứa các tài liệu ứng viên.
  3. Duyệt qua từng từ khóa $t \in QueryTokens$:
     * Tra cứu $t$ trong chỉ mục đảo ngược. Nếu tồn tại, lấy danh sách postings của $t$.
     * Thêm tất cả các $doc\_id$ xuất hiện trong danh sách postings của $t$ vào tập `CandidateDocuments`.
  4. Trả về tập `CandidateDocuments` (các tài liệu không chứa bất kỳ từ khóa truy vấn nào đều bị loại bỏ ngay lập tức, tối ưu hóa đáng kể tốc độ truy xuất).

* **Tính giá trị xếp hạng (Ranking Computation)**:
  1. Khởi tạo bảng băm điểm số rỗng `Scores` cho các tài liệu ứng viên.
  2. Duyệt qua từng từ khóa độc bản trong `QueryTokens`:
     * Tra cứu postings list của từ khóa trong chỉ mục.
     * Với mỗi tài liệu ứng viên $doc\_id$ có chứa từ khóa đó, tính đóng góp điểm số (weight contribution):
       * Đối với VSM: Nhân trọng số TF-IDF của từ khóa trong query với trọng số đã chuẩn hóa L2 của từ khóa trong tài liệu ($w_{t,q} \times w_{t,d}'$), rồi cộng tích lũy vào `Scores[doc_id]`.
       * Đối với BM25: Nhân giá trị $IDF_{BM25}(t)$ với tỷ số bão hòa tần suất có phạt độ dài của tài liệu, rồi cộng tích lũy vào `Scores[doc_id]`.
  3. Đối với mô hình VSM, sau khi cộng dồn, chia toàn bộ điểm số trong `Scores[doc_id]` cho độ dài vector câu truy vấn $\|\vec{q}\|_2$ để tính đúng giá trị Cosine Similarity lý thuyết đầy đủ.
  4. Sắp xếp các tài liệu trong `Scores` theo thứ tự điểm số giảm dần.
  5. Trả về Top $k$ kết quả cao nhất làm đầu ra của quá trình tìm kiếm.

#### 5.3.2. Truy xuất và tính điểm với mô hình không gian vector (VSM)
```text
ALGORITHM SearchVSM(QueryText Q, VSM_Index, tfidf_matrix, vocab_index, k)
INPUT: Câu truy vấn Q, chỉ mục đảo ngược VSM, ma trận tfidf của docs, số lượng kết quả k
OUTPUT: Top k tài liệu có điểm Cosine Similarity cao nhất

1. QueryTokens ← TermSelection(Q, STOP_WORDS, ABBREVIATIONS, STEMMER)
2. QueryVector ← Vector kích thước |vocab_index| toàn giá trị 0
3. For each token in QueryTokens:
       If token in vocab_index:
           QueryVector[vocab_index[token]] ← QueryVector[vocab_index[token]] + 1

4. QueryVector ← QueryVector / length(QueryTokens) # Chuẩn hóa tần suất
5. CandidateDocuments ← Khởi tạo tập hợp rỗng

# Bước 1: Nhân thêm IDF (NTC weighting) và tìm tài liệu ứng viên
6. For each term in unique(QueryTokens):
       If term in VSM_Index:
           term_idx ← vocab_index[term]
           df ← VSM_Index[term]["nDoc"]
           idf ← ln(N / df)
           QueryVector[term_idx] ← QueryVector[term_idx] * idf
           
           For each doc_id, _ in VSM_Index[term]["postings"]:
               Thêm doc_id vào CandidateDocuments

# Bước 2: Tính Cosine Similarity giữa câu truy vấn và các tài liệu ứng viên
7. Similarities ← Khởi tạo danh sách rỗng
8. For each doc_id in CandidateDocuments:
       doc_vector ← tfidf_matrix[doc_id - 1] # Vector tài liệu đã được chuẩn hóa L2
       
       dot_product ← dot(QueryVector, doc_vector)
       query_norm ← sqrt(sum(QueryVector^2))
       doc_norm ← sqrt(sum(doc_vector^2)) # doc_norm luôn bằng 1.0 do chuẩn hóa trước
       
       If query_norm > 0 and doc_norm > 0:
           similarity ← dot_product / (query_norm * doc_norm)
       Else:
           similarity ← 0.0
       Append (doc_id, similarity) vào Similarities

9. SortedResults ← Sắp xếp Similarities theo similarity giảm dần
10. Return Top k phần tử của SortedResults
```

#### 5.3.3. Truy xuất và tính điểm với mô hình Okapi BM25
```text
ALGORITHM SearchBM25(QueryText Q, Index, DocLengths, IDF, AvgDocLength, k)
INPUT: Câu truy vấn Q, các cấu trúc chỉ mục đảo ngược, số lượng kết quả cần trả về k
OUTPUT: Top k tài liệu có điểm số cao nhất

1. QueryTokens ← TermSelection(Q, STOP_WORDS, ABBREVIATIONS, STEMMER)
2. Scores ← Khởi tạo bảng băm rỗng (defaultdict(float))

3. For each term in QueryTokens:
       If term NOT in Index:
           Continue
       
       idf_t ← IDF[term]
       If idf_t ≤ 0: # Bỏ qua từ quá phổ biến
           Continue
           
       For each doc_id, f in Index[term]:
           dl ← DocLengths[doc_id]
           numerator ← f * (k1 + 1)
           denominator ← f + k1 * (1 - b + b * dl / AvgDocLength)
           term_score ← idf_t * (numerator / denominator)
           
           Scores[doc_id] ← Scores[doc_id] + term_score

4. SortedResults ← Sắp xếp các cặp (doc_id, score) trong Scores theo score giảm dần
5. Return Top k phần tử của SortedResults
```

---

## 6. Thử nghiệm và Đánh giá Hiệu năng Hệ thống

Hệ thống được đánh giá bằng phương pháp kiểm thử toàn diện trên toàn bộ **225 câu truy vấn** của tập Cranfield.

### 6.1. Phương pháp cải tiến: Cluster-based Reranking
Để cải thiện độ chính xác, hệ thống áp dụng kỹ thuật Reranking dựa trên phân cụm tài liệu (KMeans Clustering):
1. **Biểu diễn ngữ nghĩa tài liệu (LSA space)**: Ma trận TF-IDF của 1.400 tài liệu được giảm chiều bằng phương pháp **TruncatedSVD** xuống không gian 100 chiều để trích xuất các đặc trưng ngữ nghĩa tiềm ẩn (LSA space), sau đó thực hiện chuẩn hóa L2.
2. **Phân cụm**: Sử dụng thuật toán KMeans phân chia 1.400 tài liệu thành **200 cụm** tài liệu nhỏ (mỗi cụm trung bình chứa ~7 tài liệu nhằm đảm bảo độ chính xác của tiểu chủ đề - *micro-topic*).
3. **Cơ chế bỏ phiếu động (Online Voting)**:
   * Khi nhận câu truy vấn, mô hình BM25 hoặc VSM sẽ trả về bảng xếp hạng thô.
   * Lấy Top 20 tài liệu đầu tiên để "bỏ phiếu" chọn ra các cụm tài liệu quan trọng nhất đối với truy vấn.
   * Điểm số của cụm $c$ được chuẩn hóa: $\text{cluster\_score}[c] = \text{count}[c] / \text{max\_count}$.
4. **Tích hợp điểm lai (Hybrid Scoring)**:
   * Điểm số truy hồi thô của tài liệu được chuẩn hóa về $[0, 1]$: $\text{norm\_score}[d] = \text{score}[d] / \text{max\_score}$.
   * Kết hợp điểm số theo công thức:
     $$\text{final\_score}[d] = \alpha \cdot \text{norm\_score}[d] + (1 - \alpha) \cdot \text{cluster\_score}[\text{cluster}(d)]$$
   * Tham số tối ưu hóa thực nghiệm: $\alpha = 0.85$.

### 6.2. Bảng kết quả so sánh hiệu năng (Final Evaluation)

| Chỉ số đánh giá | VSM Baseline | VSM + Cluster Reranking | BM25 Baseline | BM25 + Cluster Reranking |
|:---|:---:|:---:|:---:|:---:|
| **MAP** | 0.2923 | **0.3045** *(+4.17%)* | 0.3118 | **0.3297** *(+5.74%)* |
| **P@20** | 0.1573 | **0.1660** *(+5.53%)* | 0.1622 | **0.1720** *(+6.04%)* |
| **Recall@20** | 0.5048 | **0.5305** *(+5.09%)* | 0.5182 | **0.5500** *(+6.13%)* |

**Nhận xét**: 
* **BM25 vượt trội hơn VSM** ở tất cả các khía cạnh nhờ cơ chế bão hòa tần suất và hiệu chỉnh độ dài tài liệu tối ưu.
* **Cluster Reranking cải thiện đồng loạt tất cả các mô hình**. Điểm MAP của BM25 tăng từ 0.3118 lên **0.3297** (+5.74%). Recall@20 tăng lên **0.5500**, giúp tìm thấy thêm hơn 3% tài liệu liên quan thực sự trong Top 20 kết quả trả về.

---

## 7. Minh họa Tính toán Chạy tay trên Query 1
Để làm sáng tỏ quy trình vận hành chi tiết của hệ thống, dưới đây là phần phân tích từng bước tính toán đối với **Query 1**:

### 7.1. Chạy tay Pipeline Tiền xử lý (Query 1)
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

### 7.2. Kết quả Xếp hạng và Điểm số Top 5 (Query 1)

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

### 7.3. Chạy tay đánh giá độ chính xác (Evaluation) cho Query 1 với k=5
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

## 8. Phân tích các trường hợp truy vấn Tốt nhất và Kém nhất
Dựa trên tệp thống kê hiệu năng chi tiết của từng truy vấn (`evaluation_per_query.csv`), chúng tôi phân tích nguyên nhân tạo nên sự chênh lệch lớn giữa các nhóm kết quả:

### 8.1. Các truy vấn có kết quả Tốt nhất (AP ≈ 1.0)
* **Đặc trưng**: Câu truy vấn chứa các thuật ngữ khí động học đặc thù, rất hiếm gặp trên toàn hệ thống (IDF cực lớn) như `photoelasticity`, `buckling`, `supersonic nozzle`.
* **Lý do thành công**:
  * Các từ khóa này chỉ xuất hiện trong một số rất ít tài liệu cụ thể. Mô hình chỉ mục đảo ngược lập tức loại bỏ hầu hết các tài liệu không liên quan, đưa các tài liệu chứa từ khóa này lên top đầu với điểm số áp đảo.
  * Sự trùng khớp từ khóa hiếm mang lại tín hiệu ngữ nghĩa cực kỳ mạnh mẽ, không bị ảnh hưởng bởi độ dài hay tần suất của các từ thông thường.

### 8.2. Các truy vấn có kết quả Kém nhất (AP ≈ 0.0)
* **Đặc trưng**: Câu truy vấn có xu hướng ngắn và sử dụng các từ ngữ chung chung như `boundary layer theory`, `experimental research`, `high speed flow`.
* **Lý do thất bại**:
  * **Sự mơ hồ về ngữ nghĩa (Semantic Ambiguity)**: Các từ khóa như `layer`, `flow`, `speed` xuất hiện ở hàng trăm tài liệu khác nhau trong tập Cranfield (DF lớn, IDF nhỏ), dẫn đến việc tính điểm bị phân tán và nhiễu.
  * **Thiếu từ khóa đặc trưng (Vocabulary Mismatch)**: Tài liệu liên quan thực sự có thể viết về các khía cạnh hẹp cụ thể như `viscous fluid flow`, `Prandtl number` (IDF lớn) nhưng không chứa từ khóa chung chung `speed` hay `research` mà người dùng nhập vào.
  * **Giải pháp khắc phục**: Cần áp dụng kỹ thuật mở rộng câu truy vấn (Query Expansion) bằng cách sử dụng các từ điển đồng nghĩa hoặc phản hồi liên quan giả định (Pseudo Relevance Feedback) để bổ sung thêm các term đặc trưng ngữ nghĩa.

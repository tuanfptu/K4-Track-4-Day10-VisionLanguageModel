# Báo cáo Lab Day 10 — Qwen3-VL-4B

- **Họ tên:** Hà Mạnh Tuân
- **Dataset version:** `d10-21e4070301`
- **Setup SFT được chọn theo validation:** E1

## Cách đọc kết quả

Các số trong bảng được chấm trên test gồm 311 ảnh, với `scripts/score.py`: ghép box cùng lớp khi IoU ≥ 0.5, sau đó tính micro precision/recall/F1. CI95 được bootstrap 1.000 lần bằng cách lấy mẫu lại theo ảnh (seed 0). `parse hợp lệ` là tỷ lệ output được parser đọc thành JSON list; chỉ số này không đo độ chính xác vị trí box. Evidence đầy đủ nằm trong `report/evidence/`.

## Bảng kết quả

### Chất lượng dự đoán trên test

| Setup | Parse hợp lệ | TP / FP / FN | Micro F1 [CI95] | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 — zero-shot | 0.511 | 26 / 303 / 557 | 0.057 [0.037, 0.080] | 0.97 | 0.144 | 0.000 | 0.104 |
| E1 — LoRA LLM | 0.990 | 148 / 374 / 435 | 0.268 [0.232, 0.308] | 1.20 | 0.188 | 0.210 | 0.490 |
| E2 — E1 + projector | 0.720 | 47 / 660 / 536 | 0.073 [0.053, 0.095] | 2.12 | 0.162 | 0.013 | 0.249 |
| E3 — E2 + LoRA ViT | 0.884 | 4 / 955 / 579 | 0.005 [0.001, 0.011] | 3.07 | 0.000 | 0.006 | 0.000 |
| E4 — data_size 1.0 | 0.990 | 148 / 374 / 435 | 0.268 [0.232, 0.308] | 1.20 | 0.188 | 0.210 | 0.490 |
| E4 — data_size 0.1 | 0.994 | 132 / 637 / 451 | 0.195 [0.163, 0.229] | 2.05 | 0.138 | 0.115 | 0.480 |
| E5 — GRPO từ E1 | 0.994 | 154 / 368 / 429 | 0.279 [0.242, 0.319] | 1.18 | 0.203 | 0.215 | 0.512 |

### Thiết lập và tài nguyên

| Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Thời gian train (phút) | Ghi chú |
|---|---:|---:|---:|---|
| E0 | 0 | không ghi nhận | không train | Chỉ chạy suy luận zero-shot. |
| E1 | 33,030,144 | 5.85 | 21.3 | 300 optimizer steps; batch 2 × tích lũy 4; seed 42. |
| E2 | 87,609,344 | 6.06 | 19.5 | E1 cộng projector được mở huấn luyện. |
| E3 | 93,900,800 | 6.77 | 23.7 | E2 cộng LoRA trên ViT. |
| E4 — data_size 1.0 | 33,030,144 | 5.85 | 21.3 | Tái sử dụng checkpoint và evidence E1; không có lượt train E4 riêng. |
| E4 — data_size 0.1 | 33,030,144 | 6.33 | 21.0 | 10% ảnh train; 300 optimizer steps; seed 42. |
| E5 | 33,030,144 | 3.77 | 15.8 | GRPO 60 bước, 4 generations; khởi tạo từ E1. |

VRAM và thời gian lấy từ `meta` trong các file evidence. E0 không có VRAM trong evidence nên để “không ghi nhận”; thời gian 0 không áp dụng vì đây không phải lượt train. E4 100% giống E1 vì cùng cấu hình, seed và checkpoint; số liệu này là baseline đầy đủ dữ liệu, không phải lần chạy độc lập thứ hai.

## Trả lời 10 câu hỏi

### Câu 1 — Kiến trúc ViT → projector → LLM

Trong mô hình, **vision tower/ViT** trích xuất đặc trưng không gian từ ảnh; **projector** (`visual.merger` và `visual.deepstack_merger_list`) gộp, biến đổi các đặc trưng ảnh sang biểu diễn mà LLM có thể tiếp nhận; **LLM** kết hợp đặc trưng ảnh với prompt và sinh danh sách JSON box. Trong lần chạy này, script đếm được 155,247,616 tham số ở vision tower, 54,579,200 ở projector và 2,205,810,176 ở nhánh language model (tổng số đếm theo `named_parameters()` của model đã nạp: 2,415,636,992; đây là số đếm trong cấu hình đang chạy, không phải tổng tham số dense gốc của model).

E1 huấn luyện LoRA ở 252 projection của LLM, tổng 33,030,144 tham số. E2 giữ LoRA đó và mở projector, tổng 87,609,344 tham số. E3 thêm LoRA tại 96 projection attention/MLP của ViT, tổng 93,900,800 tham số. Như vậy E1 không cập nhật ViT/projector; E2 cập nhật projector; E3 cập nhật thêm một phần ViT.

### Câu 2 — Lỗi của E0

E0 có `parse_valid_rate=0.511`, tức 159/311 output được parser đọc hợp lệ. Hai nhóm lỗi được đếm nhiều nhất là JSON sai/hỏng cấu trúc ở 152/311 ảnh và trả `[]` ở 61/311 ảnh dù ảnh có nhãn. Một output lỗi được lưu làm ví dụ bắt đầu như sau; đoạn hiển thị bị cắt giữa trường `bbox`, còn parser đánh dấu toàn bộ output gốc là không hợp lệ:

```text
[{"label":"manhole","bbox_2d":[259,630,319,660]},
 {"label":"manhole","bbox_2d":[375,630,425,660]},
 {"label":"crack","bbox_2d":[319,630,375,660]}, ...
 {"label":"crack","bbox
```

Đây là thống kê trên output E0, không phải kết luận mọi lỗi của E0 đều do JSON: các output parse được vẫn có thể sai lớp, thiếu box hoặc định vị lệch.

### Câu 3 — Thang tọa độ

Nhãn và prompt dùng tọa độ chuẩn hóa 0–1000, vì thế scorer cũng phải diễn giải box theo `norm1000`. Trên cùng output E0 của 311 ảnh test, F1 là 0.057 khi chấm đúng `norm1000`, và 0.000 khi diễn giải nhầm các số đó thành pixel. Scorer pixel quy đổi tọa độ theo kích thước ảnh 640×360; dùng nhầm thang làm box lệch kích thước/vị trí và gần như không khớp IoU. Phép đối chiếu chỉ đổi `coord` của scorer, giữ nguyên output dự đoán.

### Câu 4 — Mask loss

Khi tạo mẫu SFT, label của prompt và token đầu vào ảnh được mask bằng `-100`; loss chỉ áp lên phần JSON assistant cần sinh. Nếu tính loss trên prompt, mô hình bị yêu cầu học lại đầu vào thay vì tập trung sinh đáp án. Token ảnh biểu diễn dữ liệu quan sát, không phải token đầu ra có nhãn văn bản để dự đoán. Vì vậy mask giúp mục tiêu huấn luyện khớp với tác vụ phát hiện và xuất box.

### Câu 5 — Projector, NF4 và FP32

E1–E3 dùng QLoRA: phần LLM nền được nạp NF4 4-bit; ViT và `lm_head` được bỏ qua lượng tử hóa. Trong lần train trên RTX 3090 này, forward dùng BF16; các tham số được mở huấn luyện được đưa về FP32 trước khi tạo optimizer. Projector phải được lưu riêng vì adapter PEFT chỉ chứa LoRA, không chứa merger/deepstack merger đã huấn luyện. Nếu chỉ nạp adapter mà bỏ `extra_trainable.pt`, suy luận sẽ dùng projector gốc và không tái tạo đúng checkpoint E2/E3. Trọng số 4-bit bị đóng gói theo biểu diễn lượng tử, không thể cập nhật trực tiếp như tham số projector đầy đủ; do đó projector trainable được giữ FP32 để optimizer cập nhật ổn định.

### Câu 6 — Chia dữ liệu theo phiên quay

Các frame gần nhau trong cùng video có thể gần như trùng cảnh. Chia ngẫu nhiên theo ảnh có thể đưa những frame tương tự vào cả train và test, làm kết quả test lạc quan do rò rỉ cảnh. Chia theo `session` giữ nguyên nhóm frame liên quan trong một split. Split cố định có `dataset_version=d10-21e4070301` và phân bố:

| Split | Ảnh | Pothole boxes | Crack boxes | Manhole boxes |
|---|---:|---:|---:|---:|
| Train | 1,422 | 1,074 | 1,750 | 691 |
| Validation | 276 | 66 | 470 | 103 |
| Test | 311 | 121 | 299 | 163 |

Phân bố lớp không đều: test chỉ có 121 pothole box, trong khi crack có 299. Vì vậy cần xem F1 theo lớp và CI95, không chỉ micro F1.

### Câu 7 — So sánh E1, E2 và E3

E1 đạt F1=0.268 [0.232, 0.308], dùng 5.85 GB VRAM và train 21.3 phút. E2 đạt 0.073 [0.053, 0.095], dùng 6.06 GB và 19.5 phút. E3 đạt 0.005 [0.001, 0.011], dùng 6.77 GB và 23.7 phút. Trên lần chạy này E1 là setup SFT tốt nhất theo validation và cũng có F1 test cao nhất; test F1 E2 thấp hơn E1 0.195, E3 thấp hơn 0.263. CI95 từng setup không giao nhau, củng cố thứ hạng mô tả của lần chạy này; để kiểm định chênh lệch một cách trực tiếp trên cùng ảnh test, paired bootstrap sẽ phù hợp hơn so với chỉ so CI riêng.

E3 có parse hợp lệ 0.884 nhưng chỉ 4 TP và 955 FP. JSON đọc được không có nghĩa box đã được định vị đúng. E2/E3 cho thấy mở thêm tham số không tự động cải thiện grounding trong cấu hình 300 bước này.

### Câu 8 — CI95 và độ chắc chắn

Test gồm 311 ảnh; CI95 trong evidence là bootstrap 1,000 lần trên ảnh. Độ rộng CI95 (cận trên trừ cận dưới) lần lượt là: E0 0.043; E1 0.077; E2 0.043; E3 0.009. CI hẹp của E3 đi cùng F1 gần 0, không có nghĩa mô hình tốt; nó phản ánh bootstrap quanh kết quả thấp này. CI theo ảnh cũng không loại bỏ biến thiên do một split cụ thể, và các lớp có ít box như pothole kém ổn định hơn. Khi so hai setup trên cùng test, nên bootstrap chênh lệch theo cặp ảnh thay vì coi hai CI riêng là kiểm định ý nghĩa thống kê.

### Câu 9 — Lỗi theo lớp và ảnh minh họa

E1 là SFT được chọn theo validation. Trên test, pothole có F1 thấp nhất (0.188), precision 0.254, recall 0.149 với 18 TP, 53 FP và 103 FN; lớp này có 121 ground-truth box. Diện tích trung vị box pothole là 0.419% diện tích khung hình, nên các box nhỏ dễ bị bỏ sót hoặc lệch IoU. Crack đạt F1=0.210; manhole đạt 0.490. Sáu ảnh dưới là minh họa validation của E1: ba ảnh dự đoán đúng và ba ảnh có lỗi; GT màu đỏ, dự đoán màu xanh. Đây là ví dụ định tính, không thay cho metrics trên test.

![Ảnh 1: box đúng màu đỏ, dự đoán màu xanh](report/figures/01_correct_vlcsnap-2025-02-18-18h30m33s080.jpg)

![Ảnh 2: box đúng màu đỏ, dự đoán màu xanh](report/figures/02_correct_vlcsnap-2025-02-18-18h30m34s270.jpg)

![Ảnh 3: box đúng màu đỏ, dự đoán màu xanh](report/figures/03_correct_vlcsnap-2025-02-18-18h31m04s291.jpg)

![Ảnh 4: box đúng màu đỏ, dự đoán màu xanh](report/figures/04_error_vlcsnap-2025-02-18-18h30m17s408.jpg)

![Ảnh 5: box đúng màu đỏ, dự đoán màu xanh](report/figures/05_error_vlcsnap-2025-02-18-18h30m20s293.jpg)

![Ảnh 6: box đúng màu đỏ, dự đoán màu xanh](report/figures/06_error_vlcsnap-2025-02-18-18h30m22s081.jpg)

### Câu 10 — E4 ablation data_size

E4 chỉ đổi tỷ lệ dữ liệu train, giữ setup LoRA LLM, rank 16, seed 42 và 300 bước. Với 100% dữ liệu, điểm gốc được tái sử dụng từ E1: F1=0.268 [0.232, 0.308]. Với 10%, F1=0.195 [0.163, 0.229], thấp hơn 0.073; FP/ảnh tăng từ 1.20 lên 2.05. Hai CI riêng gần chạm nhưng không giao nhau ở các giá trị chưa làm tròn; đây là tín hiệu mô tả rằng 10% dữ liệu kém hơn trong lần chạy này. Tuy nhiên 100% là checkpoint E1 đã có, không phải lượt lặp độc lập, và phép so CI riêng không thay cho paired bootstrap. Kết quả phù hợp với giả thuyết dữ liệu ít làm suy giảm chất lượng, nhưng cần thận trọng khi khái quát.

## E5 — Bonus GRPO

E5 khởi tạo từ adapter E1, dùng reward IoU, 4 generations, completion tối đa 128 token, learning rate `5e-6`, `beta=0` và 60 bước. Reward trả 0 khi JSON không parse được; trường hợp cả dự đoán và ground truth rỗng được 1; trường hợp còn lại dùng `0.2 + 0.8 × tổng IoU các cặp khớp / max(số GT, số dự đoán, 1)`, với ghép cùng lớp và IoU ≥ 0.5. Các ca unit test cho reward nằm trong `student/test_reward.py` (JSON đúng hoàn hảo, JSON hỏng, sai lớp, box thừa và cả hai rỗng).

Tóm tắt log trainer trong `report/evidence/e5.json`: 60/60 bước, runtime 947.52 giây (15.8 phút), `train_loss=0.04431`, 0.063 bước/giây; VRAM đỉnh 3.77 GB. Trên test, E5 đạt F1=0.279 [0.242, 0.319], so với E1 là 0.268 [0.232, 0.308], chênh `+0.011`. E5 có 154 TP, 368 FP, 429 FN; E1 có 148 TP, 374 FP, 435 FN. Khoảng CI95 giao nhau rộng, nên chưa đủ bằng chứng để kết luận GRPO cải thiện chắc chắn.

Kiểm tra hai dạng reward hacking đơn giản trên test: E5 trả `[]` ở 0/311 ảnh có nhãn; 0/522 box dự đoán chiếm hơn 50% diện tích ảnh. Điều này không phát hiện hai dạng hành vi đó, nhưng không loại trừ các dạng reward hacking khác. Cần xem thêm reward theo bước và ảnh validation trước khi kết luận. `train_loss` của GRPO là mục tiêu trainer, không so trực tiếp với SFT loss.

## Nhìn lại

E3 làm loss giảm từ 1.2446 ở bước đầu xuống `mean_last_20_loss=0.6839`, nhưng F1 test chỉ 0.0052 với 955 false positive. Điều này cho thấy loss giảm và JSON parse được chưa đảm bảo box được định vị đúng. Tôi nghi việc mở thêm LoRA trên ViT làm lệch grounding trong cấu hình 300 bước, nhưng đây chỉ là giả thuyết từ một lần chạy. Có thể kiểm tra tiếp bằng ảnh dự đoán theo lớp, theo dõi F1 validation qua checkpoint và thử nhiều seed trước khi kết luận về tác động của fine-tune ViT.

## Kết luận

Trong các setup SFT đã chạy, E1 là lựa chọn tốt nhất: F1 test tăng 0.211 so với E0, parse hợp lệ tăng từ 0.511 lên 0.990, trong khi VRAM đỉnh là 5.85 GB. E2/E3 cho kết quả kém hơn rõ trên bộ test này dù mở thêm projector/ViT. E4 cho thấy run 10% dữ liệu thấp hơn baseline đầy đủ dữ liệu; E5 có điểm F1 nhỉnh hơn E1 nhưng CI95 còn giao nhau nên chưa thể khẳng định có cải thiện đáng tin cậy. Các kết luận chỉ áp dụng cho split `d10-21e4070301`, seed và ngân sách train đã ghi ở trên.

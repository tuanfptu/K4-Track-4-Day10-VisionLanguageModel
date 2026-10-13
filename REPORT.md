# Báo cáo Lab Day 10 — Qwen3-VL-4B

Họ tên: Hà Mạnh Tuân
`dataset_version`: `d10-21e4070301`
Setup SFT được chọn theo validation: **E1**

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot | 0 | không ghi nhận | 0 (không train) | 0.057 | [0.037, 0.080] | 0.97 | 0.144 | 0.000 | 0.104 |
| E1 | LoRA LLM | 33030144 | 5.85 | 21.3 | 0.268 | [0.232, 0.308] | 1.20 | 0.188 | 0.210 | 0.490 |
| E2 | + projector | 87609344 | 6.06 | 19.5 | 0.073 | [0.053, 0.095] | 2.12 | 0.162 | 0.013 | 0.249 |
| E3 | + LoRA ViT | 93900800 | 6.77 | 23.7 | 0.005 | [0.001, 0.011] | 3.07 | 0.000 | 0.006 | 0.000 |
| E4 100% | data_size=1.0 | 33030144 | 5.85 | 21.3 | 0.268 | [0.232, 0.308] | 1.20 | 0.188 | 0.210 | 0.490 |
| E4 10% | data_size=0.1 | 33030144 | 6.33 | 21.0 | 0.195 | [0.163, 0.229] | 2.05 | 0.138 | 0.115 | 0.480 |
| E5 | GRPO | 33030144 | 3.77 | 15.8 | 0.279 | [0.242, 0.319] | 1.18 | 0.203 | 0.215 | 0.512 |


Ghi chú: E0 chỉ chạy baseline suy luận; e0.json không ghi VRAM đỉnh nên mục này để không ghi nhận. Thời gian 0 phút chỉ có nghĩa là không có bước huấn luyện. E4 data_size=1.0 tái sử dụng checkpoint E1; 21.3 phút là thời gian train checkpoint E1 nguồn, không phải lượt train E4 mới.

## Trả lời 10 câu hỏi

### Câu 1 — Kiến trúc
Theo cây tham số được đếm trong lần chạy này: vision tower 155,247,616, projector 54,579,200, LLM 2,205,810,176; tổng 2,415,636,992. E1 cập nhật LoRA trên 252 projection của LLM; E2 cập nhật cùng LoRA và merger/deepstack merger; E3 bổ sung LoRA cho 96 projection attention/MLP của ViT.

### Câu 2 — E0
E0 parse_valid_rate = 0.511. Hai nhóm output/lỗi phổ biến nhất theo bộ đếm: JSON lỗi hoặc sai cấu trúc: 152/311 ảnh; Trả [] khi ảnh có nhãn: 61/311 ảnh. Ví dụ output thô: `[{"label": "manhole", "bbox_2d": [259, 630, 319, 660]}, {"label": "manhole", "bbox_2d": [375, 630, 425, 660]}, {"label": "crack", "bbox_2d": [319, 630, 375, 660]}, {"label": "crack", "bbox_2d": [375, 630, 425, 660]}, {"label": "crack", "bbox_2d": [425, 630, 475, 660]}, {"label": "crack", "bbox_2d": [475, 630, 525, 660]}, {"label": "crack", "bbox_2d": [525, 630, 575, 660]}, {"label": "crack", "bbox`

### Câu 3 — Thang tọa độ
Chấm cùng output E0 trên test (311 ảnh) theo norm1000 cho F1=0.057; chấm nhầm là pixel cho F1=0.000. Kiểm chứng bằng cách dùng lại chính file e0_test.jsonl và thay duy nhất tham số coord của trọng tài.

### Câu 4 — Mask loss
Mask label bằng -100 ở prompt và token ảnh để loss chỉ tối ưu JSON do assistant tạo. Nếu tính loss cả prompt/vision token, mô hình bị tối ưu để dự đoán lại đầu vào hội thoại thay vì học phát hiện và định dạng box; token ảnh cũng không phải nhãn sinh cần học.

### Câu 5 — Projector và lượng tử hóa
NF4 chỉ lượng tử hóa phần LLM; cấu hình BitsAndBytes bỏ qua visual và lm_head để merger không thành trọng số 4-bit bất khả vi tối ưu như tham số đầy đủ. E2/E3 lưu riêng visual.merger và deepstack_merger_list ở FP32, đồng thời lưu LoRA bằng adapter. FP32 giữ cập nhật projector có độ chính xác cao hơn; khi nạp lại, chương trình khôi phục extra_trainable.pt trước khi suy luận.

### Câu 6 — Split theo phiên quay
Chia theo phiên quay để các frame gần nhau không xuất hiện đồng thời ở train và test; chia ngẫu nhiên có thể rò rỉ cảnh/khung hình và làm F1 lạc quan. train: 1422 ảnh; pothole 1074, crack 1750, manhole 691 box; val: 276 ảnh; pothole 66, crack 470, manhole 103 box; test: 311 ảnh; pothole 121, crack 299, manhole 163 box.

### Câu 7 — So sánh E1/E2/E3
E1 F1 0.268 CI95 [0.232, 0.308], VRAM 5.85 GB, 21.3 phút; E2 F1 0.073 CI95 [0.053, 0.095], VRAM 6.06 GB, 19.5 phút; E3 F1 0.005 CI95 [0.001, 0.011], VRAM 6.77 GB, 23.7 phút. Đọc độ chồng lấp CI95 trước khi kết luận setup tốt hơn.

### Câu 8 — Khoảng tin cậy
Tập test có 311 ảnh. Độ rộng CI95 bootstrap: E0 rộng 0.043; E1 rộng 0.077; E2 rộng 0.043; E3 rộng 0.009. Khoảng tin cậy rộng/chồng lấp nghĩa là thứ hạng nhỏ giữa các setup có thể do biến thiên mẫu; cần ưu tiên độ lớn chênh lệch và CI, không chỉ nhìn điểm F1.

### Câu 9 — Phân tích lỗi theo lớp
Setup được chọn theo val là E1; lớp F1 thấp nhất trên test là pothole (F1=0.188), với 121 box test và diện tích box trung vị 0.42% khung hình. Số mẫu ít làm CI theo lớp kém ổn định; hình minh họa dưới dùng validation, GT đỏ và dự đoán xanh.

### Câu 10 — E4
E4 giữ nguyên setup E1 và chỉ đổi tỷ lệ dữ liệu: 100% F1=0.268 CI95=[0.232, 0.308]; 10% F1=0.195 CI95=[0.163, 0.229]. Đối chiếu chênh lệch F1 với CI95 để đánh giá độ tin cậy.

## Ảnh minh họa validation

Box đúng màu đỏ; dự đoán màu xanh.

![Ảnh 1: box đúng màu đỏ, dự đoán màu xanh](report/figures/01_correct_vlcsnap-2025-02-18-18h30m33s080.jpg)

![Ảnh 2: box đúng màu đỏ, dự đoán màu xanh](report/figures/02_correct_vlcsnap-2025-02-18-18h30m34s270.jpg)

![Ảnh 3: box đúng màu đỏ, dự đoán màu xanh](report/figures/03_correct_vlcsnap-2025-02-18-18h31m04s291.jpg)

![Ảnh 4: box đúng màu đỏ, dự đoán màu xanh](report/figures/04_error_vlcsnap-2025-02-18-18h30m17s408.jpg)

![Ảnh 5: box đúng màu đỏ, dự đoán màu xanh](report/figures/05_error_vlcsnap-2025-02-18-18h30m20s293.jpg)

![Ảnh 6: box đúng màu đỏ, dự đoán màu xanh](report/figures/06_error_vlcsnap-2025-02-18-18h30m22s081.jpg)

## E5 — Bonus GRPO

E5 F1=0.279, CI95=[0.242, 0.319]; so với setup SFT chọn E1 F1=0.268 CI95=[0.232, 0.308]. Trên test, model trả `[]` ở 0/311 ảnh có nhãn; 0/522 box dự đoán chiếm hơn 50% diện tích khung hình. Chênh F1 E5 − SFT = +0.011 (CI95 giao nhau). Đây là dấu hiệu cần xem để đánh giá reward hacking; đối chiếu thêm `work/runs/e5_grpo/trainer` và ảnh validation trước khi kết luận.

## Nhìn lại

Khi xem log và ảnh dự đoán, tôi thấy E3 giảm loss từ 1.2446 ở bước đầu xuống mean_last_20_loss=0.6839 nhưng test F1 chỉ đạt 0.0052 và có 955 false positive. Điều này cho thấy loss giảm và output parse được chưa đảm bảo box được định vị đúng. Tôi nghi việc mở thêm LoRA trên ViT làm lệch grounding trong cấu hình 300 bước này; đây là giả thuyết từ kết quả, không phải kết luận rằng fine-tune ViT luôn có hại.

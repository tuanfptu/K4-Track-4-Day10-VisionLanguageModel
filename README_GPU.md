# Chạy Lab Day 10 trên RTX 3090 PyTorch

Gói này chạy lần lượt **E0, E1, E2, E3, E4 và E5**. E5 dùng GRPO 60 bước. E4 giữ nguyên setup SFT có F1 validation cao nhất rồi so sánh train trên 100% và 10% ảnh; chỉ đổi một biến là `data_size`.

## 1. Giải nén và kiểm tra GPU

Upload `K4-Day10-RTX3090-Package.zip` lên máy thuê, giải nén, mở terminal tại thư mục `K4-Track-4-Day10-VisionLanguageModel`. Chọn image PyTorch đã chốt và gắn RTX 3090. Mặc định storage 100 GB đủ cho dataset, cache model và mã nguồn; chừa dung lượng cho cache Hugging Face.

```bash
nvidia-smi
python --version
```

Đặt giới hạn cache model nếu cần:

```bash
export HF_HOME="$PWD/.hf-cache"
```

Model tải từ Hugging Face trong lần đầu. Nếu tài khoản của bạn cần token, đặt `HF_TOKEN` trong terminal; không ghi token vào file hoặc notebook.

## 2. Cài thư viện và chuẩn bị dữ liệu

`archive.zip` đã được đóng trong gói. Tạo split chuẩn một lần:

```bash
python -m pip install -r requirements-colab.txt
python scripts/prepare_data.py --zip archive.zip
```

Hoặc chạy trình bao gồm cả cài đặt, chuẩn bị data và toàn bộ thí nghiệm:

```bash
bash run_gpu.sh
```

## 3. Chạy E0–E5

```bash
python -m student.run_lab all
```

Lệnh `all` chạy 5 lượt huấn luyện: E1, E2, E3, một lượt E4 với 10% dữ liệu và E5 GRPO; E4 100% tái sử dụng kết quả setup tốt nhất trong E1–E3. Các setup SFT dùng 300 optimizer steps, batch 2, tích lũy gradient 4, seed 42. E5 khởi tạo từ setup tốt nhất theo F1 validation. Thứ tự validation → chọn setup → test được ghi rõ trong log.

RTX 3090 có 24 GB VRAM. Nếu E3 báo hết bộ nhớ, giảm batch nhưng giữ effective batch 8:

```bash
python -m student.run_lab all --train-batch-size 1 --grad-accum 8
```

Để chạy lại riêng baseline E0: `python -m student.run_lab e0`. Để tạo lại báo cáo từ evidence đã có: `python -m student.run_lab report`. `--skip-e5` có thể bỏ lượt GRPO nếu chỉ cần phần bắt buộc, nhưng để hoàn tất yêu cầu E0–E5 hãy chạy mặc định không có cờ này.

## 4. File kết quả

- `report/evidence/e0.json` đến `e3.json`: kết quả test sinh từ `scripts/score.py`.
- `report/evidence/e4_data_size_1.0.json` và `e4_data_size_0.1.json`: hai mức của ablation.
- `report/evidence/e5.json`: kết quả test GRPO.
- `report/figures/`: 6 ảnh validation có box GT đỏ và dự đoán xanh.
- `REPORT.md`: bảng kết quả, 10 câu trả lời được điền bằng metrics thật, phân tích tọa độ và E4/E5.
- `work/runs/`: adapter, projector đã huấn luyện, checkpoint 100 bước, log loss và metadata tài nguyên.
- `work/predictions/`: JSONL output thô trên validation/test.

Không xóa `work/` trước khi tạo bản sao lưu. Adapter LoRA và projector là hai phần riêng; E2/E3 cần cả hai để suy luận đúng.

## 5. Nộp bài

Đọc lại `REPORT.md`, điền họ tên và đoạn “Nhìn lại” bằng trải nghiệm thật của bạn. Chạy `python scripts/check_submission.py`. Lưu ý checker còn yêu cầu evidence nằm trong repo có Git và tối thiểu 5 commit; hãy commit notebook/mã theo tiến độ, nhưng không đưa `data/`, `work/`, token hoặc trọng số lên GitHub. Script chấm, checker, tests và configs của giảng viên được giữ nguyên.

`student/test_reward.py` chứa các ca test reward E5 theo yêu cầu đề. Chạy khi cần kiểm tra độc lập:

```bash
python -m unittest student.test_reward
```

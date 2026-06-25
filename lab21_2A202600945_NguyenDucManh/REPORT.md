# Lab 21 — Evaluation Report (Legal Domain Bonus)

**Học viên**: Nguyễn Đức Mạnh — 2A202600945
**Ngày nộp**: 2026-06-25
**Submission option**: B (HF Hub) + Stretch Goals (Target ALL layers, Custom Domain: Legal)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `duyet/vietnamese-legal-instruct` (QA Pháp lý Việt Nam), 300 samples (270 train + 30 eval)
- **max_seq_length**: 1024
- **GPU**: Tesla T4, 16 GB VRAM (Google Colab Free)
- **Training cost**: $0 (~1.5 giờ @ $0/hr trên Colab Free)
- **HF Hub link**: https://huggingface.co/<username>/<adapter-name>

## 2. Rank Experiment Results
*Lưu ý: Thực nghiệm sử dụng target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] để đáp ứng Stretch Goal target ALL layers trên domain phức tạp (Luật).*

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.41 min   | 2.82 GB   | 0.00054   | 1.00054    |
| 16   | 3,686,400       | 4.98 min   | 2.22 GB   | 0.00005   | 1.00005    |
| 64   | 14,745,600      | 4.60 min   | 3.60 GB   | 0.00001   | 1.00001    |
| Base | -               | -          | -         | -         | -          |

## 3. Loss Curve Analysis
*(Hãy dán hình ảnh loss_curve.png sinh ra từ Colab vào file báo cáo Word/PDF cuối cùng của bạn)*

- **Quan sát**: 
  - Train loss giảm đều đặn và hội tụ rất rõ ràng ở những step cuối.
  - Eval loss cũng đồng thuận giảm theo và đạt ngưỡng rất thấp (gần như 0), không quan sát thấy hiện tượng đường Eval loss bị tăng vọt ngược lại ở epoch cuối, cho thấy chưa có dấu hiệu overfitting nghiêm trọng.
  - Tuy nhiên, do cấu trúc của câu hỏi QA có nhiều đoạn định dạng lặp lại và tập dữ liệu mẫu đánh giá tương đối nhỏ, mô hình học các cấu trúc này rất nhanh khiến Perplexity tiến sát tới ngưỡng hoàn hảo là 1.0.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một kỹ thuật trong trí thông minhhệ máy tính để máy tự học và cải thiện bằng việc tiếp nhận dữ liệu và thực hiện phân tích, dự đoán kết quả dựa trên những gì đã học được. Nó giúp máy tự động học và cải thiện mà không cần phải được hướng dẫn chi tiết từng bước.
**Fine-tuned (r=16/64)**: Machine learning là một loại trí thông minhell, giúp máy tự động học và cải thiện qua thời gian. Nó sử dụng thuật toán và mô hình toán học để phân tích dữ liệu và đưa ra dự đoán dựa trên những gì đã học được trước đó. Trong machine learning, máy không cần được dạy cụ thể từng câu lệnh, mà nó có thể
**Nhận xét**: Cả hai đều giải thích đúng, bản fine-tuned có vẻ tự nhiên hơn một chút nhưng bị cắt cụt do giới hạn max_tokens.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: (Có viết code tính Fibonacci cơ bản bằng vòng lặp for)
**Fine-tuned**: (Có sinh ra đoạn mã nguồn Python tính Fibonacci sử dụng đệ quy)
**Nhận xét**: Bản Base tạo ra cách dùng vòng lặp hiệu quả hơn, trong khi bản fine-tuned tạo ra cách dùng đệ quy (chưa tối ưu cho số lớn nhưng đúng logic).

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Người dùng là trung tâm... 2. Đơn giản và trực quan... 3. Phân phối hiệ...
**Fine-tuned**: 1. Thỏa mãn nhu cầu người dùng... 2. Hiệu quả... 3. Dễ nhớ...
**Nhận xét**: Cả hai đều liệt kê đúng các nguyên tắc, tuy nhiên model bị cắt nội dung vì token length limit khi generate.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA là một phương pháp được sử dụng để làm cho các mô hình NLU nhỏ hơn bằng cách thêm một số tham số nhỏ vào mô hình...
**Fine-tuned**: LoRA cải thiện hiệu suất bằng cách thêm một số lớp mới, trong khi QLoRA sử dụng kỹ thuật phân cấp (Quantization) để giảm số...
**Nhận xét**: Bản Fine-tuned giải thích chính xác từ khóa "Quantization" cho QLoRA so với bản gốc.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt Engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách tối ưu hóa việc cung cấp dữ liệu... RAG là một cách để lập trí máy học...
**Fine-tuned**: Prompt Engineering là kỹ thuật cải tiến lời đề nghị để mô hình hiểu yêu cầu tốt hơn. RAG là một thuật toán để phân tích dữ liệu bằng cách sử dụng ba nguồn dữ liệu...
**Nhận xét**: Mô hình fine-tuned giải thích khái niệm Prompt Engineering tốt và sát nghĩa tiếng Việt hơn.

## 5. Conclusion về Rank Trade-off
- **Rank nào cho ROI tốt nhất trên dataset này? Tại sao?**
  Rank 16 mang lại hiệu quả đầu tư (ROI) tối ưu nhất. Mặc dù số lượng tham số tăng gấp đôi so với Rank 8, Eval Loss đã giảm đi 10 lần (từ 0.00054 xuống 0.00005), cho thấy mô hình học được các pattern ngôn ngữ phức tạp tốt hơn hẳn.
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
  Điểm "diminishing returns" (quy luật hiệu suất giảm dần) xảy ra rõ ràng khi nâng từ Rank 16 lên Rank 64. Số lượng tham số cần huấn luyện tăng vọt lên hơn 14 triệu và peak VRAM vọt lên 3.6GB, nhưng Perplexity chỉ nhích một mức cực nhỏ (từ 1.00005 xuống 1.00001). Mức độ cải thiện này là hoàn toàn không đáng kể so với tài nguyên phần cứng phải bỏ ra thêm.
- **Recommendation:**
  Khi đưa vào môi trường Production thực tế đối với các domain đòi hỏi cấu trúc câu trả lời chuyên sâu, cấu hình **r=16 target ALL layers** là sự lựa chọn hoàn hảo. Nó mang lại chất lượng văn bản sinh ra vượt trội mà vẫn giữ được sự tinh gọn về dung lượng Adapter và VRAM hệ thống.

## 6. What I Learned
- Tầm quan trọng của việc lựa chọn siêu tham số (Hyperparameters): Không phải cứ nhồi nhét `Rank (r)` càng to thì mô hình càng thông minh. Việc tìm ra ngưỡng tối ưu giúp tiết kiệm được khổng lồ thời gian và chi phí huấn luyện.
- QLoRA kết hợp cùng hệ sinh thái Unsloth là một bước tiến thần kỳ trong việc dân chủ hóa AI, cho phép sinh viên chạy tinh chỉnh các mô hình LLM hàng tỷ tham số ngay trên GPU miễn phí của Google Colab mà không lo lỗi Out-of-Memory.
- Nâng cao kỹ năng MLOps: Nắm được toàn bộ vòng đời (lifecycle) chuẩn xác từ lúc tải dữ liệu định dạng ShareGPT, huấn luyện mô hình với SFTTrainer, so sánh các metrics định lượng, cho đến việc đóng gói và đẩy trực tiếp Adapter lên nền tảng HuggingFace Hub thay vì chỉ lưu cục bộ ở máy cá nhân.

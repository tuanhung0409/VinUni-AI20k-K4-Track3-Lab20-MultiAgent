# Báo cáo Đánh giá So sánh: Single-Agent Baseline vs Multi-Agent Research System

## 1. Tóm tắt Đánh giá (Executive Summary)

Báo cáo này trình bày kết quả đo lường và đánh giá thực nghiệm giữa mô hình **Single-Agent Baseline** (một LLM xử lý từ đầu đến cuối) và mô hình **Multi-Agent Architecture** (hệ thống đa tác nhân gồm *Supervisor, Researcher, Analyst, Writer*) được xây dựng bằng **LangGraph** và vận hành qua OpenRouter với model `google/gemini-2.5-flash`.

Đợt benchmark được thực hiện tự động qua 3 truy vấn nghiên cứu tiêu chuẩn (theo `configs/lab_default.yaml`):
1. **Query 1**: *"Research GraphRAG state-of-the-art and write a 500-word summary"*
2. **Query 2**: *"Compare single-agent and multi-agent workflows for customer support"*
3. **Query 3**: *"Summarize production guardrails for LLM agents"*

---

## 2. Bảng Số liệu Đối sánh Thực nghiệm (Benchmark Quantitative Table)

### A. Chi tiết từng lượt chạy

| Lượt chạy (Run Name) | Truy vấn nghiên cứu | Thời gian (Latency) | Chi phí (Cost USD) | Điểm chất lượng (0-10) | Độ phủ trích dẫn (Citation) | Tỷ lệ lỗi (Failure Rate) | Số lượng Token (In / Out) |
|---|---|---:|---:|---:|---:|---:|---|
| **Q1_Baseline** | GraphRAG SOTA | **8.62s** | **$0.000582** | 8.5 / 10 | 100% | 0% | 45 in / 958 out |
| **Q1_MultiAgent** | GraphRAG SOTA | 37.15s | $0.003942 | **10.0 / 10** | **100%** | 0% | 5,498 in / 5,196 out |
| **Q2_Baseline** | Customer Support Workflows | **24.36s** | **$0.001354** | 8.5 / 10 | 100% | 0% | 36 in / 2,248 out |
| **Q2_MultiAgent** | Customer Support Workflows | 44.63s | $0.004978 | **10.0 / 10** | **100%** | 0% | 6,238 in / 6,737 out |
| **Q3_Baseline** | Production Guardrails | **15.57s** | **$0.001211** | 9.0 / 10 | 100% | 0% | 33 in / 2,010 out |
| **Q3_MultiAgent** | Production Guardrails | 31.68s | $0.003702 | **10.0 / 10** | **100%** | 0% | 5,065 in / 4,903 out |

---

### B. Bảng tổng hợp Trung bình (Averages Comparison)

| Tiêu chí đánh giá | Single-Agent Baseline | Multi-Agent Workflow | Mức độ chênh lệch | Ý nghĩa kiến trúc |
|---|---:|---:|:---:|---|
| **Thời gian phản hồi (Latency TB)** | **16.18s** | 37.82s | **+133% (+21.6s)** | Multi-Agent có độ trễ cao hơn do luân chuyển tuần tự qua 4 bước node. |
| **Chi phí ước tính (Cost TB)** | **$0.001049** | $0.004207 | **~4.0x** | Multi-Agent tiêu tốn token gấp 4 lần do nạp ngữ cảnh trung gian giữa các Agent. |
| **Chất lượng báo cáo (Quality TB)** | 8.67 / 10 | **10.0 / 10** | **+15.3%** | Multi-Agent đạt điểm tuyệt đối nhờ phân tích phản biện và đánh giá nguồn đa chiều. |
| **Độ phủ trích dẫn (Citation Coverage)** | 100% | **100%** | Tương đương | Cả hai đều trích dẫn đầy đủ các nguồn đã cung cấp. |
| **Tỷ lệ thành công (Success Rate)** | 100% (0% Fail) | **100% (0% Fail)** | Hoàn hảo | Không gặp sự cố đứt gãy hoặc timeout. |

---

## 3. Phân tích Các Đánh đổi Kiến trúc (Architectural Trade-offs)

1. **Chiều sâu & Tính phản biện (Depth & Critical Analysis)**:
   - **Multi-Agent**: Đạt hiệu quả vượt trội nhờ **phân vai chuyên trách** (Separation of Concerns). `Researcher` tập trung trích xuất dữ liệu, `Analyst` chỉ ra trade-off ma trận và đánh giá độ tin cậy của tài liệu, `Writer` tổng hợp thành cấu trúc bài báo khoa học.
   - **Single-Agent**: Có xu hướng khái quát hóa bề mặt, dễ bỏ qua việc đánh giá tính chân thực của nguồn hoặc thiếu ma trận so sánh các phương án.

2. **Chi phí tài nguyên & Độ trễ (Cost & Latency)**:
   - **Single-Agent**: Vượt trội hoàn toàn về tốc độ (~16s) và giá thành (~$0.001/query), lý tưởng cho các tác vụ cần phản hồi nhanh.
   - **Multi-Agent**: Phù hợp cho các tác vụ nghiên cứu sâu (Deep Research) mà người dùng sẵn sàng chờ 30-45s để nhận được báo cáo toàn diện.

---

## 4. Phân tích Failure Mode gặp phải và Cách khắc phục (Failure Mode & Fix)

Trong quá trình phát triển và kiểm thử hệ thống Agentic, chúng tôi đã phát hiện và xử lý **3 Failure Modes điển hình**:

### ⚠️ Failure Mode 1: Coordination Overhead & Vòng lặp Vô tận (Infinite Routing Loop)
* **Triệu chứng gặp phải**: Nếu không có điều kiện dừng rõ ràng, Supervisor có thể tiếp tục phân vân và điều phối qua lại giữa `researcher` và `analyst` mãi mãi, gây tốn kém token và tràn bộ nhớ.
* **Nguyên nhân**: Thiếu cơ chế kiểm soát số bước và thiếu trạng thái `done`.
* **Cách khắc phục trong code**:
  - Tại [`src/multi_agent_research_lab/agents/supervisor.py`](file:///d:/VinUni/VinUni-AI20k-K4-Track3-Lab20-MultiAgent/src/multi_agent_research_lab/agents/supervisor.py), triển khai quy tắc kiểm tra điều kiện dừng:
    ```python
    if state.iteration >= settings.max_iterations or state.final_answer is not None:
        next_route = "done"
    ```
  - Tại [`src/multi_agent_research_lab/graph/workflow.py`](file:///d:/VinUni/VinUni-AI20k-K4-Track3-Lab20-MultiAgent/src/multi_agent_research_lab/graph/workflow.py), cài đặt `_route_condition` ép đồ thị ngắt về `END` khi Supervisor phát tín hiệu `done` hoặc chạm ngưỡng `max_iterations = 6`.

---

### ⚠️ Failure Mode 2: Context Drift & Mất mát Ngữ cảnh gốc (Context Drift across Handoffs)
* **Triệu chứng gặp phải**: Khi chuyển dữ liệu từ `Researcher` $\rightarrow$ `Analyst` $\rightarrow$ `Writer`, các agent sau có thể "quên" yêu cầu ban đầu của người dùng mà chỉ tập trung tóm tắt lại văn bản của agent trước.
* **Cách khắc phục trong code**:
  - Đối tượng `ResearchState` duy trì trường `request: ResearchQuery` bất biến.
  - Trong prompt của `AnalystAgent` và `WriterAgent`, luôn tiêm trực tiếp `state.request.query` và `state.request.audience` vào System/User Prompt để đảm bảo mọi suy luận đều hướng về mục tiêu ban đầu.

---

### ⚠️ Failure Mode 3: Nguồn trích xuất rỗng hoặc Thiếu độ tin cậy (Shallow / Ungrounded Sources)
* **Triệu chứng gặp phải**: Khi tìm kiếm bên ngoài bị lỗi hoặc nguồn trả về quá ngắn, LLM sẽ tự suy diễn hoặc từ chối trả lời vì thiếu dữ liệu.
* **Cách khắc phục trong code**:
  - Cài đặt cơ chế Fallback 3 lớp trong [`SearchClient`](file:///d:/VinUni/VinUni-AI20k-K4-Track3-Lab20-MultiAgent/src/multi_agent_research_lab/services/search_client.py):
    1. Ưu tiên gọi **Tavily Search API** nếu có API Key.
    2. Fallback vào **Offline Benchmark Corpus** (30 bộ bài viết chuyên khảo chuyên sâu).
    3. Fallback vào **Deterministic Grounding Mock** với metadata nguồn rõ ràng.

---

## 5. Bằng chứng Tracing & Ảnh chụp Màn hình (Trace Evidence)

Mỗi bước thực thi của từng Agent đều đã được ghi nhận token in/out, latency và metadata tự động lên cloud:

* **LangSmith Dashboard**: [https://smith.langchain.com/](https://smith.langchain.com/)
  - **Tên Project**: `multi-agent-research-lab`
  - Cây DAG thực thi: `Supervisor` $\rightarrow$ `Researcher` $\rightarrow$ `Supervisor` $\rightarrow$ `Analyst` $\rightarrow$ `Supervisor` $\rightarrow$ `Writer` $\rightarrow$ `done`.

### Ảnh chụp màn hình Trace UI:

![LangSmith Trace UI](report/screenshort.png)


# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

### 1. Case nào NÊN dùng multi-agent? Vì sao?

**Nên dùng trong các trường hợp:**
1. **Tác vụ nghiên cứu sâu & tổng hợp phức tạp (Deep Research & Complex Multi-Hop Synthesis)**:
   - *Ví dụ*: Viết báo cáo khoa học, phân tích kiến trúc kỹ thuật đa chiều, thẩm định pháp lý/y tế.
   - *Vì sao*: Đòi hỏi **sự phân vai chuyên biệt (Role Specialization)** giữa các công đoạn: Thu thập bằng chứng (*Researcher*) $\rightarrow$ Phản biện, so sánh trade-off (*Analyst*) $\rightarrow$ Biên tập chuyên nghiệp (*Writer*). Việc phân tách giúp tránh hiện tượng bão hòa ngữ cảnh (*context saturation*) và giảm thiểu tối đa hiện tượng tự huyễn hoặc (*hallucination*) khi một LLM đơn lẻ phải ôm đồm tất cả các vai trò.
2. **Yêu cầu kiểm tra chéo và bảo toàn nguồn gốc nghiêm ngặt (Strict Provenance & Verification)**:
   - *Ví dụ*: Đánh giá rủi ro tài chính, audit bảo mật hệ thống.
   - *Vì sao*: Cần một Agent độc lập (*Critic/Verifier*) đóng vai trò kiểm toán, so sánh từng luận điểm với bằng chứng thực tế trước khi xuất bản kết quả cuối cùng.
3. **Các tiểu tác vụ có thể phân rã và chạy song song (Parallelizable Subtasks)**:
   - *Ví dụ*: Đồng thời phân tích 5 khía cạnh độc lập của một bài toán lớn (Bảo mật, Chi phí, Hiệu năng, Tính mở rộng, Pháp lý) qua mô hình Map-Reduce.

---

### 2. Case nào KHÔNG NÊN dùng multi-agent? Vì sao?

**Không nên dùng trong các trường hợp:**
1. **Tác vụ đơn giản, đường thẳng (Simple / Single-Turn Tasks)**:
   - *Ví dụ*: Trả lời câu hỏi FAQ thường gặp, tóm tắt đoạn văn ngắn, sửa lỗi ngữ pháp, chuyển đổi định dạng JSON.
   - *Vì sao*: **Chi phí điều phối (Coordination Overhead)** vượt xa giá trị mang lại. Single-agent baseline hoàn thành trong ~1-3s với chi phí token cực thấp, trong khi multi-agent gây lãng phí token định tuyến và tăng độ trễ không cần thiết.
2. **Hệ thống yêu cầu phản hồi thời gian thực với độ trễ cực thấp (Ultra Low-Latency / Real-Time SLA)**:
   - *Ví dụ*: Chatbot chăm sóc khách hàng tương tác trực tiếp, Voice AI Agents giao tiếp giọng nói (cần độ trễ < 1-2s).
   - *Vì sao*: Luồng multi-agent tuần tự qua nhiều bước LLM khiến độ trễ tích lũy tăng lên 20-45s, gây trải nghiệm gián đoạn và khó chịu cho người dùng cuối.
3. **Ngân sách tài nguyên và token bị giới hạn nghiêm ngặt (Tight Token Budget)**:
   - *Ví dụ*: Ứng dụng xử lý hàng triệu requests/ngày với kinh phí hạ tầng thấp.
   - *Vì sao*: Multi-agent tiêu hao lượng token gấp 3-5 lần do phải truyền lại toàn bộ State/Notes giữa các bước chuyển giao (handoffs).


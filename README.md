# Medicalink AI Service

Worker Python xử lý **gợi ý bác sĩ (RAG)**: nhận RPC qua **RabbitMQ**, đồng bộ vector **Qdrant**, embedding **OpenAI**, bước sinh câu trả lời **OpenAI Chat** hoặc **Google Gemini**. Đồng bộ batch ban đầu qua HTTP public API của API Gateway.

## Kiến trúc ngắn

- **Ingress RPC**: hàng đợi `ai_service_queue` (pattern `ai.doctor-recommendation.request`), trả lời qua `reply_to` (NestJS microservices).
- **Sự kiện**: `doctor.profile.created` / `updated` / `deleted` trên exchange topic `medicalink.topic` → cập nhật/xóa vector.
- **Vector store**: Qdrant (tùy chọn **hybrid** dense OpenAI + sparse BM25/FastEmbed, RRF + rerank).
- **Batch**: script kéo toàn bộ `GET /api/doctors/profile/public` rồi `upsert` vào Qdrant (không cần RabbitMQ).

## Yêu cầu

- Python **3.11+**
- Tài khoản **OpenAI** (API key — **bắt buộc** cho embedding).
- **Qdrant** (local hoặc [Qdrant Cloud](https://cloud.qdrant.io/): URL + API key).
- **RabbitMQ** cùng hệ với `medicalink-microservice` (khi chạy worker).
- Tuỳ chọn **Gemini**: khi `LLM_PROVIDER=gemini`, cần `GOOGLE_GENAI_API_KEY` (hoặc `GEMINI_API_KEY`).

## Cài đặt (máy dev)

```powershell
cd medicalink-ai-service
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install . --no-deps
```

Đóng gói bánh xe cài trên server (tuỳ chọn; cần `pip install build`):

```powershell
python -m build
pip install ( Resolve-Path .\dist\medicalink_ai-*.whl )
```

Trên Linux/macOS: `pip install dist/medicalink_ai-*.whl` (hoặc chỉ định đúng tên file trong `dist/`).

Sao chép biến môi trường:

```powershell
copy .env.example .env
# Chỉnh .env — đặc biệt OPENAI_API_KEY, QDRANT_*, RABBITMQ_URL
```

## Biến môi trường chính

| Biến | Ý nghĩa | Mặc định / ghi chú |
|------|---------|-------------------|
| `RABBITMQ_URL` | AMQP | `amqp://...` |
| `OPENAI_API_KEY` | Embedding (và chat nếu dùng OpenAI LLM) | Bắt buộc |
| `QDRANT_URL` | HTTP(S) Qdrant | Local hoặc Cloud |
| `QDRANT_API_KEY` | Qdrant Cloud | Để trống nếu local không bật key |
| `QDRANT_COLLECTION_NAME` | Tên collection | `doctor_profiles` |
| `LLM_PROVIDER` | `openai` hoặc `gemini` | `openai` |
| `GOOGLE_GENAI_API_KEY` | Gemini (alias `GEMINI_API_KEY`) | Bắt buộc nếu `LLM_PROVIDER=gemini` |
| `GOOGLE_GENAI_MODEL` | Model Gemini (alias `GEMINI_MODEL`) | vd. `gemini-2.0-flash` |
| `GOOGLE_GENAI_TIMEOUT` | Timeout ms (alias `GEMINI_TIMEOUT_MS`) | `120000` |
| `API_GATEWAY_BASE_URL` | Cho batch sync | vd. `http://localhost:3000` |
| `RAG_HYBRID_ENABLED` | Hybrid dense + sparse | `true` |
| `RAG_RERANK_MODE` | `none` / `lexical` / `flashrank` | `flashrank` |
| `RAG_EVAL_LOG_PATH` | File JSONL đánh giá (để trống = tắt) | vd. `./data/rag_eval.jsonl` |

Chi tiết thêm xem `.env.example`. File **`.env.docker.example`** dùng khi worker chạy trong Docker **cùng mạng** với Compose của backend.

## Hợp nhất với medicalink-microservice (Qdrant local, không cần Cloud)

Trong repo **medicalink-microservice**, `development/docker-compose.yml` đã gồm **Postgres, Redis, RabbitMQ và Qdrant**: dữ liệu vector lưu trong volume `medicalink_qdrant`, truy cập nội bộ rất nhanh. Mạng Docker dùng tên cố định **`medicalink-network`** để repo **medicalink-ai-service** có thể `join` từ compose riêng.

### Trình tự chạy demo end-to-end

1. **Hạ tầng Docker (từ thư mục gốc microservice):**

   ```bash
   pnpm dev-docker:up
   ```

   Kiểm tra: RabbitMQ UI `http://localhost:15672`, Qdrant UI `http://localhost:6333/dashboard`.

2. **API Gateway + service provider (trên máy host):**  
   Cần có endpoint public doctors cho batch sync, và gateway nhận RPC tới AI.

   ```bash
   pnpm start:gateway
   pnpm start:provider
   ```

   (hoặc các service tương đương trong monorepo của bạn.)

3. **Cấu hình worker AI:** trong repo **medicalink-ai-service**:

   ```powershell
   copy .env.docker.example .env
   ```

   Chỉnh ít nhất `OPENAI_API_KEY`. Giữ `RABBITMQ_URL=...@rabbitmq:5672/` và `QDRANT_URL=http://qdrant:6333`.  
   **`API_GATEWAY_BASE_URL`:** gateway chạy trên host → `http://host.docker.internal:3000` (Docker Desktop Windows/macOS). Linux: có thể cần `http://172.17.0.1:3000` hoặc thêm `extra_hosts` — xem comment trong `.env.docker.example`.

4. **Nạp vector lần đầu (batch):** vẫn từ máy **host** (`.env` dùng `localhost`) *hoặc* chạy job một lần qua compose tích hợp:

   ```powershell
   docker compose -f compose.integrated.yaml run --rm medicalink-ai medicalink-ai-sync
   ```

5. **Chạy worker tích hợp:**

   ```powershell
   docker compose -f compose.integrated.yaml up -d --build
   ```

6. **Kiểm tra:** gọi API recommend bác sĩ từ frontend/gateway như luồng production; xem log container `medicalink-ai`.

**Lưu ý:** Nếu bạn đã từng chạy compose cũ, Docker có thể đã tạo mạng tên khác. Sau khi cập nhật `docker-compose.yml` (thêm `name: medicalink-network`), hãy `pnpm dev-docker:down` rồi `pnpm dev-docker:up` một lần để tạo đúng mạng `medicalink-network`.

| Biến (worker trong Docker) | Giá trị điển hình |
|-----------------------------|-------------------|
| `RABBITMQ_URL` | `amqp://admin:admin123@rabbitmq:5672/` |
| `QDRANT_URL` | `http://qdrant:6333` |
| `QDRANT_API_KEY` | để trống (Qdrant local) |
| `API_GATEWAY_BASE_URL` | `http://host.docker.internal:3000` (gateway trên host) |

Khi full stack chạy trong Docker deployment và service gateway tên **`api-gateway`**: dùng `http://api-gateway:3000` (không dùng `localhost` bên trong container).

## Chạy worker (RPC + consumer sự kiện)

Đảm bảo RabbitMQ, Qdrant và (tuỳ chọn) API Gateway đã sẵn sàng theo cấu hình trong `.env`.

```powershell
medicalink-ai-worker
# hoặc
python -m medicalink_ai.worker
```

Worker sẽ:

1. Kết nối Qdrant (có `api_key` nếu set).
2. Khai báo RPC queue + bind exchange topic cho sự kiện bác sĩ.
3. `ensure_collection` trên Qdrant.

## Đồng bộ dữ liệu ban đầu (batch)

Cần **OPENAI**, **QDRANT**, và gateway trả public doctors. **Không** cần RabbitMQ.

```powershell
medicalink-ai-sync
# hoặc
python -m medicalink_ai.scripts.batch_sync
```

Sau khi đổi schema hybrid / collection, có thể cần collection mới hoặc ingest lại (xem log worker về hybrid vs legacy).

## Docker

### Build

```powershell
docker build -t medicalink-ai:latest .
```

### Chạy worker

Truyền biến qua `--env-file` hoặc `-e` (không commit file `.env` chứa secret).

```powershell
docker run --rm -it --env-file .env medicalink-ai:latest
```

**Cache model (FlashRank / FastEmbed):** lần chạy đầu có thể tải model; production nên mount volume:

```powershell
docker run --rm -it --env-file .env -v medicalink-ai-cache:/app/.cache medicalink-ai:latest
```

**Ghi log RAG eval:** mount thư mục `data` nếu `RAG_EVAL_LOG_PATH=./data/rag_eval.jsonl`:

```powershell
docker run --rm -it --env-file .env -v ${PWD}/data:/app/data medicalink-ai:latest
```

### Chạy batch sync một lần (job)

```powershell
docker run --rm -it --env-file .env medicalink-ai:latest medicalink-ai-sync
```

Worker mặc định là `medicalink-ai-worker` (xem `CMD` trong `Dockerfile`).

### Docker Compose

- **`compose.yaml`**: worker độc lập; `.env` trỏ tới RabbitMQ/Qdrant/cloud (vd. `localhost` từ host).
- **`compose.integrated.yaml`**: worker join mạng **`medicalink-network`** của backend (đọc `.env.docker.example`).

```powershell
docker compose up -d --build
```

Đồng bộ batch một lần:

```powershell
docker compose run --rm medicalink-ai medicalink-ai-sync
```

Compose tích hợp:

```powershell
docker compose -f compose.integrated.yaml run --rm medicalink-ai medicalink-ai-sync
docker compose -f compose.integrated.yaml up -d --build
```

## Deploy (gợi ý)

**Full stack (medicalink-microservice + worker AI trên server thật / Docker):** xem hướng dẫn chi tiết tiếng Việt trong repo backend tại  
`medicalink-microservice/deployment/DEPLOYMENT_PRODUCTION_VI.md` (nếu clone hai repo cạnh nhau: `../medicalink-microservice/deployment/DEPLOYMENT_PRODUCTION_VI.md`).

**CI/CD:** GitHub Actions — `.github/workflows/ci.yml` (kiểm tra + build image thử), `.github/workflows/cd-docker.yml` (đẩy image lên GHCR). Tổng quan hai repo: `medicalink-microservice/deployment/GITHUB_CI_CD.md`.

### 1. Máy chủ / VPS (systemd)

1. Cài Docker hoặc Python 3.11+.
2. Clone repo, tạo `.env` production (secret từ vault/CI, không lưu trong git).
3. **Docker:** `docker run` như trên, hoặc dùng `docker compose` (tự thêm file `compose` nếu cần cùng mạng với RabbitMQ).
4. **systemd** (ví dụ): unit `ExecStart=/usr/bin/docker run ...` hoặc `ExecStart=/path/to/.venv/bin/medicalink-ai-worker`, `Restart=always`.

### 2. Nền tảng container (ECS, Cloud Run, Fly.io, Railway, …)

- **Một process:** chỉ chạy worker (`medicalink-ai-worker`).
- **Secrets:** map `OPENAI_API_KEY`, `QDRANT_API_KEY`, `GOOGLE_GENAI_API_KEY`, `RABBITMQ_URL` vào biến môi trường của platform.
- **Mạng:** container phải tới được RabbitMQ, Qdrant URL public (Cloud) và (khi batch) API Gateway.
- **CPU/RAM:** rerank + FastEmbed nhẹ hơn LLM; tăng RAM nếu tải model lớn hoặc nhiều concurrent (RPC `prefetch_count=1` mặc định trong code).

### 3. Kiểm tra sau deploy

- Gọi RPC từ gateway (endpoint recommend) hoặc publish message thủ công lên queue.
- Xem log worker: lỗi thường gặp — thiếu `OPENAI_API_KEY`, thiếu Gemini key khi `LLM_PROVIDER=gemini`, Qdrant sai URL/key, RabbitMQ không reachable.

## Cấu trúc gói

- `medicalink_ai/worker.py` — entry worker.
- `medicalink_ai/rag.py` — pipeline RAG + LLM.
- `medicalink_ai/vector_store.py` — Qdrant hybrid / legacy.
- `medicalink_ai/scripts/batch_sync.py` — đồng bộ HTTP → Qdrant.

Phiên bản package: `medicalink_ai.__version__` / `pyproject.toml`.

## Giấy phép

Nội bộ / theo quy định dự án Medicalink.

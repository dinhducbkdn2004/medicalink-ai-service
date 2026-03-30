"""
Định nghĩa dữ liệu lưu trong Qdrant cho RAG gợi ý bác sĩ.

Giai đoạn 1 (MVP)
-----------------
- Một điểm (point) trên mỗi doctor profile id.
- **Vector dense**: embedding câu văn ghép từ hồ sơ (OpenAI + LangChain).
- **Payload**: metadata để filter (is_active), audit và hiển thị; không dùng OpenSearch.

Giai đoạn 2 (Hybrid trên Qdrant, không thêm DB)
---------------------------------------------
- Bổ sung **sparse vector** trong cùng collection (ví dụ FastEmbed sparse / SPLADE tương thích Qdrant).
- Query: fusion dense + sparse để khớp tên riêng / thuật ngữ y khoa — vẫn một cluster Qdrant.
"""

from __future__ import annotations

from typing import TypedDict


class DoctorQdrantPayload(TypedDict, total=False):
    """
    Các trường payload khuyến nghị (đồng bộ với build_doctor_document).

    source_json: snapshot JSON ngắn để debug; không thay cho PostgreSQL.
    """

    doctor_id: str
    staff_account_id: str
    full_name: str
    is_active: bool
    specialty_ids: list[str]
    specialties_label: str
    location_ids: list[str]
    source_json: str

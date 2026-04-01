from __future__ import annotations

import json
import unicodedata
import uuid
from typing import Any


def doctor_point_id(doctor_profile_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medicalink:doctor:{doctor_profile_id}"))


def _join_lines(label: str, items: list[str] | None) -> str:
    if not items:
        return ""
    lines = [str(x).strip() for x in items if str(x).strip()]
    if not lines:
        return ""
    return f"{label}:\n" + "\n".join(f"- {x}" for x in lines)


def _specialties_text(specialties: Any) -> str:
    if not specialties:
        return ""
    names: list[str] = []
    for s in specialties:
        if isinstance(s, dict):
            n = s.get("name") or s.get("title")
            if n:
                names.append(str(n))
        elif isinstance(s, str):
            names.append(s)
    return ", ".join(names) if names else ""


def _locations_text(locations: Any) -> str:
    if not locations:
        return ""
    parts: list[str] = []
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        name = loc.get("name") or loc.get("address")
        if name:
            parts.append(str(name))
    return ", ".join(parts) if parts else ""


def _fold_ascii_lower(s: str) -> str:
    t = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _compute_seniority_score(profile: dict[str, Any]) -> float:
    """
    Điểm 0..1 heuristic từ chức danh / học hàm / độ dài kinh nghiệm (không cần cột năm riêng ở DB).
    Dùng boost khi user hỏi kiểu 'giỏi', 'lâu năm', 'giàu kinh nghiệm'.
    """
    score = 0.0
    position = profile.get("position") or []
    if isinstance(position, str):
        position = [position]
    degree = str(profile.get("degree") or "")
    intro = str(profile.get("introduction") or "")
    exp = profile.get("experience") or []
    if isinstance(exp, str):
        exp = [exp] if exp.strip() else []
    n_exp = len([x for x in exp if str(x).strip()]) if isinstance(exp, list) else 0

    blob = _fold_ascii_lower(
        " ".join(str(p) for p in position if p)
        + " "
        + degree
        + " "
        + intro
    )
    if any(k in blob for k in ("truong khoa", "pho truong khoa")):
        score += 0.35
    if any(
        k in blob
        for k in (
            "giao su",
            "pho giao su",
            " gs ",
            " gs.",
            " gs,",
            " pgs ",
            " pgs.",
        )
    ):
        score += 0.28
    if any(
        k in blob
        for k in (
            "tien si",
            " ths ",
            " ths.",
            "thac si",
            "chuyen khoa ii",
            "bs ckii",
            "bsckii",
        )
    ):
        score += 0.12
    score += min(0.35, n_exp * 0.065)
    return min(1.0, score)


def build_doctor_document(profile: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Trả về (text để embed, payload metadata cho Qdrant).
    profile: DoctorProfileResponseDto-like hoặc public list item từ API.
    """
    doctor_id = str(profile.get("id") or "")
    full_name = str(profile.get("fullName") or "").strip()
    degree = str(profile.get("degree") or "").strip()
    position = profile.get("position") or []
    if isinstance(position, str):
        position = [position]
    intro = str(profile.get("introduction") or "").strip()
    research = str(profile.get("research") or "").strip()

    specs = _specialties_text(profile.get("specialties"))
    locs = _locations_text(profile.get("workLocations"))

    text_parts = [
        f"Bác sĩ: {full_name}" if full_name else "Bác sĩ: (chưa rõ tên)",
        f"Mã hồ sơ (doctor profile id): {doctor_id}" if doctor_id else None,
        f"Học vị / bằng cấp: {degree}" if degree else None,
        _join_lines("Chức danh / vai trò", position if isinstance(position, list) else []),
        f"Chuyên khoa: {specs}" if specs else None,
        f"Địa điểm làm việc: {locs}" if locs else None,
        _join_lines("Quá trình đào tạo", profile.get("trainingProcess")),
        _join_lines("Kinh nghiệm", profile.get("experience")),
        f"Giới thiệu: {intro}" if intro else None,
        f"Nghiên cứu: {research}" if research else None,
        _join_lines("Hội viên", profile.get("memberships")),
        _join_lines("Giải thưởng", profile.get("awards")),
    ]
    text = "\n".join(p for p in text_parts if p)

    specialty_ids: list[str] = []
    for s in profile.get("specialties") or []:
        if isinstance(s, dict) and s.get("id") is not None:
            specialty_ids.append(str(s["id"]))

    location_ids: list[str] = []
    for loc in profile.get("workLocations") or []:
        if isinstance(loc, dict) and loc.get("id") is not None:
            location_ids.append(str(loc["id"]))

    seniority = _compute_seniority_score(profile)

    payload: dict[str, Any] = {
        "doctor_id": doctor_id,
        "staff_account_id": str(profile.get("staffAccountId") or ""),
        "full_name": full_name,
        "is_active": bool(profile.get("isActive", True)),
        "specialty_ids": specialty_ids,
        "specialties_label": specs,
        "locations_label": locs,
        "location_ids": location_ids,
        "seniority_score": seniority,
        "source_json": json.dumps(profile, ensure_ascii=False, default=str)[:8000],
    }
    return text, payload


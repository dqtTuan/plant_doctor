import pytest
from plant_doctor.rag.query_expansion import parse_label, label_to_query, yolo_detections_to_queries


def test_parse_standard_plantvillage_label():
    info = parse_label("Tomato___Early_blight")
    assert info["crop_en"].lower() == "tomato"
    assert "cà chua" in info["crop_vi"].lower()
    assert "early blight" in info["disease_en"].lower()
    assert "đốm vòng" in info["disease_vi"].lower()
    assert info["is_healthy"] is False
    assert "Tomato" in info["display_name"]


def test_parse_healthy_label():
    info = parse_label("Apple___healthy")
    assert info["is_healthy"] is True
    assert "táo" in info["crop_vi"].lower()
    assert "khỏe mạnh" in info["disease_vi"].lower()


def test_parse_custom_or_unknown_label():
    info = parse_label("Dragonfruit___Stem_canker")
    assert info["crop_en"] == "Dragonfruit"
    assert info["disease_en"] == "Stem canker"
    assert info["is_healthy"] is False


def test_label_to_query_generates_rich_text():
    q = label_to_query("Tomato___Early_blight")
    assert "cà chua" in q.lower() or "tomato" in q.lower()
    assert "early blight" in q.lower() or "đốm vòng" in q.lower()
    assert "triệu chứng" in q.lower() or "điều trị" in q.lower()


def test_label_to_query_with_aspect():
    q_symptom = label_to_query("Tomato___Late_blight", aspect="symptoms")
    assert "triệu chứng" in q_symptom.lower()

    q_treatment = label_to_query("Tomato___Late_blight", aspect="treatment")
    assert "điều trị" in q_treatment.lower() or "phòng trừ" in q_treatment.lower()


def test_yolo_detections_to_queries():
    dets = [("Tomato___Early_blight", 0.92), ("Potato___Late_blight", 0.85)]
    queries = yolo_detections_to_queries(dets)
    assert len(queries) == 2
    assert "tomato" in queries[0].lower() or "cà chua" in queries[0].lower()
    assert "potato" in queries[1].lower() or "khoai tây" in queries[1].lower()

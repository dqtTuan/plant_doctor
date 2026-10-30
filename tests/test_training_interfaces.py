"""Kiểm tra các pure function và interface của các module training."""
from plant_doctor.training.train_embedding import build_pairs, queries_for, train_custom_embedding
from plant_doctor.training.train_llm import build_samples, train_custom_llm
from plant_doctor.training.train_yolo import train_yolo


def test_training_pure_functions_exist():
    assert callable(train_yolo)
    assert callable(train_custom_embedding)
    assert callable(train_custom_llm)
    assert callable(build_pairs)
    assert callable(build_samples)
    assert callable(queries_for)

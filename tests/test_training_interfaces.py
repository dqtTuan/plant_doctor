"""Kiểm tra các pure function và interface của các module training."""
import yaml

from plant_doctor.training.train_embedding import build_pairs, queries_for, train_custom_embedding
from plant_doctor.training.train_llm import build_samples, train_custom_llm
from plant_doctor.training.train_yolo import normalize_data_yaml, train_yolo


def test_training_pure_functions_exist():
    assert callable(train_yolo)
    assert callable(train_custom_embedding)
    assert callable(train_custom_llm)
    assert callable(build_pairs)
    assert callable(build_samples)
    assert callable(queries_for)
    assert callable(normalize_data_yaml)


def test_normalize_data_yaml_fixes_roboflow_paths(tmp_path):
    yaml_file = tmp_path / "data.yaml"
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "valid" / "images").mkdir(parents=True)
    (tmp_path / "test" / "images").mkdir(parents=True)

    content = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 2,
        "names": ["class_a", "class_b"],
    }
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f)

    res = normalize_data_yaml(yaml_file)
    assert res == yaml_file

    with open(yaml_file, "r", encoding="utf-8") as f:
        fixed = yaml.safe_load(f)

    assert fixed["train"] == "train/images"
    assert fixed["val"] == "valid/images"
    assert fixed["test"] == "test/images"
    assert "path" in fixed
    assert fixed["path"] == str(tmp_path.resolve()).replace("\\", "/")

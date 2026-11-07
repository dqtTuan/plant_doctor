"""Cấu hình: đường dẫn/hằng số (`settings`) và nạp siêu tham số từ YAML (`loader`)."""
from plant_doctor.config.loader import apply_overrides, load_config  # noqa: F401
from plant_doctor.config.settings import *  # noqa: F401,F403
from plant_doctor.config.settings import embed_max_tokens  # noqa: F401


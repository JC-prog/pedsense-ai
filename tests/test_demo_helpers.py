import json

from pedsense.demo import (
    _find_yolo_weights,
    _list_available_models,
    _get_latest_model,
    _detect_model_type,
)


# --- _find_yolo_weights ---


class TestFindYoloWeights:
    def test_returns_best_pt_when_present(self, tmp_path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").touch()
        (weights_dir / "last.pt").touch()

        assert _find_yolo_weights(tmp_path) == weights_dir / "best.pt"

    def test_falls_back_to_last_pt(self, tmp_path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "last.pt").touch()

        assert _find_yolo_weights(tmp_path) == weights_dir / "last.pt"

    def test_falls_back_to_any_pt(self, tmp_path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "custom_model.pt").touch()

        assert _find_yolo_weights(tmp_path) == weights_dir / "custom_model.pt"

    def test_returns_none_when_empty_weights_dir(self, tmp_path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()

        assert _find_yolo_weights(tmp_path) is None

    def test_returns_none_when_no_weights_dir(self, tmp_path):
        assert _find_yolo_weights(tmp_path) is None


# --- _list_available_models ---


class TestListAvailableModels:
    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pedsense.demo.CUSTOM_MODELS_DIR", tmp_path / "nonexistent")

        assert _list_available_models() == []

    def test_finds_model_with_config_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pedsense.demo.CUSTOM_MODELS_DIR", tmp_path)
        model_dir = tmp_path / "resnet_20260215"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "resnet-lstm"}')

        result = _list_available_models()
        assert result == ["resnet_20260215"]

    def test_finds_yolo_model_with_weights(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pedsense.demo.CUSTOM_MODELS_DIR", tmp_path)
        model_dir = tmp_path / "yolo_20260214"
        weights_dir = model_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "last.pt").touch()

        result = _list_available_models()
        assert result == ["yolo_20260214"]

    def test_skips_dir_without_config_or_weights(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pedsense.demo.CUSTOM_MODELS_DIR", tmp_path)
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        assert _list_available_models() == []

    def test_sorted_reverse_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pedsense.demo.CUSTOM_MODELS_DIR", tmp_path)

        for name in ["model_a", "model_c", "model_b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "config.json").write_text("{}")

        result = _list_available_models()
        assert result == ["model_c", "model_b", "model_a"]


# --- _get_latest_model ---


class TestGetLatestModel:
    def test_returns_none_when_no_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pedsense.demo.CUSTOM_MODELS_DIR", tmp_path / "nonexistent")

        assert _get_latest_model() is None

    def test_returns_first_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pedsense.demo.CUSTOM_MODELS_DIR", tmp_path)
        for name in ["model_a", "model_b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "config.json").write_text("{}")

        assert _get_latest_model() == "model_b"


# --- _detect_model_type ---


class TestDetectModelType:
    def test_reads_type_from_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "hybrid"}))

        assert _detect_model_type(tmp_path) == "hybrid"

    def test_defaults_to_yolo_when_config_has_no_type(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")

        assert _detect_model_type(tmp_path) == "yolo"

    def test_detects_yolo_from_weights_dir(self, tmp_path):
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").touch()

        assert _detect_model_type(tmp_path) == "yolo"

    def test_detects_resnet_lstm_from_best_pt_at_root(self, tmp_path):
        (tmp_path / "best.pt").touch()

        assert _detect_model_type(tmp_path) == "resnet-lstm"

    def test_defaults_to_yolo_when_nothing_matches(self, tmp_path):
        assert _detect_model_type(tmp_path) == "yolo"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Any
    best_threshold: float
    meta: Dict[str, Any]


class PurchaseIntentModelAdapter:
    """
    - artifacts/*.joblib 로딩 (pipeline + threshold)
    - 서비스에서 predict/predict_proba 호출할 수 있게 제공
    """

    def __init__(self, model_path: str | Path):
        self._model_path = Path(model_path)
        self._artifact: Optional[ModelArtifact] = None

    def load(self) -> ModelArtifact:
        if self._artifact is not None:
            return self._artifact

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self._model_path}")

        raw = joblib.load(self._model_path)
        
        pipeline = None
        best_threshold = 0.5
        meta = {}

        # Case 1: The loaded object is the pipeline itself (legacy support)
        if hasattr(raw, "predict_proba"):
            pipeline = raw
            # Try to infer meta if possible, otherwise empty
        
        # Case 2: It is a dictionary
        elif isinstance(raw, dict):
            # Try to find pipeline
            if "pipeline" in raw:
                pipeline = raw["pipeline"]
            else:
                # Search for any value that looks like a model
                for k, v in raw.items():
                    if hasattr(v, "predict_proba"):
                        pipeline = v
                        break
            
            # Try to find threshold
            if "best_threshold" in raw:
                best_threshold = float(raw["best_threshold"])
            
            # Store everything else as meta
            meta = {k: v for k, v in raw.items() if k not in ("pipeline", "best_threshold")}

        if pipeline is None:
            raise ValueError(
                f"Invalid artifact format in {self._model_path}. "
                "Could not find a valid pipeline model with 'predict_proba'."
            )

        self._artifact = ModelArtifact(
            pipeline=pipeline,
            best_threshold=best_threshold,
            meta=meta,
        )
        return self._artifact

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        art = self.load()
        proba = art.pipeline.predict_proba(features)[:, 1]
        return pd.Series(proba, index=features.index, name="purchase_proba")

    def predict(self, features: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        art = self.load()
        thr = art.best_threshold if threshold is None else float(threshold)
        proba = self.predict_proba(features)
        pred = (proba >= thr).astype(int)
        return pd.Series(pred.values, index=features.index, name="purchase_pred")

    def get_threshold(self) -> float:
        return float(self.load().best_threshold)

    def get_training_data(self) -> pd.DataFrame:
        """
        학습에 사용된 원본 데이터를 로드하여 반환합니다.
        (EDA 및 시각화용)
        
        PurchaseIntentModelAdapterConfig와 유사한 Robust Path Logic 사용
        """
        # 현재 파일: app/adapters/PurchaseIntentModelAdapter.py
        adapter_dir = Path(__file__).resolve().parent
        app_dir = adapter_dir.parent
        
        candidates = [
            app_dir.parent,      # Standard ROOT
            app_dir,             # app is root?
            Path.cwd(),          # CWD
        ]
        
        root_dir = candidates[0]
        for candidate in candidates:
            if (candidate / "data").exists():
                root_dir = candidate
                break
        
        # 우선순위: app/artifacts/train.csv (Self-contained) -> data/processed/train.csv (Original)
        # Plan 500: We copied data to app/artifacts/train.csv
        artifact_data = app_dir / "artifacts" / "train.csv"
        if artifact_data.exists():
            data_path = artifact_data
        else:
            data_path = root_dir / "data" / "processed" / "train.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at: {data_path}")

        df = pd.read_csv(data_path)
        # Model was trained with row_id included (implicit feature), so we must NOT drop it
        # if "row_id" in df.columns:
        #     df = df.drop(columns=["row_id"])
        
        return df

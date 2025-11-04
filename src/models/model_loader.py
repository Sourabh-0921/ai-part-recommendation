"""
Model loading utilities for LightGBM and MLflow.

Returns loaded model object and version metadata.
"""

from typing import Any, Dict, Tuple
import os
import logging

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Tuple[Any, Dict[str, str]]:
    """Load a model from path (LightGBM, joblib/pickle, or MLflow URI).
    
    Args:
        model_path: File path or MLflow URI (e.g., 'models:/name/Production')
    
    Returns:
        (model_object, metadata_dict)
    """
    if not model_path:
        raise ValueError("model_path is required")

    # MLflow model URI
    if model_path.startswith('models:') or model_path.startswith('runs:'):
        try:
            import mlflow.pyfunc  # type: ignore
            model = mlflow.pyfunc.load_model(model_path)
            meta = {
                'framework': 'mlflow.pyfunc',
                'model_uri': model_path,
                'model_version': getattr(model, 'metadata', None).get('model_uuid', '') if hasattr(model, 'metadata') else '',
            }
            logger.info(f"Loaded MLflow model from {model_path}")
            return model, meta
        except Exception as e:
            logger.error(f"Failed to load MLflow model: {e}")
            raise

    # LightGBM native model
    ext = os.path.splitext(model_path)[1].lower()
    if ext in ['.txt', '.lgb', '.lightgbm']:
        try:
            import lightgbm as lgb  # type: ignore
            booster = lgb.Booster(model_file=model_path)
            meta = {
                'framework': 'lightgbm',
                'model_uri': model_path,
                'model_version': booster.params.get('version', '') if hasattr(booster, 'params') else ''
            }
            logger.info(f"Loaded LightGBM model from {model_path}")
            return booster, meta
        except Exception as e:
            logger.error(f"Failed to load LightGBM model: {e}")
            raise

    # joblib/pickle
    if ext in ['.joblib', '.pkl', '.pickle']:
        try:
            if ext == '.joblib':
                import joblib  # type: ignore
                model = joblib.load(model_path)
            else:
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            meta = {
                'framework': 'pickle',
                'model_uri': model_path,
                'model_version': getattr(model, 'version', '')
            }
            logger.info(f"Loaded pickle/joblib model from {model_path}")
            return model, meta
        except Exception as e:
            logger.error(f"Failed to load pickle/joblib model: {e}")
            raise

    raise ValueError(f"Unsupported model format for path: {model_path}")



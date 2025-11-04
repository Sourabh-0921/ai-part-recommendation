"""
Model factory for creating and managing ML models.

This module implements the factory pattern for creating different types of
ML models and managing model lifecycle operations.
"""

from typing import Dict, List, Optional, Any, Type
import logging
from pathlib import Path
import os

from .base_model import BaseMLModel
from .lightgbm_model import LightGBMPartModel
from .ema_calculator import EMACalculator

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating and managing ML models.
    
    This factory provides a centralized way to create, load, and manage
    different types of ML models for the parts recommendation system.
    """
    
    # Registry of available model types
    MODEL_TYPES = {
        'lightgbm': LightGBMPartModel,
        'lightgbm_part': LightGBMPartModel,
    }
    
    def __init__(self, models_directory: str = "models"):
        """
        Initialize the model factory.
        
        Args:
            models_directory: Base directory for storing models
        """
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, BaseMLModel] = {}
        
        logger.info(f"Initialized ModelFactory with directory: {models_directory}")
    
    def create_model(
        self,
        model_type: str,
        part_code: str,
        model_version: str = "1.0.0",
        **kwargs
    ) -> BaseMLModel:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model to create
            part_code: Code of the part this model predicts
            model_version: Version of the model
            **kwargs: Additional model parameters
            
        Returns:
            Created model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {list(self.MODEL_TYPES.keys())}")
        
        model_class = self.MODEL_TYPES[model_type]
        model_key = f"{model_type}_{part_code}_{model_version}"
        
        try:
            model = model_class(
                part_code=part_code,
                model_version=model_version,
                **kwargs
            )
            
            self.loaded_models[model_key] = model
            logger.info(f"Created {model_type} model for part {part_code}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating {model_type} model for part {part_code}: {e}")
            raise
    
    def load_model(
        self,
        model_path: str,
        model_type: Optional[str] = None
    ) -> BaseMLModel:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            model_type: Type of model (optional, will be inferred if not provided)
            
        Returns:
            Loaded model instance
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Try to infer model type from path if not provided
            if model_type is None:
                model_type = self._infer_model_type(model_path)
            
            if model_type not in self.MODEL_TYPES:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Create model instance
            model_class = self.MODEL_TYPES[model_type]
            model = model_class.__new__(model_class)
            
            # Load the model
            model.load_model(model_path)
            
            # Generate model key
            model_key = f"{model_type}_{model.part_code}_{model.model_version}"
            self.loaded_models[model_key] = model
            
            logger.info(f"Loaded {model_type} model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def get_model(
        self,
        part_code: str,
        model_type: str = "lightgbm",
        model_version: str = "1.0.0"
    ) -> Optional[BaseMLModel]:
        """
        Get a loaded model by part code and type.
        
        Args:
            part_code: Code of the part
            model_type: Type of model
            model_version: Version of the model
            
        Returns:
            Model instance if found, None otherwise
        """
        model_key = f"{model_type}_{part_code}_{model_version}"
        return self.loaded_models.get(model_key)
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        List all currently loaded models.
        
        Returns:
            List of model information dictionaries
        """
        models_info = []
        
        for model_key, model in self.loaded_models.items():
            models_info.append({
                'model_key': model_key,
                'part_code': getattr(model, 'part_code', 'unknown'),
                'model_type': model.__class__.__name__,
                'model_version': model.model_version,
                'is_trained': model.is_trained,
                'created_at': model.created_at.isoformat()
            })
        
        return models_info
    
    def unload_model(
        self,
        part_code: str,
        model_type: str = "lightgbm",
        model_version: str = "1.0.0"
    ) -> bool:
        """
        Unload a model from memory.
        
        Args:
            part_code: Code of the part
            model_type: Type of model
            model_version: Version of the model
            
        Returns:
            True if model was unloaded, False if not found
        """
        model_key = f"{model_type}_{part_code}_{model_version}"
        
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            logger.info(f"Unloaded model {model_key}")
            return True
        
        return False
    
    def unload_all_models(self) -> int:
        """
        Unload all models from memory.
        
        Returns:
            Number of models unloaded
        """
        count = len(self.loaded_models)
        self.loaded_models.clear()
        logger.info(f"Unloaded {count} models")
        return count
    
    def save_model(
        self,
        model: BaseMLModel,
        model_path: Optional[str] = None
    ) -> str:
        """
        Save a model to disk.
        
        Args:
            model: Model instance to save
            model_path: Custom path to save the model (optional)
            
        Returns:
            Path where the model was saved
        """
        try:
            if model_path is None:
                model_path = self._generate_model_path(model)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            model.save_model(model_path)
            
            logger.info(f"Saved model to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models on disk.
        
        Returns:
            List of saved model information
        """
        saved_models = []
        
        try:
            for model_file in self.models_directory.rglob("*.pkl"):
                try:
                    # Try to load model metadata without loading the full model
                    import joblib
                    model_data = joblib.load(model_file)
                    
                    saved_models.append({
                        'file_path': str(model_file),
                        'model_name': model_data.get('model_name', 'unknown'),
                        'model_version': model_data.get('model_version', 'unknown'),
                        'part_code': model_data.get('part_code', 'unknown'),
                        'is_trained': model_data.get('is_trained', False),
                        'created_at': model_data.get('created_at', 'unknown'),
                        'saved_at': model_data.get('saved_at', 'unknown'),
                        'file_size': model_file.stat().st_size
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not read model metadata from {model_file}: {e}")
                    saved_models.append({
                        'file_path': str(model_file),
                        'error': str(e)
                    })
        
        except Exception as e:
            logger.error(f"Error listing saved models: {e}")
        
        return saved_models
    
    def delete_model(self, model_path: str) -> bool:
        """
        Delete a saved model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model was deleted, False if not found
        """
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted model {model_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting model {model_path}: {e}")
            return False
    
    def create_ema_calculator(self, **kwargs) -> EMACalculator:
        """
        Create an EMA calculator instance.
        
        Args:
            **kwargs: EMA calculator parameters
            
        Returns:
            EMA calculator instance
        """
        return EMACalculator(**kwargs)
    
    def _infer_model_type(self, model_path: str) -> str:
        """
        Infer model type from file path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Inferred model type
        """
        path_lower = model_path.lower()
        
        if 'lightgbm' in path_lower:
            return 'lightgbm'
        elif 'xgboost' in path_lower:
            return 'xgboost'
        elif 'random_forest' in path_lower:
            return 'random_forest'
        else:
            # Default to lightgbm
            return 'lightgbm'
    
    def _generate_model_path(self, model: BaseMLModel) -> str:
        """
        Generate a model path based on model properties.
        
        Args:
            model: Model instance
            
        Returns:
            Generated model path
        """
        part_code = getattr(model, 'part_code', 'unknown')
        model_type = model.__class__.__name__.lower()
        model_version = model.model_version
        
        filename = f"{model_type}_{part_code}_v{model_version}.pkl"
        return str(self.models_directory / filename)
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded models.
        
        Returns:
            Dictionary containing model statistics
        """
        if not self.loaded_models:
            return {
                'total_models': 0,
                'trained_models': 0,
                'model_types': {},
                'part_codes': []
            }
        
        trained_count = sum(1 for model in self.loaded_models.values() if model.is_trained)
        model_types = {}
        part_codes = set()
        
        for model in self.loaded_models.values():
            model_type = model.__class__.__name__
            model_types[model_type] = model_types.get(model_type, 0) + 1
            
            if hasattr(model, 'part_code'):
                part_codes.add(model.part_code)
        
        return {
            'total_models': len(self.loaded_models),
            'trained_models': trained_count,
            'untrained_models': len(self.loaded_models) - trained_count,
            'model_types': model_types,
            'part_codes': list(part_codes),
            'models_directory': str(self.models_directory)
        }

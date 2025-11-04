"""
Base model class for ML models in the AI Parts Recommendation System.

This module provides the foundation for all machine learning models,
including common functionality for training, prediction, and model management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseMLModel(ABC):
    """
    Abstract base class for all ML models.
    
    This class defines the interface that all ML models must implement,
    providing common functionality for training, prediction, and persistence.
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.is_trained = False
        self.training_metadata = {}
        self.feature_importance = None
        self.created_at = datetime.utcnow()
        
        logger.info(f"Initialized {model_name} model version {model_version}")
    
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of prediction probabilities
        """
        pass
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 80.0
    ) -> List[Dict[str, Any]]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Features for prediction
            confidence_threshold: Minimum confidence threshold (0-100)
            
        Returns:
            List of predictions with confidence scores
        """
        try:
            probabilities = self.predict_proba(X)
            predictions = []
            
            for i, prob in enumerate(probabilities):
                confidence_score = prob * 100
                
                if confidence_score >= confidence_threshold:
                    predictions.append({
                        'probability': float(prob),
                        'confidence_score': float(confidence_score),
                        'prediction': 1 if prob > 0.5 else 0,
                        'meets_threshold': True
                    })
                else:
                    predictions.append({
                        'probability': float(prob),
                        'confidence_score': float(confidence_score),
                        'prediction': 0,
                        'meets_threshold': False
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_with_confidence: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        return self.feature_importance
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'model_version': self.model_version,
                'is_trained': self.is_trained,
                'training_metadata': self.training_metadata,
                'feature_importance': self.feature_importance,
                'created_at': self.created_at,
                'saved_at': datetime.utcnow()
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.model_version = model_data['model_version']
            self.is_trained = model_data['is_trained']
            self.training_metadata = model_data['training_metadata']
            self.feature_importance = model_data['feature_importance']
            self.created_at = model_data['created_at']
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'feature_importance': self.feature_importance,
            'created_at': self.created_at.isoformat(),
            'model_type': self.__class__.__name__
        }
    
    def validate_features(self, X: pd.DataFrame, required_features: List[str]) -> None:
        """
        Validate that required features are present.
        
        Args:
            X: Feature DataFrame
            required_features: List of required feature names
            
        Raises:
            ValueError: If required features are missing
        """
        missing_features = set(required_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        
        Args:
            X: Raw feature DataFrame
            
        Returns:
            Preprocessed feature DataFrame
        """
        # Default implementation - can be overridden by subclasses
        return X.copy()
    
    def postprocess_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess predictions and probabilities.
        
        Args:
            predictions: Raw predictions
            probabilities: Raw probabilities
            
        Returns:
            Tuple of (processed_predictions, processed_probabilities)
        """
        # Default implementation - can be overridden by subclasses
        return predictions, probabilities

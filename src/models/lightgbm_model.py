"""
LightGBM model implementation for parts recommendation.

This module implements the primary ML model using LightGBM with native
categorical feature support as specified in the .cursorrules file.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from .base_model import BaseMLModel

logger = logging.getLogger(__name__)


class LightGBMPartModel(BaseMLModel):
    """
    LightGBM model for part recommendation.
    
    This model uses LightGBM with native categorical feature support
    to predict which parts need replacement for a given vehicle.
    """
    
    def __init__(
        self,
        part_code: str,
        model_version: str = "1.0.0",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LightGBM model.
        
        Args:
            part_code: Code of the part this model predicts
            model_version: Version of the model
            params: LightGBM parameters
        """
        super().__init__(f"lightgbm_{part_code}", model_version)
        self.part_code = part_code
        self.params = params or self._get_default_params()
        self.categorical_features = []
        self.feature_names = []
        
        logger.info(f"Initialized LightGBM model for part {part_code}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default LightGBM parameters optimized for parts recommendation.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True  # Optimize for categorical features
        }
    
    def _identify_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """
        Identify categorical features in the dataset.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of categorical feature names
        """
        categorical_features = []
        
        # Features that should always be categorical
        always_categorical = [
            'vehicle_model',
            'dealer_code', 
            'region_code',
            'terrain_type',
            'season_code',
            'part_category'
        ]
        
        for feature in always_categorical:
            if feature in X.columns:
                categorical_features.append(feature)
        
        # Identify other categorical features
        for col in X.columns:
            if col not in categorical_features:
                # Check if column has low cardinality and string/object type
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    unique_values = X[col].nunique()
                    if unique_values < 50:  # Low cardinality threshold
                        categorical_features.append(col)
        
        logger.info(f"Identified categorical features: {categorical_features}")
        return categorical_features
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels (1=needs replacement, 0=doesn't need)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            logger.info(f"Starting training for part {self.part_code}")
            
            # Identify categorical features
            self.categorical_features = self._identify_categorical_features(X_train)
            self.feature_names = list(X_train.columns)
            
            # Prepare training data
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                categorical_feature=self.categorical_features,
                free_raw_data=False
            )
            
            # Prepare validation data if provided
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val,
                    categorical_feature=self.categorical_features,
                    free_raw_data=False
                )
                valid_sets.append(val_data)
                valid_names.append('valid')
            
            # Training parameters
            num_boost_round = kwargs.get('num_boost_round', 500)
            early_stopping_rounds = kwargs.get('early_stopping_rounds', 50)
            verbose_eval = kwargs.get('verbose_eval', 100)
            
            # Train the model
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(period=verbose_eval)
                ]
            )
            
            # Get feature importance
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importance()
            ))
            
            # Calculate training metrics
            train_pred = self.predict(X_train)
            train_prob = self.predict_proba(X_train)
            
            training_metrics = {
                'train_accuracy': np.mean(train_pred == y_train),
                'train_auc': self._calculate_auc(y_train, train_prob),
                'best_iteration': self.model.best_iteration,
                'num_features': len(self.feature_names),
                'categorical_features': len(self.categorical_features)
            }
            
            # Calculate validation metrics if validation data provided
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_prob = self.predict_proba(X_val)
                
                training_metrics.update({
                    'val_accuracy': np.mean(val_pred == y_val),
                    'val_auc': self._calculate_auc(y_val, val_prob)
                })
            
            self.is_trained = True
            self.training_metadata = {
                'part_code': self.part_code,
                'training_date': datetime.utcnow().isoformat(),
                'num_training_samples': len(X_train),
                'num_validation_samples': len(X_val) if X_val is not None else 0,
                'metrics': training_metrics,
                'params': self.params
            }
            
            logger.info(f"Training completed for part {self.part_code}")
            logger.info(f"Training metrics: {training_metrics}")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Error training model for part {self.part_code}: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            probabilities = self.predict_proba(X)
            return (probabilities > 0.5).astype(int)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Ensure features are in the same order as training
            X_ordered = X[self.feature_names]
            
            # Make predictions
            probabilities = self.model.predict(X_ordered)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            raise
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 80.0
    ) -> List[Dict[str, Any]]:
        """
        Make predictions with confidence scores and detailed information.
        
        Args:
            X: Features for prediction
            confidence_threshold: Minimum confidence threshold (0-100)
            
        Returns:
            List of predictions with confidence scores and reasoning
        """
        try:
            probabilities = self.predict_proba(X)
            predictions = []
            
            for i, prob in enumerate(probabilities):
                confidence_score = prob * 100
                
                # Get feature contributions for reasoning
                feature_contributions = self._get_feature_contributions(X.iloc[i:i+1])
                
                prediction_data = {
                    'probability': float(prob),
                    'confidence_score': float(confidence_score),
                    'prediction': 1 if prob > 0.5 else 0,
                    'meets_threshold': confidence_score >= confidence_threshold,
                    'part_code': self.part_code,
                    'feature_contributions': feature_contributions,
                    'top_features': self._get_top_features(feature_contributions, top_k=5)
                }
                
                predictions.append(prediction_data)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_with_confidence: {e}")
            raise
    
    def _get_feature_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature contributions for a single prediction.
        
        Args:
            X: Single row DataFrame for prediction
            
        Returns:
            Dictionary of feature contributions
        """
        try:
            # Get SHAP-like feature contributions
            contributions = self.model.predict(
                X, 
                pred_contrib=True
            )
            
            # Remove the bias term (last element)
            feature_contributions = contributions[0][:-1]
            
            return dict(zip(self.feature_names, feature_contributions))
            
        except Exception as e:
            logger.warning(f"Could not get feature contributions: {e}")
            return {}
    
    def _get_top_features(self, contributions: Dict[str, float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top contributing features.
        
        Args:
            contributions: Feature contributions dictionary
            top_k: Number of top features to return
            
        Returns:
            List of top features with their contributions
        """
        if not contributions:
            return []
        
        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_features = []
        for feature, contribution in sorted_features[:top_k]:
            top_features.append({
                'feature': feature,
                'contribution': float(contribution),
                'importance': self.feature_importance.get(feature, 0.0)
            })
        
        return top_features
    
    def _calculate_auc(self, y_true: pd.Series, y_prob: np.ndarray) -> float:
        """
        Calculate AUC score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            AUC score
        """
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            return 0.0
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get detailed model summary.
        
        Returns:
            Dictionary containing model summary information
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'part_code': self.part_code,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'num_features': len(self.feature_names),
            'categorical_features': self.categorical_features,
            'feature_importance': self.feature_importance,
            'training_metadata': self.training_metadata,
            'best_iteration': self.model.best_iteration if self.model else None,
            'params': self.params
        }

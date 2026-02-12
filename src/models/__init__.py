"""Machine learning models and evaluation utilities.

This package contains:
- ML model wrappers (XGBoost classifier/regressor)
- Benchmark comparison tools
- Pace/PIR strategy evaluators
- Model performance metrics

Modules:
    ml_model: Main XGBoost model wrapper with training/prediction
    benchmark_cmp: Compare model performance against benchmarks
    pace_strategy: Pace-based betting strategy
    pir_evaluator: PIR (Position in Run) evaluation metrics
    benchmark_fast_updater: Fast benchmark updates for rolling stats

Example:
    >>> from src.models.ml_model import MLModel
    >>> model = MLModel(model_type='classifier')
    >>> model.train(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

__all__ = [
    'MLModel',
]

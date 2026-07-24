from .auth_routes import auth_bp
from .dashboard_routes import dashboard_bp
from .dataset_routes import dataset_bp
from .training_routes import legacy_training_bp, model_management_bp
from .evaluation_routes import evaluation_bp
from .classification_routes import classification_bp
from .report_routes import report_bp
from .algorithm_calculation_routes import algorithm_calculation_bp
from .cnn_calculation_routes import cnn_calculation_bp
from .db_admin_routes import db_admin_bp

__all__ = [
    "auth_bp",
    "dashboard_bp",
    "dataset_bp",
    "model_management_bp",
    "legacy_training_bp",
    "evaluation_bp",
    "classification_bp",
    "report_bp",
    "algorithm_calculation_bp",
    "cnn_calculation_bp",
    "db_admin_bp",
]

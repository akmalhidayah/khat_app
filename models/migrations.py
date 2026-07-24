"""Safe schema migrations for classification_results."""

from sqlalchemy import inspect, text

from models.database import db


def ensure_classification_result_columns() -> None:
    """Add OOD detection columns without breaking existing records."""
    table = "classification_results"
    inspector = inspect(db.engine)
    if table not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns(table)}
    alters = []

    if "input_status" not in existing:
        alters.append("ADD COLUMN input_status VARCHAR(50) DEFAULT 'khat'")
    if "is_khat" not in existing:
        alters.append("ADD COLUMN is_khat TINYINT(1) DEFAULT 1")
    if "khat_probability" not in existing:
        alters.append("ADD COLUMN khat_probability FLOAT NULL")
    if "non_khat_probability" not in existing:
        alters.append("ADD COLUMN non_khat_probability FLOAT NULL")
    if "rejection_reason" not in existing:
        alters.append("ADD COLUMN rejection_reason TEXT NULL")
    if "top2_margin" not in existing:
        alters.append("ADD COLUMN top2_margin FLOAT NULL")
    if "reliability_level" not in existing:
        alters.append("ADD COLUMN reliability_level VARCHAR(100) NULL")
    if "detection_status" not in existing:
        alters.append("ADD COLUMN detection_status VARCHAR(50) NULL")
    if "detection_decision" not in existing:
        alters.append("ADD COLUMN detection_decision VARCHAR(50) NULL")
    if "manual_review_required" not in existing:
        alters.append("ADD COLUMN manual_review_required TINYINT(1) DEFAULT 0")
    if "uploaded_filename" not in existing:
        alters.append("ADD COLUMN uploaded_filename VARCHAR(255) NULL")
    if "expected_class" not in existing:
        alters.append("ADD COLUMN expected_class VARCHAR(50) NULL")
    if "confidence_score" not in existing:
        alters.append("ADD COLUMN confidence_score FLOAT NULL")
    if "probability_diwani" not in existing:
        alters.append("ADD COLUMN probability_diwani FLOAT NULL")
    if "probability_diwani_jali" not in existing:
        alters.append("ADD COLUMN probability_diwani_jali FLOAT NULL")
    if "probability_naskhi" not in existing:
        alters.append("ADD COLUMN probability_naskhi FLOAT NULL")
    if "probability_tsuluts" not in existing:
        alters.append("ADD COLUMN probability_tsuluts FLOAT NULL")
    if "validation_status" not in existing:
        alters.append("ADD COLUMN validation_status VARCHAR(100) NULL")
    if "review_status" not in existing:
        alters.append("ADD COLUMN review_status VARCHAR(100) NULL")
    if "confidence_label" not in existing:
        alters.append("ADD COLUMN confidence_label VARCHAR(100) NULL")
    if "top_2_class" not in existing:
        alters.append("ADD COLUMN top_2_class VARCHAR(50) NULL")
    if "top_2_score" not in existing:
        alters.append("ADD COLUMN top_2_score FLOAT NULL")
    if "final_decision" not in existing:
        alters.append("ADD COLUMN final_decision VARCHAR(100) NULL")
    if "explanation_text" not in existing:
        alters.append("ADD COLUMN explanation_text TEXT NULL")
    if "model_name" not in existing:
        alters.append("ADD COLUMN model_name VARCHAR(100) NULL")
    if "model_source" not in existing:
        alters.append("ADD COLUMN model_source VARCHAR(100) NULL")
    if "model_runtime" not in existing:
        alters.append("ADD COLUMN model_runtime VARCHAR(100) NULL")
    if "model_input_size" not in existing:
        alters.append("ADD COLUMN model_input_size VARCHAR(20) NULL")
    if "similarity_status" not in existing:
        alters.append("ADD COLUMN similarity_status VARCHAR(100) NULL")
    if "similarity_score" not in existing:
        alters.append("ADD COLUMN similarity_score FLOAT NULL")
    if "nearest_dataset_image" not in existing:
        alters.append("ADD COLUMN nearest_dataset_image VARCHAR(255) NULL")
    if "nearest_dataset_class" not in existing:
        alters.append("ADD COLUMN nearest_dataset_class VARCHAR(50) NULL")
    if "similarity_risk_level" not in existing:
        alters.append("ADD COLUMN similarity_risk_level VARCHAR(100) NULL")
    if "similarity_message" not in existing:
        alters.append("ADD COLUMN similarity_message TEXT NULL")
    if "similarity_recommendation" not in existing:
        alters.append("ADD COLUMN similarity_recommendation TEXT NULL")
    if "manual_expected_class" not in existing:
        alters.append("ADD COLUMN manual_expected_class VARCHAR(50) NULL")
    if "correction_label" not in existing:
        alters.append("ADD COLUMN correction_label VARCHAR(50) NULL")
    if "correction_notes" not in existing:
        alters.append("ADD COLUMN correction_notes TEXT NULL")
    if "source_type" not in existing:
        alters.append("ADD COLUMN source_type VARCHAR(50) NULL")
    if "preprocessing_mode" not in existing:
        alters.append("ADD COLUMN preprocessing_mode VARCHAR(50) NULL")
    if "known_class_status" not in existing:
        alters.append("ADD COLUMN known_class_status VARCHAR(100) NULL")
    if "input_source" not in existing:
        alters.append("ADD COLUMN input_source VARCHAR(50) DEFAULT 'upload'")
    if "original_width" not in existing:
        alters.append("ADD COLUMN original_width INT NULL")
    if "original_height" not in existing:
        alters.append("ADD COLUMN original_height INT NULL")
    if "processed_width" not in existing:
        alters.append("ADD COLUMN processed_width INT NULL")
    if "processed_height" not in existing:
        alters.append("ADD COLUMN processed_height INT NULL")
    if "top1_class" not in existing:
        alters.append("ADD COLUMN top1_class VARCHAR(50) NULL")
    if "top1_score" not in existing:
        alters.append("ADD COLUMN top1_score FLOAT NULL")
    if "confidence_level" not in existing:
        alters.append("ADD COLUMN confidence_level VARCHAR(100) NULL")
    if "expected_class_source" not in existing:
        alters.append("ADD COLUMN expected_class_source VARCHAR(50) NULL")
    if "stage1_decision" not in existing:
        alters.append("ADD COLUMN stage1_decision VARCHAR(100) NULL")
    if "softmax_scores_json" not in existing:
        alters.append("ADD COLUMN softmax_scores_json TEXT NULL")
    if "preprocessing_steps_json" not in existing:
        alters.append("ADD COLUMN preprocessing_steps_json TEXT NULL")
    if "calculation_notes" not in existing:
        alters.append("ADD COLUMN calculation_notes TEXT NULL")

    for clause in alters:
        db.session.execute(text(f"ALTER TABLE {table} {clause}"))
    if alters:
        db.session.commit()

    cols = {col["name"]: col for col in inspector.get_columns(table)}
    nullable_updates = []
    if "predicted_class" in cols and not cols["predicted_class"].get("nullable", True):
        nullable_updates.append("MODIFY predicted_class VARCHAR(50) NULL")
    if "confidence" in cols and not cols["confidence"].get("nullable", True):
        nullable_updates.append("MODIFY confidence FLOAT NULL")
    if "naskhi_score" in cols and not cols["naskhi_score"].get("nullable", True):
        nullable_updates.append("MODIFY naskhi_score FLOAT NULL")
    if "diwani_score" in cols and not cols["diwani_score"].get("nullable", True):
        nullable_updates.append("MODIFY diwani_score FLOAT NULL")
    if "diwani_jali_score" in cols and not cols["diwani_jali_score"].get("nullable", True):
        nullable_updates.append("MODIFY diwani_jali_score FLOAT NULL")
    if "tsuluts_score" in cols and not cols["tsuluts_score"].get("nullable", True):
        nullable_updates.append("MODIFY tsuluts_score FLOAT NULL")

    for clause in nullable_updates:
        db.session.execute(text(f"ALTER TABLE {table} {clause}"))
    if nullable_updates:
        db.session.commit()


def ensure_model_versions_table() -> None:
    """Create model_versions table if missing."""
    inspector = inspect(db.engine)
    if "model_versions" in inspector.get_table_names() or "versi_model" in inspector.get_table_names():
        return
    db.session.execute(text("""
        CREATE TABLE model_versions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            model_source VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            model_runtime VARCHAR(100) NULL,
            training_date DATETIME NULL,
            num_classes INT NULL DEFAULT 4,
            dataset_images INT NULL,
            evaluation_accuracy FLOAT NULL,
            notes TEXT NULL,
            is_active TINYINT(1) NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """))
    db.session.commit()

"""EEG preprocessing and outcome extraction."""

from cog_surp.eeg.cluster import (
    ClusterAnalysisConfig,
    ClusterArtifacts,
    run_sensor_time_cluster_analysis,
)
from cog_surp.eeg.cohort import (
    CohortArtifacts,
    SubjectRun,
    aggregate_erp_core_cohort,
    discover_erp_core_subject_runs,
    paired_condition_effect,
)
from cog_surp.eeg.derco import (
    DERCoExtractionArtifacts,
    DERCoPreprocessingConfig,
    extract_derco_subject_article,
)
from cog_surp.eeg.preprocessing import (
    ERPPreprocessingConfig,
    preprocess_erp_core_subject,
)

__all__ = [
    "ClusterAnalysisConfig",
    "ClusterArtifacts",
    "CohortArtifacts",
    "DERCoExtractionArtifacts",
    "DERCoPreprocessingConfig",
    "ERPPreprocessingConfig",
    "SubjectRun",
    "aggregate_erp_core_cohort",
    "discover_erp_core_subject_runs",
    "extract_derco_subject_article",
    "paired_condition_effect",
    "preprocess_erp_core_subject",
    "run_sensor_time_cluster_analysis",
]

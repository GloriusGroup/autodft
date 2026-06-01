import pytest
from unittest.mock import patch
from dft_pipeline_execution.job_level.job_optimization import OptimizationJob

@patch("dft_pipeline_execution.job_level.base_job.Job._get_failed_attempt_info")
def test_optimization_maxiter_added(mock_get_info, mock_db_connection):
    mock_get_info.return_value = ("/fake/path", "Optimization Convergence")
    job = OptimizationJob(1, mock_db_connection)

    original = "*xyzfile 0 1\n"
    updated = job._increase_max_iter(original, 300)
    assert "MaxIter 300" in updated
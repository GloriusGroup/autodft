import pytest
from unittest.mock import patch
from dft_pipeline_execution.job_level.job_singlepoint import SinglepointJob

@patch("dft_pipeline_execution.job_level.base_job.Job._increase_resources")
@patch("dft_pipeline_execution.job_level.base_job.Job._get_failed_attempt_info")
def test_singlepoint_retry_triggers_resource_increase(mock_get_info, mock_increase, mock_db_connection):
    mock_get_info.return_value = ("/fake/path", "Termination")
    mock_increase.side_effect = lambda x: x + " increased"

    job = SinglepointJob(1, mock_db_connection)
    result = job._increase_resources("original")

    assert "increased" in result

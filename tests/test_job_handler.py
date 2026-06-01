import pytest
from unittest.mock import patch
from dft_pipeline_execution.job_level.job_handler import JobHandler

@patch("dft_pipeline_execution.job_level.job_handler.SinglepointJob.submit_to_slurm")
@patch("dft_pipeline_execution.job_level.job_handler.JobHandler._get_new_created_jobs")
def test_job_handler_dispatches_correct_class(mock_get_jobs, mock_submit, mock_db_connection):
    mock_get_jobs.return_value = [{"job_id": 1, "task_type": "singlepoint"}]
    handler = JobHandler(mock_db_connection)
    handler.submit_new_created_jobs()
    mock_submit.assert_called_once()
import pytest
from unittest.mock import patch
from dft_pipeline_execution.job_level.base_job import Job

class DummyJob(Job):
    job_type = "dummy"
    def modify_input_file(self): pass

@patch("dft_pipeline_execution.job_level.base_job.Job._create_db_entry", return_value=42)
def test_create_initial_job_creates_db_entry(mock_create_entry, mock_db_connection):
    job = DummyJob.create_initial_job(task_id=1, db_connection=mock_db_connection)
    assert isinstance(job, DummyJob)
    assert job.job_id == 42
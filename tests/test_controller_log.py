"""The controller also logs to <data_path>/logs/controller.log."""

from __future__ import annotations

import logging

from autodft.cli.run import _log_to_file
from autodft.config import Settings


def test_controller_logs_to_the_data_path(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    root = logging.getLogger()
    before = list(root.handlers)
    _log_to_file(settings)
    added = [h for h in root.handlers if h not in before]
    try:
        logging.getLogger("autodft.test").warning("hello log")
        for h in added:
            h.flush()
        assert "hello log" in (tmp_path / "logs" / "controller.log").read_text()
    finally:
        for h in added:
            root.removeHandler(h)
            h.close()

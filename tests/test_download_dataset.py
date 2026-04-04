"""Tests for the dataset download plan."""

from download_dataset import DatasetDownloader


def test_select_subjects_uses_supported_default_order(tmp_path):
    downloader = DatasetDownloader(output_dir=tmp_path)

    subjects = downloader.select_subjects()

    assert subjects[:5] == ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
    assert subjects[-1] == "sub-29"
    assert "sub-30" not in subjects


def test_task_runs_for_special_subject_skip_missing_outputs(tmp_path):
    downloader = DatasetDownloader(output_dir=tmp_path)

    task_runs = downloader.task_runs_for_subject("sub-29")

    assert ("rest", 1) in task_runs
    assert ("imagery", 3) in task_runs
    assert ("rest", 2) not in task_runs
    assert ("imagery", 4) not in task_runs


def test_download_connectivity_data_respects_max_subjects(tmp_path, monkeypatch):
    downloader = DatasetDownloader(output_dir=tmp_path)
    requested_paths = []

    def fake_download_file(remote_path, local_path, show_progress=True):
        requested_paths.append(remote_path)
        return True

    monkeypatch.setattr(downloader, "download_file", fake_download_file)

    downloader.download_connectivity_data(max_subjects=1)

    assert requested_paths
    assert all("/sub-02/" in remote_path for remote_path in requested_paths)
    assert len(requested_paths) == 12
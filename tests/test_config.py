"""Tests for configuration constants and subject metadata."""

from config import (
    SUBJECTS, N_SUBJECTS, LOR_TIME, ROR_TIME, SPECIAL_SUBJECTS,
    CONDITIONS, CONSCIOUS_CONDITIONS, UNCONSCIOUS_CONDITIONS,
    N_ROIS, FD_THRESHOLD, FD_COLUMN, TR, TRANSITION_BUFFER,
    RANDOM_STATE, RUN2_TOTAL_TRS, ATLAS,
)


class TestSubjects:
    def test_subject_count(self):
        assert N_SUBJECTS == 25
        assert len(SUBJECTS) == N_SUBJECTS

    def test_subject_ids_format(self):
        for s in SUBJECTS:
            assert s.startswith("sub-"), f"Subject {s} does not start with 'sub-'"
            num = int(s.split("-")[1])
            assert 1 <= num <= 30, f"Subject number {num} out of expected range"

    def test_no_duplicate_subjects(self):
        assert len(SUBJECTS) == len(set(SUBJECTS))

    def test_sub30_excluded(self):
        assert "sub-30" not in SUBJECTS


class TestTimingData:
    def test_lor_time_covers_all_subjects(self):
        for s in SUBJECTS:
            assert s in LOR_TIME, f"Missing LOR_TIME for {s}"

    def test_ror_time_covers_all_subjects(self):
        for s in SUBJECTS:
            assert s in ROR_TIME, f"Missing ROR_TIME for {s}"

    def test_lor_values_positive(self):
        for s, t in LOR_TIME.items():
            assert t > 0, f"LOR_TIME for {s} is not positive: {t}"

    def test_ror_values_positive(self):
        for s, t in ROR_TIME.items():
            assert t > 0, f"ROR_TIME for {s} is not positive: {t}"

    def test_lor_within_run2_bounds(self):
        for s, t in LOR_TIME.items():
            assert t < RUN2_TOTAL_TRS, f"LOR_TIME for {s} exceeds run-2 total TRs"

    def test_special_subjects_in_subject_list(self):
        for s in SPECIAL_SUBJECTS:
            assert s in SUBJECTS, f"Special subject {s} not in SUBJECTS"


class TestConditions:
    def test_seven_conditions(self):
        assert len(CONDITIONS) == 7

    def test_condition_indices_contiguous(self):
        assert set(CONDITIONS.keys()) == set(range(7))

    def test_conscious_unconscious_partition(self):
        all_conds = set(CONSCIOUS_CONDITIONS) | set(UNCONSCIOUS_CONDITIONS)
        assert all_conds == set(range(7))
        assert len(set(CONSCIOUS_CONDITIONS) & set(UNCONSCIOUS_CONDITIONS)) == 0

    def test_unconscious_is_lor(self):
        assert UNCONSCIOUS_CONDITIONS == [3]
        assert CONDITIONS[3] == "imagery_LOR"


class TestScanParameters:
    def test_n_rois(self):
        assert N_ROIS == 446

    def test_atlas_name(self):
        assert ATLAS == "4S456Parcels"

    def test_fd_threshold(self):
        assert FD_THRESHOLD == 0.8

    def test_fd_column_zero_indexed(self):
        assert FD_COLUMN == 7

    def test_tr_seconds(self):
        assert TR == 3.0

    def test_transition_buffer(self):
        assert TRANSITION_BUFFER == 375

    def test_random_state(self):
        assert RANDOM_STATE == 42

from feral.dataset import get_frame_ids


def test_every_frame():
    result = get_frame_ids(total_frames=10, chunk_shift=1, chunk_length=3, chunk_step=1)
    expected = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
    assert result == expected


def test_every_other_frame():
    result = get_frame_ids(total_frames=10, chunk_shift=2, chunk_length=3, chunk_step=2)
    expected = [[0, 2, 4], [2, 4, 6], [4, 6, 8]]
    assert result == expected


def test_every_third_frame():
    result = get_frame_ids(total_frames=15, chunk_shift=3, chunk_length=4, chunk_step=3)
    expected = [[0, 3, 6, 9], [3, 6, 9, 12]]
    assert result == expected


def test_not_enough_frames():
    result = get_frame_ids(total_frames=5, chunk_shift=1, chunk_length=6, chunk_step=1)
    assert result == []


def test_edge_case_exact_fit():
    result = get_frame_ids(total_frames=6, chunk_shift=3, chunk_length=2, chunk_step=3)
    expected = [[0, 3]]
    assert result == expected


def test_single_chunk():
    result = get_frame_ids(total_frames=10, chunk_shift=20, chunk_length=3, chunk_step=1)
    expected = [[0, 1, 2]]
    assert result == expected


def test_no_chunks_two_frames():
    result = get_frame_ids(total_frames=2, chunk_shift=1, chunk_length=3, chunk_step=1)
    assert result == []

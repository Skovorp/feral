import unittest

from dataset import get_frame_ids  

class TestGetFrameIds(unittest.TestCase):
    def test_every_frame(self):
        result = get_frame_ids(total_frames=10, chunk_shift=1, chunk_length=3, chunk_step=1)
        expected = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
        self.assertEqual(result, expected)

    def test_every_other_frame(self):
        result = get_frame_ids(total_frames=10, chunk_shift=2, chunk_length=3, chunk_step=2)
        expected = [[0, 2, 4], [2, 4, 6], [4, 6, 8]]
        self.assertEqual(result, expected)

    def test_every_third_frame(self):
        result = get_frame_ids(total_frames=15, chunk_shift=3, chunk_length=4, chunk_step=3)
        expected = [[0, 3, 6, 9], [3, 6, 9, 12]]
        self.assertEqual(result, expected)

    def test_not_enough_frames(self):
        result = get_frame_ids(total_frames=5, chunk_shift=1, chunk_length=6, chunk_step=1)
        self.assertEqual(result, [])

    def test_edge_case_exact_fit(self):
        result = get_frame_ids(total_frames=6, chunk_shift=3, chunk_length=2, chunk_step=3)
        expected = [[0, 3]]
        self.assertEqual(result, expected)

    def test_single_chunk(self):
        result = get_frame_ids(total_frames=10, chunk_shift=20, chunk_length=3, chunk_step=1)
        expected = [[0, 1, 2]]
        self.assertEqual(result, expected)

    def test_no_chunks(self):
        result = get_frame_ids(total_frames=2, chunk_shift=1, chunk_length=3, chunk_step=1)
        self.assertEqual(result, [])

    def test_no_chunks(self):
        result = get_frame_ids(total_frames=100, chunk_shift=1, chunk_length=3, chunk_step=1)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()

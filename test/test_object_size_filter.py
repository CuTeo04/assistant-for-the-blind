import unittest

from vision.object_size_filter import filter_object_data_by_size


class ObjectSizeFilterTest(unittest.TestCase):
    def _make_item(self, label: str, width_cm: float, height_cm: float):
        det = [0, 0, 10, 10, 0.9, 0.0]
        return (det, label, 1.0, width_cm, height_cm)

    def test_cup_within_range_is_kept(self):
        kept, rejected = filter_object_data_by_size([self._make_item("cup", 6.0, 12.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 0)

    def test_cup_outside_range_is_dropped(self):
        kept, rejected = filter_object_data_by_size([self._make_item("cup", 18.0, 30.0)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["label"], "cup")

    def test_bucket_within_range_is_kept(self):
        kept, rejected = filter_object_data_by_size([self._make_item("bucket", 24.0, 30.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 0)

    def test_bucket_too_small_is_dropped(self):
        kept, rejected = filter_object_data_by_size([self._make_item("bucket", 8.0, 10.0)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(rejected), 1)


if __name__ == "__main__":
    unittest.main()

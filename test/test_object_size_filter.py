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

    def test_cup_outside_range_is_relabeled_to_bucket(self):
        kept, relabeled = filter_object_data_by_size([self._make_item("cup", 18.0, 30.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], "bucket")
        self.assertEqual(len(relabeled), 1)
        self.assertEqual(relabeled[0]["from_label"], "cup")
        self.assertEqual(relabeled[0]["to_label"], "bucket")

    def test_bucket_within_range_is_kept(self):
        kept, rejected = filter_object_data_by_size([self._make_item("bucket", 24.0, 30.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 0)

    def test_bucket_too_small_is_dropped(self):
        kept, relabeled = filter_object_data_by_size([self._make_item("bucket", 8.0, 10.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], "cup")
        self.assertEqual(len(relabeled), 1)

    def test_mattress_uses_short_and_long_side(self):
        kept, rejected = filter_object_data_by_size([self._make_item("mattress", 200.0, 90.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 0)

    def test_plate_bbox_uses_footprint_not_thickness(self):
        kept, rejected = filter_object_data_by_size([self._make_item("plate", 24.0, 22.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 0)

    def test_extension_cord_allows_coiled_bbox(self):
        kept, rejected = filter_object_data_by_size([self._make_item("extension cord", 28.0, 18.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 0)

    def test_gas_stove_not_in_similar_shape_group_keeps_original_label(self):
        kept, rejected = filter_object_data_by_size([self._make_item("gas stove", 50.0, 45.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], "gas stove")
        self.assertEqual(len(rejected), 0)

    def test_spoon_can_be_relabeled_to_frying_pan_by_size(self):
        kept, relabeled = filter_object_data_by_size([self._make_item("spoon", 28.0, 36.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], "frying pan")
        self.assertEqual(len(relabeled), 1)

    def test_non_group_label_is_not_size_filtered(self):
        kept, relabeled = filter_object_data_by_size([self._make_item("chair", 10.0, 20.0)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], "chair")
        self.assertEqual(len(relabeled), 0)


if __name__ == "__main__":
    unittest.main()

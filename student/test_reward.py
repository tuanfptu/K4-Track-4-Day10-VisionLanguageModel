"""Required E5 reward checks. Run with: python -m unittest student.test_reward"""
import unittest

from student.reward import reward_one


class IoURewardTests(unittest.TestCase):
    gt = [{"label": "pothole", "bbox_2d": [100, 100, 300, 300]}]

    def test_perfect_box(self):
        self.assertAlmostEqual(reward_one('[{"label":"pothole","bbox_2d":[100,100,300,300]}]', self.gt), 1.0)

    def test_malformed_json(self):
        self.assertEqual(reward_one("not json", self.gt), 0.0)

    def test_wrong_class(self):
        self.assertAlmostEqual(reward_one('[{"label":"crack","bbox_2d":[100,100,300,300]}]', self.gt), 0.2)

    def test_extra_box_is_penalized(self):
        output = '[{"label":"pothole","bbox_2d":[100,100,300,300]},{"label":"crack","bbox_2d":[1,1,20,20]}]'
        self.assertAlmostEqual(reward_one(output, self.gt), 0.6)

    def test_both_empty(self):
        self.assertEqual(reward_one("[]", []), 1.0)


if __name__ == "__main__":
    unittest.main()

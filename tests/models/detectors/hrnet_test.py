import unittest

from propose.models.detectors import HRNet


class HRNetTests(unittest.TestCase):
    def test_smoke(self):
        model = HRNet()


if __name__ == "__main__":
    unittest.main()

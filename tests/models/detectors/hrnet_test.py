import unittest

from propose.models.detectors import HRNet

from unittest.mock import MagicMock, patch


class HRNetTests(unittest.TestCase):
    @patch("propose.models.detectors.hrnet.hrnet.wandb")
    @patch("propose.models.detectors.hrnet.hrnet.torch.load")
    def test_has_pretrained_option(self, wandb_mock, load_mock):
        load_mock.return_value = {}
        model = HRNet.from_pretrained("artifact", MagicMock())

        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()

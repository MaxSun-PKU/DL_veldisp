from pathlib import Path
import unittest

import h5py
import numpy as np
import torch

from physmodel import HybridAGNNet
from scripts.train_hybrid_sdss import load_h5_assets, load_h5_physical_config, velocity_pixel_scale


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "artifacts" / "reference_model" / "inference_assets.h5"
CHECKPOINT = ROOT / "artifacts" / "reference_model" / "model.pth"


class ReferenceModelTest(unittest.TestCase):
    def test_compact_assets_have_no_training_spectra(self):
        with h5py.File(ASSETS, "r") as handle:
            self.assertTrue(bool(handle.attrs["asset_only"]))
            self.assertNotIn("spectra", handle)
            self.assertEqual(handle["wave_grid"].shape, (6000,))
            self.assertEqual(handle["template_library"].shape, (288, 6000))
            self.assertEqual(handle["feii_template_bank"].shape[1], 6000)

    def test_reference_checkpoint_forward_pass(self):
        wave, templates, feii, _, _, sigma_min, sigma_max = load_h5_assets(ASSETS)
        physical = load_h5_physical_config(ASSETS)
        state = torch.load(CHECKPOINT, map_location="cpu")
        input_channels = int(state["backbone.0.weight"].shape[1])
        self.assertEqual(input_channels, 10)

        model = HybridAGNNet(
            input_dim=len(wave),
            template_library=templates,
            dv_per_pixel=velocity_pixel_scale(wave),
            input_channels=input_channels,
            wave_grid=wave,
            fe_template=feii,
            instrument_sigma_kms=physical["instrument_sigma_kms"],
            template_sigma_inst_kms=physical["template_sigma_inst_kms"],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        model.load_state_dict(state)
        model.eval()

        inputs = torch.zeros((1, input_channels, len(wave)), dtype=torch.float32)
        inputs[:, 0, :] = 1.0
        inputs[:, 1, :] = 1.0
        inputs[:, 2, :] = 1.0
        inputs[:, 4, :] = physical["instrument_sigma_kms"] / 100.0
        inputs[:, 5, :] = 1.0
        inputs[:, 8, :] = 0.0
        inputs[:, 9, :] = 1.0

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(len(outputs), 7)
        for output in outputs:
            self.assertTrue(bool(torch.isfinite(output).all()))
        pred_sigma, _, pred_sfh, pred_host, recon, agn, host = outputs
        self.assertLessEqual(sigma_min, float(pred_sigma.item()))
        self.assertLessEqual(float(pred_sigma.item()), sigma_max)
        self.assertTrue(np.isclose(float(pred_sfh.sum().item()), 1.0, atol=1e-5))
        self.assertGreaterEqual(float(pred_host.item()), 0.0)
        self.assertLessEqual(float(pred_host.item()), 1.0)
        self.assertEqual(recon.shape, (1, len(wave)))
        self.assertEqual(agn.shape, recon.shape)
        self.assertEqual(host.shape, recon.shape)


if __name__ == "__main__":
    unittest.main()

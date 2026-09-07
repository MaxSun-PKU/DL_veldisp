import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json


SPLIT_NAME_TO_CODE = {"train": 0, "val": 1, "test": 2}

class MockAGNDataset(Dataset):
    def __init__(
        self,
        h5_path,
        channels=("flux",),
        flux_normalization=None,
        return_sfh=False,
        split=None,
        indices=None,
        min_host_fraction=None,
        min_snr=None,
        min_redshift=None,
        max_redshift=None,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.channels = tuple(channels)
        self.flux_normalization = flux_normalization
        self.return_sfh = return_sfh
        self.split = split
        self.filters = {
            "min_host_fraction": min_host_fraction,
            "min_snr": min_snr,
            "min_redshift": min_redshift,
            "max_redshift": max_redshift,
        }
        self.dataset_file = None
        self.indices = None

        # 初始化时只读长度，读完立刻关闭（with 语句自动关闭）
        with h5py.File(self.h5_path, 'r') as f:
            total_length = len(f['spectra'])
            self.available_keys = set(f.keys())
            config = json.loads(f.attrs.get("config_json", "{}"))
            self.default_sigma_inst_kms = float(config.get("sdss_sigma_inst", 70.0))
            if indices is not None:
                self.indices = np.asarray(indices, dtype=np.int64)
            elif split is not None:
                split_key = f"{split}_indices"
                if split_key in f:
                    self.indices = f[split_key][:].astype(np.int64)
                elif "split" in f and split in SPLIT_NAME_TO_CODE:
                    self.indices = np.flatnonzero(f["split"][:] == SPLIT_NAME_TO_CODE[split]).astype(np.int64)
                else:
                    raise ValueError(f"Split {split!r} not found in {self.h5_path}")
            if any(value is not None for value in self.filters.values()):
                base_indices = self.indices if self.indices is not None else np.arange(total_length, dtype=np.int64)
                keep = np.ones(len(base_indices), dtype=bool)
                if min_host_fraction is not None and "host_fraction" in f:
                    keep &= f["host_fraction"][base_indices] >= float(min_host_fraction)
                if min_snr is not None and "snr" in f:
                    keep &= f["snr"][base_indices] >= float(min_snr)
                if min_redshift is not None and "redshift" in f:
                    keep &= f["redshift"][base_indices] >= float(min_redshift)
                if max_redshift is not None and "redshift" in f:
                    keep &= f["redshift"][base_indices] <= float(max_redshift)
                self.indices = base_indices[keep].astype(np.int64)
            self.length = len(self.indices) if self.indices is not None else total_length

        split_text = f", split={self.split}" if self.split is not None else ""
        filter_text = ""
        active_filters = {k: v for k, v in self.filters.items() if v is not None}
        if active_filters:
            filter_text = f", filters={active_filters}"
        print(f"📦 [DataLoader] 采用懒加载模式映射了 {self.length} 条数据。channels={self.channels}{split_text}{filter_text}")

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state['dataset_file'] = None
        return state

    def _get_file(self):
        if not hasattr(self, 'dataset_file') or self.dataset_file is None:
            # 开启 swmr=True (Single Writer Multiple Reader) 模式，专门针对多进程并发读取优化。
            self.dataset_file = h5py.File(self.h5_path, 'r', swmr=True)
        return self.dataset_file

    def _normalize_flux(self, flux):
        if self.flux_normalization is None:
            return flux
        if self.flux_normalization == "median":
            scale = torch.median(flux[torch.isfinite(flux)]).clamp_min(1e-6)
            return flux / scale
        if self.flux_normalization == "center":
            scale = torch.median(flux[torch.isfinite(flux)]).clamp_min(1e-6)
            return flux / scale - 1.0
        raise ValueError(f"Unknown flux_normalization={self.flux_normalization!r}")

    def _read_optional_1d(self, f, key, idx, default_value):
        if key in f:
            return torch.FloatTensor(f[key][idx])
        return torch.full_like(torch.FloatTensor(f['spectra'][idx]), float(default_value))

    def _build_channels(self, f, idx):
        flux = torch.FloatTensor(f['spectra'][idx])
        flux = self._normalize_flux(flux)
        built = []
        for channel in self.channels:
            if channel == "flux":
                built.append(flux)
            elif channel == "ivar":
                ivar = self._read_optional_1d(f, 'ivar', idx, 1.0)
                positive = ivar[ivar > 0]
                if positive.numel() > 0:
                    ivar = ivar / torch.median(positive).clamp_min(1e-6)
                built.append(ivar.clamp(0.0, 10.0))
            elif channel == "sqrt_ivar":
                ivar = self._read_optional_1d(f, 'ivar', idx, 1.0)
                positive = ivar[ivar > 0]
                if positive.numel() > 0:
                    ivar = ivar / torch.median(positive).clamp_min(1e-6)
                built.append(torch.sqrt(ivar.clamp_min(0.0)).clamp(0.0, 5.0))
            elif channel == "mask":
                mask_key = 'fit_mask' if 'fit_mask' in f else 'valid_pixels'
                built.append(self._read_optional_1d(f, mask_key, idx, 1.0).clamp(0.0, 1.0))
            elif channel == "sigma_mask":
                built.append(self._read_optional_1d(f, 'sigma_mask', idx, 1.0).clamp(0.0, 1.0))
            elif channel == "sigma_flux":
                sigma_mask = self._read_optional_1d(f, 'sigma_mask', idx, 1.0).clamp(0.0, 1.0)
                values = flux[sigma_mask > 0.5]
                if values.numel() > 0:
                    center = torch.median(values)
                    scale = torch.median(torch.abs(values - center)).clamp_min(1e-3)
                    sigma_flux = ((flux - center) / scale).clamp(-8.0, 8.0) * sigma_mask
                else:
                    sigma_flux = torch.zeros_like(flux)
                built.append(sigma_flux)
            elif channel in ("inst_sigma", "instrument_sigma"):
                if "inst_sigma_kms_grid" in f:
                    inst_grid = torch.FloatTensor(f["inst_sigma_kms_grid"][idx]) / 100.0
                    built.append(inst_grid.clamp(0.0, 3.0))
                    continue
                if "sigma_inst_kms" in f:
                    sigma_inst = float(f["sigma_inst_kms"][idx])
                else:
                    sigma_inst = self.default_sigma_inst_kms
                built.append(torch.full_like(flux, sigma_inst / 100.0).clamp(0.0, 3.0))
            elif channel in ("region_cat", "cat_region"):
                region_code = int(f["region_code"][idx]) if "region_code" in f else -1
                built.append(torch.full_like(flux, 1.0 if region_code == 0 else 0.0))
            elif channel in ("region_fe", "fe_region"):
                region_code = int(f["region_code"][idx]) if "region_code" in f else -1
                built.append(torch.full_like(flux, 1.0 if region_code == 1 else 0.0))
            elif channel in ("region_cahk", "region_ca_hk", "ca_hk_region"):
                region_code = int(f["region_code"][idx]) if "region_code" in f else -1
                built.append(torch.full_like(flux, 1.0 if region_code == 2 else 0.0))
            elif channel in ("agn_log_l5100", "log_l5100", "l5100"):
                log_l5100 = float(f["agn_log_l5100"][idx]) if "agn_log_l5100" in f else 45.0
                built.append(torch.full_like(flux, (log_l5100 - 45.0) / 2.0).clamp(-2.0, 2.0))
            elif channel == "resolution_ratio":
                if "sigma_resolution_ratio" in f:
                    ratio = float(f["sigma_resolution_ratio"][idx])
                elif "sigma_inst_kms" in f:
                    ratio = float(f["sigma_star"][idx]) / max(float(f["sigma_inst_kms"][idx]), 1e-6)
                else:
                    ratio = float(f["sigma_star"][idx]) / max(self.default_sigma_inst_kms, 1e-6)
                built.append(torch.full_like(flux, ratio / 5.0).clamp(0.0, 3.0))
            elif channel == "continuum_mask":
                built.append(self._read_optional_1d(f, 'continuum_mask', idx, 1.0).clamp(0.0, 1.0))
            else:
                raise ValueError(f"Unknown channel={channel!r}")
        return torch.stack(built, dim=0)

    def __getitem__(self, idx):
        f = self._get_file()
        if self.indices is not None:
            idx = int(self.indices[idx])
        x = self._build_channels(f, idx)
        y_sigma_kms = torch.FloatTensor([f['sigma_star'][idx]])
        y_host = torch.FloatTensor([f['host_fraction'][idx]]) if 'host_fraction' in f else torch.FloatTensor([float('nan')])
        y_snr = torch.FloatTensor([f['snr'][idx]]) if 'snr' in f else torch.FloatTensor([float('nan')])

        if self.return_sfh and 'sfh_weights' in f:
            y_sfh = torch.FloatTensor(f['sfh_weights'][idx])
            return x, y_sigma_kms, y_host, y_snr, y_sfh
        return x, y_sigma_kms, y_host, y_snr

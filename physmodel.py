import torch
import torch.nn as nn
import torch.nn.functional as F

REGION_WINDOWS = (
    ("cat", 8470.0, 8700.0),
    ("fe", 5250.0, 5820.0),
    ("ca_hk", 3910.0, 4060.0),
)


def _positive_bottleneck(in_planes, ratio):
    return max(1, in_planes // ratio)


def split_spectrum_channels(x):
    """
    Accepts either a classic one-channel spectrum or a richer SDSS-style tensor.
    Channel convention: flux, normalized inverse-variance, fit mask.
    """
    if x.dim() == 2:
        return x, None, None
    flux = x[:, 0, :]
    ivar = x[:, 1, :] if x.size(1) > 1 else None
    fit_mask = x[:, 2, :] if x.size(1) > 2 else None
    return flux, ivar, fit_mask

# ==========================================
# 1. 1D-CBAM 注意力机制 (用于锁定吸收线波段)
# ==========================================
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention1D, self).__init__()
        hidden = _positive_bottleneck(in_planes, ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1   = nn.Conv1d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM1D(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(in_planes, ratio)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# ==========================================
# 2. 多尺度 Inception 模块 (捕捉不同宽度的谱线特征)
# ==========================================
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock1D, self).__init__()
        bottleneck = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck, kernel_size=1),
            nn.BatchNorm1d(bottleneck), nn.GELU(),
            nn.Conv1d(bottleneck, bottleneck, kernel_size=3, padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck, kernel_size=1),
            nn.BatchNorm1d(bottleneck), nn.GELU(),
            nn.Conv1d(bottleneck, bottleneck, kernel_size=11, padding=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck, kernel_size=1),
            nn.BatchNorm1d(bottleneck), nn.GELU(),
            nn.Conv1d(bottleneck, bottleneck, kernel_size=31, padding=15)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, bottleneck, kernel_size=1)
        )
        self.proj = nn.Conv1d(bottleneck * 4, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.proj(out)


class StatsPool1D(nn.Module):
    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        max_value = torch.amax(x, dim=-1)
        centered = x - mean.unsqueeze(-1)
        std = torch.sqrt(torch.mean(centered**2, dim=-1).clamp_min(1e-6))
        return torch.cat([mean, max_value, std], dim=1)

# ==========================================
# 3. 物理分支：可微星族卷积引擎
# ==========================================
class SPSPhysicsBroadener(nn.Module):
    """
    输入: 权重(sfh)和速度弥散(sigma)
    输出: 只有恒星特征的归一化连续谱
    """
    def __init__(
        self,
        template_library,
        dv_per_pixel=69.0,
        kernel_size=101,
        instrument_sigma_kms=70.0,
        template_sigma_inst_kms=0.0,
    ):
        super().__init__()
        self.register_buffer('templates', torch.FloatTensor(template_library))
        self.dv_per_pixel = dv_per_pixel
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.instrument_sigma_kms = float(instrument_sigma_kms)
        self.template_sigma_inst_kms = float(template_sigma_inst_kms)
        x = torch.arange(kernel_size).float() - self.padding
        self.register_buffer('x_grid', x)

    def forward(self, sigma_kms, sfh_weights, instrument_sigma_kms=None):
        batch_size = sigma_kms.shape[0]
        sfh_weights = sfh_weights / sfh_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # 1. 基底组合 (Mass-weighted combination)
        combined_stellar = torch.matmul(sfh_weights, self.templates)

        # 2. 生成高斯核。对真实 SDSS，instrument_sigma 可由 WDISP/dispRes
        # 转成 km/s 后作为输入通道；如果没有，就退回训练集的 median 值。
        if instrument_sigma_kms is None:
            inst = sigma_kms.new_full(sigma_kms.shape, self.instrument_sigma_kms)
        else:
            inst = instrument_sigma_kms.to(sigma_kms.device, dtype=sigma_kms.dtype).view_as(sigma_kms)
        sigma_eff2 = sigma_kms.clamp_min(1e-3) ** 2 + inst.clamp_min(0.0) ** 2
        sigma_conv2 = (sigma_eff2 - self.template_sigma_inst_kms ** 2).clamp_min(0.25**2)
        sigma_eff = torch.sqrt(sigma_conv2)
        sigma_pix = (sigma_eff / self.dv_per_pixel).clamp_min(0.25)
        kernel = torch.exp(-0.5 * (self.x_grid.view(1, -1) / (sigma_pix + 1e-6))**2)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)

        # 3. 物理展宽卷积
        x_reshaped = combined_stellar.view(1, batch_size, -1)
        k_reshaped = kernel.view(batch_size, 1, -1)
        if combined_stellar.size(1) > self.padding:
            x_reshaped = F.pad(x_reshaped, (self.padding, self.padding), mode='reflect')
            broadened = F.conv1d(x_reshaped, k_reshaped, groups=batch_size)
        else:
            broadened = F.conv1d(x_reshaped, k_reshaped, padding=self.padding, groups=batch_size)
        return broadened.view(batch_size, -1)


class AGNContaminationBranch(nn.Module):
    """
    Park et al. (2022) motivates treating optical Fe II as a real pseudo-continuum,
    not an arbitrary residual.  When wave_grid/fe_template are provided, this branch
    builds a positive AGN component from power law + broadened Fe II + weak residual
    line structure. Without them it falls back to the old nonnegative residual head.
    """
    def __init__(self, feature_dim, input_dim, wave_grid=None, fe_template=None, dv_per_pixel=69.0, kernel_size=101):
        super().__init__()
        self.input_dim = input_dim
        self.dv_per_pixel = dv_per_pixel
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.residual_head = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(),
            nn.Linear(512, input_dim),
            nn.Softplus()
        )

        x = torch.arange(kernel_size).float() - self.padding
        self.register_buffer('x_grid', x)

        if wave_grid is not None:
            self.register_buffer('wave_grid', torch.FloatTensor(wave_grid))
        else:
            self.wave_grid = None

        if fe_template is not None:
            fe_tensor = torch.FloatTensor(fe_template)
            if fe_tensor.dim() == 1:
                fe_tensor = fe_tensor.unsqueeze(0)
            normalized = []
            for row in fe_tensor:
                positive = row[row > 0]
                if positive.numel() > 0:
                    row = row / positive.median().clamp_min(1e-6)
                normalized.append(row)
            fe_tensor = torch.stack(normalized, dim=0)
            self.register_buffer('fe_templates', fe_tensor)
            self.n_fe_templates = fe_tensor.size(0)
        else:
            self.fe_templates = None
            self.n_fe_templates = 0

        # alpha, Fe amplitude, Fe width, residual amplitude, plus optional template-mixture logits.
        self.param_head = nn.Linear(feature_dim, 4 + self.n_fe_templates)
        self.detach_broadened_fe_on_mps = True

    def _broaden_fe_template(self, sigma_kms, template_weights, detach_broadening=False):
        batch_size = sigma_kms.shape[0]
        if detach_broadening:
            sigma_kms = sigma_kms.detach()
            template_weights = template_weights.detach()
        weighted_template = torch.matmul(template_weights, self.fe_templates)
        sigma_pix = (sigma_kms / self.dv_per_pixel).clamp_min(0.25)
        kernel = torch.exp(-0.5 * (self.x_grid.view(1, -1) / (sigma_pix + 1e-6))**2)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        template = weighted_template.view(1, batch_size, -1)
        k_reshaped = kernel.view(batch_size, 1, -1)
        if weighted_template.size(1) > self.padding:
            template = F.pad(template, (self.padding, self.padding), mode='reflect')
            broadened = F.conv1d(template, k_reshaped, groups=batch_size)
        else:
            broadened = F.conv1d(template, k_reshaped, padding=self.padding, groups=batch_size)
        broadened = broadened.view(batch_size, -1)
        return broadened.detach() if detach_broadening else broadened

    def forward(self, features):
        residual = self.residual_head(features)
        if self.wave_grid is None:
            return residual, {}

        raw_params = self.param_head(features)
        raw_alpha = raw_params[:, 0:1]
        raw_fe_amp = raw_params[:, 1:2]
        raw_fe_width = raw_params[:, 2:3]
        raw_resid_amp = raw_params[:, 3:4]
        alpha = -2.8 + 3.8 * torch.sigmoid(raw_alpha)
        fe_amp = 2.5 * torch.sigmoid(raw_fe_amp)
        fe_sigma = 250.0 + 4500.0 * torch.sigmoid(raw_fe_width)
        resid_amp = 0.25 * torch.sigmoid(raw_resid_amp)

        wave = (self.wave_grid / 5100.0).view(1, -1).clamp_min(1e-4)
        power_law = torch.exp(alpha * torch.log(wave))

        agn = power_law
        aux = {
            'alpha': alpha,
            'fe_amp': fe_amp,
            'fe_sigma_kms': fe_sigma,
            'residual_amp': resid_amp,
        }
        if self.fe_templates is not None:
            template_logits = raw_params[:, 4:]
            template_weights = torch.softmax(template_logits, dim=1)
            detach_fe_broadening = (
                self.training
                and features.device.type == "mps"
                and self.detach_broadened_fe_on_mps
            )
            agn = agn + fe_amp * self._broaden_fe_template(
                fe_sigma,
                template_weights,
                detach_broadening=detach_fe_broadening,
            )
            aux['fe_template_weights'] = template_weights
        agn = agn + resid_amp * residual
        agn = agn / agn.mean(dim=1, keepdim=True).clamp_min(1e-6)

        return agn, aux

# ==========================================
# 4. 主网络：HybridAGNNet
# ==========================================
class HybridAGNNet(nn.Module):
    def __init__(
        self,
        input_dim,
        template_library,
        dv_per_pixel=69.0,
        input_channels=1,
        wave_grid=None,
        fe_template=None,
        instrument_sigma_kms=70.0,
        template_sigma_inst_kms=0.0,
        sigma_min=80.0,
        sigma_max=750.0,
    ):
        super().__init__()
        self.num_templates = template_library.shape[0]
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.instrument_sigma_kms = float(instrument_sigma_kms)
        self.last_agn_aux = {}
        self.last_region_weights = None
        self.last_region_sigma = None
        self.detach_mps_broadening_for_reconstruction = True
        self.detach_mps_reconstruction_encoder = True
        if wave_grid is not None:
            wave_tensor = torch.FloatTensor(wave_grid)
            self.register_buffer('wave_grid', wave_tensor)
            region_masks = []
            for _, lo, hi in REGION_WINDOWS:
                region_masks.append(((wave_tensor >= lo) & (wave_tensor <= hi)).float())
            self.register_buffer('region_masks', torch.stack(region_masks, dim=0))
        else:
            self.wave_grid = None
            self.region_masks = None

        # --- 特征提取 Backbone ---
        self.backbone = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(32), nn.GELU(),
            InceptionBlock1D(32, 64),
            CBAM1D(64),
            nn.MaxPool1d(2, 2),
            InceptionBlock1D(64, 128),
            CBAM1D(128),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten() # [Batch, 128]
        )
        self.sigma_context = nn.Sequential(
            nn.Conv1d(input_channels, 24, kernel_size=9, padding=4, stride=2),
            nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=15, padding=7, stride=2),
            nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=31, padding=15, stride=2),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.sigma_context_pool = StatsPool1D()
        self.line_shape_context = nn.Sequential(
            nn.Conv1d(3, 24, kernel_size=9, padding=4, stride=2),
            nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=15, padding=7, stride=2),
            nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=31, padding=15, stride=2),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.line_shape_context_pool = StatsPool1D()

        # --- 物理参数头 ---
        self.sigma_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.sigma_delta_head = nn.Sequential(nn.Linear(64 * 3, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.sigma_delta_head[-1].weight)
        nn.init.zeros_(self.sigma_delta_head[-1].bias)
        self.sigma_line_delta_head = nn.Sequential(nn.Linear(64 * 3, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.sigma_line_delta_head[-1].weight)
        nn.init.zeros_(self.sigma_line_delta_head[-1].bias)
        self.sigma_region_delta_head = nn.Sequential(nn.Linear(128 + 64 * 3, 96), nn.GELU(), nn.Linear(96, 3))
        nn.init.zeros_(self.sigma_region_delta_head[-1].weight)
        nn.init.zeros_(self.sigma_region_delta_head[-1].bias)
        self.sigma_line_region_delta_head = nn.Sequential(nn.Linear(64 * 3, 48), nn.GELU(), nn.Linear(48, 3))
        nn.init.zeros_(self.sigma_line_region_delta_head[-1].weight)
        nn.init.zeros_(self.sigma_line_region_delta_head[-1].bias)
        self.sigma_bin_head = nn.Sequential(
            nn.Linear(128 + 64 * 3 + 64 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 7),
        )
        self.last_sigma_bin_logits = None
        self.sigma_logvar_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.sfh_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, self.num_templates), nn.Softmax(dim=-1))

        # --- 🌟 混合头：预测 Host Fraction ---
        self.host_frac_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 限制在 0-1 之间
        )

        # --- 分支 B & C: 物理生成层与残差生成层 ---
        self.physics_branch = SPSPhysicsBroadener(
            template_library,
            dv_per_pixel=dv_per_pixel,
            instrument_sigma_kms=instrument_sigma_kms,
            template_sigma_inst_kms=template_sigma_inst_kms,
        )
        self.agn_branch = AGNContaminationBranch(
            128,
            input_dim,
            wave_grid=wave_grid,
            fe_template=fe_template,
            dv_per_pixel=dv_per_pixel,
        )

    def load_state_dict(self, state_dict, strict=True):
        legacy_prefix = 'agn_residual_head.'
        if any(key.startswith(legacy_prefix) for key in state_dict.keys()):
            remapped = {}
            for key, value in state_dict.items():
                if key.startswith(legacy_prefix):
                    key = 'agn_branch.residual_head.' + key[len(legacy_prefix):]
                remapped[key] = value
            state_dict = remapped
            strict = False
        new_sigma_keys = (
            'sigma_context.',
            'sigma_delta_head.',
            'sigma_context_pool.',
            'line_shape_context.',
            'line_shape_context_pool.',
            'sigma_line_delta_head.',
            'sigma_line_region_delta_head.',
            'sigma_region_delta_head.',
            'sigma_bin_head.',
        )
        if not all(any(key.startswith(prefix) for key in state_dict.keys()) for prefix in new_sigma_keys):
            strict = False
        current = super().state_dict()
        compatible = {}
        for key, value in state_dict.items():
            if key in current and current[key].shape != value.shape:
                strict = False
                continue
            compatible[key] = value
        state_dict = compatible
        if set(current.keys()) - set(state_dict.keys()):
            strict = False
        return super().load_state_dict(state_dict, strict=strict)

    def _sigma_focused_input(self, x):
        if x.dim() != 3 or x.size(1) < 4:
            return x
        mask = x[:, -1:, :].clamp(0.0, 1.0)
        focused = x.clone()
        focused[:, 0:1, :] = focused[:, 0:1, :] * mask
        if focused.size(1) > 1:
            focused[:, 1:2, :] = focused[:, 1:2, :] * mask
        return focused

    def _line_shape_input(self, x):
        if x.dim() != 3:
            raise ValueError("HybridAGNNet expects a 3D tensor [batch, channels, pixels]")
        mask = x[:, -1:, :].clamp(0.0, 1.0)
        if x.size(1) > 3:
            line_flux = x[:, 3:4, :]
        else:
            line_flux = x[:, 0:1, :]
            finite_center = torch.median(line_flux, dim=-1, keepdim=True).values
            line_flux = line_flux - finite_center
        line_flux = line_flux * mask
        d1 = F.pad(line_flux[:, :, 1:] - line_flux[:, :, :-1], (1, 0)) * mask
        d2 = F.pad(d1[:, :, 1:] - d1[:, :, :-1], (1, 0)) * mask
        return torch.cat([line_flux, d1, d2], dim=1).clamp(-12.0, 12.0)

    def _instrument_sigma_from_input(self, x):
        if x.dim() != 3 or x.size(1) < 6:
            return None
        # Convention for the improved SDSS training channels:
        # flux, ivar, mask, sigma_flux, inst_sigma/100, [...], sigma_mask.
        candidate = x[:, 4, :].clamp_min(0.0)
        sigma_kms = candidate.median(dim=1).values.view(-1, 1) * 100.0
        plausible = (sigma_kms > 20.0) & (sigma_kms < 180.0)
        if not torch.any(plausible):
            return None
        fallback = sigma_kms.new_full(sigma_kms.shape, self.instrument_sigma_kms)
        return torch.where(plausible, sigma_kms, fallback)

    def _region_weights_from_input(self, x):
        batch_size = x.size(0)
        if x.dim() == 3 and x.size(1) >= 9:
            region_values = torch.stack(
                [
                    x[:, 5, :].median(dim=1).values,
                    x[:, 6, :].median(dim=1).values,
                    x[:, 7, :].median(dim=1).values,
                ],
                dim=1,
            ).clamp_min(0.0)
            total = region_values.sum(dim=1, keepdim=True)
            if torch.any(total > 0):
                fallback = region_values.new_full(region_values.shape, 1.0 / 3.0)
                return torch.where(total > 0, region_values / total.clamp_min(1e-6), fallback)

        if self.region_masks is None or x.dim() != 3:
            return x.new_full((batch_size, 3), 1.0 / 3.0)

        sigma_mask = x[:, -1, :].clamp(0.0, 1.0)
        masks = self.region_masks.to(device=x.device, dtype=x.dtype)
        coverage = torch.einsum('bl,rl->br', sigma_mask, masks)
        # Prefer CaT when available, then the Fe 5250-5820 window, then Ca H+K.
        priority = x.new_tensor([3.0, 1.5, 1.0]).view(1, 3)
        weighted = coverage * priority
        total = weighted.sum(dim=1, keepdim=True)
        fallback = x.new_full((batch_size, 3), 1.0 / 3.0)
        return torch.where(total > 0, weighted / total.clamp_min(1e-6), fallback)

    def forward(self, x):
        features = self.backbone(x)
        sigma_features = self.sigma_context_pool(self.sigma_context(self._sigma_focused_input(x)))
        line_features = self.line_shape_context_pool(self.line_shape_context(self._line_shape_input(x)))
        sigma_bin_features = torch.cat([features, sigma_features, line_features], dim=1)
        self.last_sigma_bin_logits = self.sigma_bin_head(sigma_bin_features)

        # 1. 物理参数推断
        sigma_raw_base = (
            self.sigma_head(features)
            + self.sigma_delta_head(sigma_features)
            + self.sigma_line_delta_head(line_features)
        )
        region_weights = self._region_weights_from_input(x)
        self.last_region_weights = region_weights.detach()
        region_delta = (
            self.sigma_region_delta_head(torch.cat([features, sigma_features], dim=1))
            + self.sigma_line_region_delta_head(line_features)
        )
        region_sigma_raw = sigma_raw_base + region_delta
        self.last_region_sigma = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(region_sigma_raw)
        ).detach()
        sigma_raw = sigma_raw_base + (region_delta * region_weights).sum(dim=1, keepdim=True)
        pred_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma_raw)
        pred_logvar = self.sigma_logvar_head(features)
        pred_sfh = self.sfh_head(features)
        pred_host_frac = self.host_frac_head(features)

        # 2. 独立分量生成
        mps_safe_reconstruction = self.training and x.device.type == "mps"
        detach_stellar_broadening = mps_safe_reconstruction and self.detach_mps_broadening_for_reconstruction
        detach_reconstruction_encoder = mps_safe_reconstruction and self.detach_mps_reconstruction_encoder
        recon_features = features.detach() if detach_reconstruction_encoder else features
        instrument_sigma = self._instrument_sigma_from_input(x)
        if detach_stellar_broadening:
            instrument_sigma = instrument_sigma.detach() if instrument_sigma is not None else None
        stellar_base = self.physics_branch(
            pred_sigma.detach() if detach_stellar_broadening else pred_sigma,
            pred_sfh.detach() if detach_stellar_broadening else pred_sfh,
            instrument_sigma_kms=instrument_sigma,
        )
        if detach_stellar_broadening:
            stellar_base = stellar_base.detach()
        agn_base, self.last_agn_aux = self.agn_branch(recon_features)
        host_frac_for_recon = self.host_frac_head(recon_features) if detach_reconstruction_encoder else pred_host_frac

        # 3. 🚀 物理显式混合 (Physical Linear Mixing)
        stellar_part = host_frac_for_recon * stellar_base
        agn_part = (1.0 - host_frac_for_recon) * agn_base
        recon_spec = stellar_part + agn_part

        return pred_sigma, pred_logvar, pred_sfh, pred_host_frac, recon_spec, agn_part, stellar_part


def weighted_reconstruction_loss(obs_spec, recon_spec, ivar=None, fit_mask=None, beta=0.05):
    flux, inferred_ivar, inferred_mask = split_spectrum_channels(obs_spec)
    if ivar is None:
        ivar = inferred_ivar
    if fit_mask is None:
        fit_mask = inferred_mask

    per_pixel = F.smooth_l1_loss(recon_spec, flux, reduction='none', beta=beta)
    weights = torch.ones_like(per_pixel)
    if ivar is not None:
        valid_ivar = (ivar > 0).float()
        ivar_scale = (ivar * valid_ivar).sum(dim=1, keepdim=True) / valid_ivar.sum(dim=1, keepdim=True).clamp_min(1.0)
        ivar_scale = ivar_scale.clamp_min(1e-6)
        weights = weights * (ivar / ivar_scale).clamp(0.0, 10.0)
    if fit_mask is not None:
        weights = weights * fit_mask.clamp(0.0, 1.0)
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


def sigma_window_reconstruction_loss(obs_spec, recon_spec, beta=0.03):
    if obs_spec.dim() != 3 or obs_spec.size(1) < 2:
        return recon_spec.new_tensor(0.0)
    flux = obs_spec[:, 0, :]
    sigma_mask = obs_spec[:, -1, :].clamp(0.0, 1.0)
    ivar = obs_spec[:, 1, :].clamp_min(0.0) if obs_spec.size(1) > 1 else torch.ones_like(flux)
    per_pixel = F.smooth_l1_loss(recon_spec, flux, reduction='none', beta=beta)
    valid_ivar = (ivar > 0).float()
    ivar_scale = (ivar * valid_ivar).sum(dim=1, keepdim=True) / valid_ivar.sum(dim=1, keepdim=True).clamp_min(1.0)
    weights = sigma_mask * (ivar / ivar_scale.clamp_min(1e-6)).clamp(0.0, 10.0)
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


def sigma_window_derivative_loss(obs_spec, recon_spec, beta=0.03):
    if obs_spec.dim() != 3 or obs_spec.size(-1) < 3:
        return recon_spec.new_tensor(0.0)
    flux = obs_spec[:, 0, :]
    sigma_mask = obs_spec[:, -1, :].clamp(0.0, 1.0)
    d_flux = flux[:, 1:] - flux[:, :-1]
    d_recon = recon_spec[:, 1:] - recon_spec[:, :-1]
    pair_mask = sigma_mask[:, 1:] * sigma_mask[:, :-1]
    per_pixel = F.smooth_l1_loss(d_recon, d_flux, reduction='none', beta=beta)
    return (per_pixel * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def sigma_window_derivative_correlation_loss(obs_spec, recon_spec):
    if obs_spec.dim() != 3 or obs_spec.size(-1) < 3:
        return recon_spec.new_tensor(0.0)
    flux = obs_spec[:, 0, :]
    sigma_mask = obs_spec[:, -1, :].clamp(0.0, 1.0)
    d_flux = flux[:, 1:] - flux[:, :-1]
    d_recon = recon_spec[:, 1:] - recon_spec[:, :-1]
    pair_mask = sigma_mask[:, 1:] * sigma_mask[:, :-1]
    weight_sum = pair_mask.sum(dim=1).clamp_min(1.0)
    flux_mean = (d_flux * pair_mask).sum(dim=1, keepdim=True) / weight_sum.view(-1, 1)
    recon_mean = (d_recon * pair_mask).sum(dim=1, keepdim=True) / weight_sum.view(-1, 1)
    f = (d_flux - flux_mean) * pair_mask
    r = (d_recon - recon_mean) * pair_mask
    cov = (f * r).sum(dim=1) / weight_sum
    f_std = torch.sqrt(((f * f).sum(dim=1) / weight_sum).clamp_min(1e-8))
    r_std = torch.sqrt(((r * r).sum(dim=1) / weight_sum).clamp_min(1e-8))
    corr = torch.nan_to_num(cov / (f_std * r_std), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    valid = (pair_mask.sum(dim=1) > 8).float()
    return ((1.0 - corr) * valid).sum() / valid.sum().clamp_min(1.0)


def host_sigma_window_flux_fraction(stellar_part, recon_spec, obs_spec):
    if obs_spec.dim() != 3:
        return stellar_part.new_zeros(stellar_part.size(0))
    sigma_mask = obs_spec[:, -1, :].clamp(0.0, 1.0)
    host_power = (stellar_part.abs() * sigma_mask).sum(dim=1)
    total_power = (recon_spec.abs() * sigma_mask).sum(dim=1).clamp_min(1e-6)
    return (host_power / total_power).clamp(0.0, 2.0)


def sfh_smoothness_loss(sfh_weights, n_z=10, n_age=20):
    if sfh_weights.size(1) != n_z * n_age:
        return sfh_weights.new_tensor(0.0)
    sfh_grid = sfh_weights.view(-1, n_z, n_age)
    diff2_age = sfh_grid[:, :, 2:] - 2 * sfh_grid[:, :, 1:-1] + sfh_grid[:, :, :-2]
    diff2_z = sfh_grid[:, 2:, :] - 2 * sfh_grid[:, 1:-1, :] + sfh_grid[:, :-2, :]
    return torch.mean(diff2_age ** 2) + torch.mean(diff2_z ** 2)


def sigma_distribution_loss(pred_sigma, true_sigma_kms, sigma_scale_kms=120.0):
    pred = pred_sigma.squeeze()
    true = true_sigma_kms.squeeze()
    if pred.numel() < 3:
        return pred.new_tensor(0.0)

    pred_centered = pred - pred.mean()
    true_centered = true - true.mean()
    pred_std = torch.sqrt(torch.mean(pred_centered**2).clamp_min(1e-8))
    true_std = torch.sqrt(torch.mean(true_centered**2).clamp_min(1e-8))

    mean_loss = F.smooth_l1_loss(
        (pred.mean() - true.mean()).view(1) / sigma_scale_kms,
        pred.new_zeros(1),
        beta=0.5,
    )
    std_loss = F.smooth_l1_loss(
        ((pred_std - true_std) / sigma_scale_kms).view(1),
        pred.new_zeros(1),
        beta=0.5,
    )
    corr = torch.mean(pred_centered * true_centered) / (pred_std * true_std)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    corr_loss = 1.0 - corr
    return 0.2 * mean_loss + 0.8 * std_loss + 0.5 * corr_loss


def physics_hybrid_loss(
    obs_spec,
    recon_spec,
    pred_sigma,
    pred_logvar,
    true_sigma_kms,
    sfh_weights,
    pred_host,
    true_host,
    ivar=None,
    fit_mask=None,
    true_sfh=None,
    sigma_sample_weight=None,
    n_z=10,
    n_age=20,
    sigma_scale_kms=120.0,
    sigma_weight=20.0,
    sigma_distribution_weight=0.5,
    recon_weight=25.0,
    host_weight=15.0,
    smooth_weight=2.0,
    sfh_weight=0.5,
    uncertainty_weight=0.05,
    sigma_recon_weight=0.0,
    sigma_deriv_weight=0.0,
    sigma_deriv_corr_weight=0.0,
    host_sigma_fraction_weight=0.0,
    stellar_part=None,
):
    sigma_residual = (pred_sigma.squeeze() - true_sigma_kms.squeeze()) / sigma_scale_kms
    per_sample_sigma_loss = F.smooth_l1_loss(
        sigma_residual,
        torch.zeros_like(sigma_residual),
        beta=0.5,
        reduction='none',
    )
    if sigma_sample_weight is not None:
        sigma_sample_weight = sigma_sample_weight.to(per_sample_sigma_loss.device).squeeze().clamp_min(0.0)
        param_loss = (per_sample_sigma_loss * sigma_sample_weight).sum() / sigma_sample_weight.sum().clamp_min(1e-6)
    else:
        param_loss = per_sample_sigma_loss.mean()
    if sigma_distribution_weight > 0:
        param_loss = param_loss + sigma_distribution_weight * sigma_distribution_loss(
            pred_sigma,
            true_sigma_kms,
            sigma_scale_kms=sigma_scale_kms,
        )

    zero = recon_spec.new_tensor(0.0)
    uncertainty_loss = zero
    if uncertainty_weight > 0:
        logvar = torch.clamp(pred_logvar.squeeze(), min=-6.0, max=4.0)
        per_sample_uncertainty_loss = 0.5 * torch.exp(-logvar) * sigma_residual**2 + 0.5 * logvar
        per_sample_uncertainty_loss = per_sample_uncertainty_loss + 0.01 * logvar**2
        if sigma_sample_weight is not None:
            uncertainty_loss = (
                per_sample_uncertainty_loss * sigma_sample_weight
            ).sum() / sigma_sample_weight.sum().clamp_min(1e-6)
        else:
            uncertainty_loss = per_sample_uncertainty_loss.mean()

    host_loss = F.mse_loss(pred_host.squeeze(), true_host.squeeze()) if host_weight > 0 else zero
    recon_loss = (
        weighted_reconstruction_loss(obs_spec, recon_spec, ivar=ivar, fit_mask=fit_mask)
        if recon_weight > 0
        else zero
    )
    sigma_recon_loss = (
        sigma_window_reconstruction_loss(obs_spec, recon_spec)
        if sigma_recon_weight > 0
        else zero
    )
    sigma_deriv_loss = (
        sigma_window_derivative_loss(obs_spec, recon_spec)
        if sigma_deriv_weight > 0
        else zero
    )
    sigma_deriv_corr_loss = (
        sigma_window_derivative_correlation_loss(obs_spec, recon_spec)
        if sigma_deriv_corr_weight > 0
        else zero
    )
    host_sigma_fraction_loss = zero
    if host_sigma_fraction_weight > 0 and stellar_part is not None:
        host_sigma_fraction = host_sigma_window_flux_fraction(stellar_part, recon_spec, obs_spec)
        target_host = true_host.squeeze().clamp(0.0, 1.0)
        host_sigma_fraction_loss = F.smooth_l1_loss(host_sigma_fraction, target_host, beta=0.1)
    smoothness_penalty = sfh_smoothness_loss(sfh_weights, n_z=n_z, n_age=n_age) if smooth_weight > 0 else zero

    sfh_supervision = zero
    if sfh_weight > 0 and true_sfh is not None:
        sfh_supervision = F.mse_loss(sfh_weights, true_sfh)

    total_loss = (
        sigma_weight * param_loss
        + uncertainty_weight * uncertainty_loss
        + recon_weight * recon_loss
        + sigma_recon_weight * sigma_recon_loss
        + sigma_deriv_weight * sigma_deriv_loss
        + sigma_deriv_corr_weight * sigma_deriv_corr_loss
        + host_sigma_fraction_weight * host_sigma_fraction_loss
        + host_weight * host_loss
        + smooth_weight * smoothness_penalty
        + sfh_weight * sfh_supervision
    )
    return total_loss, param_loss, recon_loss, host_loss, smoothness_penalty

import math
from typing import Tuple

import torch


class EnhancedImageColourTransferNode:
    """
    Further enhanced adaptation of the Reinhard colour transfer method.

    Inputs:
      - source_image: torch.Tensor (B,H,W,3) float32 in [0,1]
      - target_image: torch.Tensor (B,H,W,3) float32 in [0,1]

    Internal processing uses float32 Torch tensors. Colour space here is the
    Reinhard Lαβ (derived from log-LMS), NOT CIELAB.
    """

    # --- Constant transforms (class-level to avoid reallocation) ---
    RGB2LMS = torch.tensor(
        [[0.3811, 0.5783, 0.0402],
         [0.1967, 0.7244, 0.0782],
         [0.0241, 0.1288, 0.8444]],
        dtype=torch.float32
    )
    LMS2LAB = torch.tensor(
        [[1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)],
         [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)],
         [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0]],
        dtype=torch.float32
    )
    INV_LMS2LAB = torch.linalg.inv(LMS2LAB)
    INV_RGB2LMS = torch.linalg.inv(RGB2LMS)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "cross_covariance_limit": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reshaping_iteration": ("INT", {"default": 1, "min": 1, "max": 10}),
                "modified_val": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.01}),
                "extra_shading": ("BOOLEAN", {"default": True}),
                "shader_val": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.01}),
                "scale_vs_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tint_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01}),
                "saturation_val": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    FUNCTION = "transfer"
    CATEGORY = "image/preprocessors"
    OUTPUT_IS_LIST = (False,)

    # ------------------------- Public entry -------------------------

    @torch.no_grad()
    def transfer(
            self,
            source_image: torch.Tensor,
            target_image: torch.Tensor,
            cross_covariance_limit: float,
            reshaping_iteration: int,
            modified_val: float,
            extra_shading: bool,
            shader_val: float,
            scale_vs_clip: float,
            tint_val: float,
            saturation_val: float
    ) -> Tuple[torch.Tensor]:

        # Ensure shapes: (B,H,W,3) float32 in [0,1]
        assert source_image.ndim == 4 and source_image.shape[-1] == 3, "source_image must be (B,H,W,3)"
        assert target_image.ndim == 4 and target_image.shape[-1] == 3, "target_image must be (B,H,W,3)"
        device = target_image.device
        self.__move_constants(device)

        # Work in 0..255 domain internally to match original algorithm
        src255 = (source_image.clamp(0, 1) * 255.0).to(torch.float32)
        tgt255 = (target_image.clamp(0, 1) * 255.0).to(torch.float32)

        core = self.__core_processing(tgt255, src255,
                                      cross_covariance_limit=cross_covariance_limit,
                                      reshaping_iteration=reshaping_iteration,
                                      shader_val=shader_val)
        res = self.__adjust_saturation(core, tgt255, saturation_val=saturation_val)
        shaded = self.__full_shading(res, tgt255, src255,
                                     extra_shading=extra_shading, shader_val=shader_val)
        final = self.__final_adjustment(shaded, tgt255,
                                        tint_val=tint_val, modified_val=modified_val,
                                        scale_vs_clip=scale_vs_clip)

        # Back to 0..1 float tensor
        out = (final.clamp(0, 255) / 255.0).to(target_image.dtype)
        return (out,)

    # ----------------------- Color conversions ----------------------

    def __rgb_to_lab(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B,H,W,3) in 0..255
        img_lms = self.__transform(img, self.RGB2LMS)  # 0..255 -> LMS
        img_lms = torch.clamp(img_lms, min=1.0)  # avoid log(0)
        img_lms_log10 = torch.log10(img_lms)  # log10
        img_lab = self.__transform(img_lms_log10, self.LMS2LAB)
        return img_lab

    def __lab_to_rgb(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B,H,W,3) in Lαβ
        lms_log10 = self.__transform(img, self.INV_LMS2LAB)
        lms = torch.pow(10.0, lms_log10)  # inverse log10
        rgb = self.__transform(lms, self.INV_RGB2LMS)
        return rgb

    # -------------------- Core transfer pipeline --------------------

    def __core_processing(
            self,
            tgt: torch.Tensor,
            src: torch.Tensor,
            cross_covariance_limit: float = 0.5,
            reshaping_iteration: int = 1,
            shader_val: float = 0.5
    ) -> torch.Tensor:
        # Convert to Lαβ
        tgtf = self.__rgb_to_lab(tgt)
        srcf = self.__rgb_to_lab(src)

        # Per-image, per-channel mean/std over H,W
        tgt_mean, tgt_std = self.__mean_stddev(tgtf)
        src_mean, src_std = self.__mean_stddev(srcf)

        # Normalize
        t_lab = (tgtf - tgt_mean) / (tgt_std + 1e-6)
        s_lab = (srcf - src_mean) / (src_std + 1e-6)

        # Iterative channel conditioning on α,β (index 1,2)
        for _ in range(max(0, reshaping_iteration - (reshaping_iteration + 1) // 2)):
            t_lab[..., 1:] = self.__channel_conditioning(t_lab[..., 1:], s_lab[..., 1:])

        t_lab = self.__adjust_covariance(t_lab, s_lab, cross_covariance_limit)

        for _ in range((reshaping_iteration + 1) // 2):
            t_lab[..., 1:] = self.__channel_conditioning(t_lab[..., 1:], s_lab[..., 1:])

        # Mix luminance stats between source and target using shader_val
        src_mean_adj = src_mean.clone()
        src_std_adj = src_std.clone()
        src_mean_adj[..., 0:1] = shader_val * src_mean[..., 0:1] + (1.0 - shader_val) * tgt_mean[..., 0:1]
        src_std_adj[..., 0:1] = shader_val * src_std[..., 0:1] + (1.0 - shader_val) * tgt_std[..., 0:1]

        # Re-scale to source stats
        t_lab = t_lab * src_std_adj + src_mean_adj

        res_rgb = self.__lab_to_rgb(t_lab)
        return res_rgb

    # ----------------------- Helper operations ----------------------

    @staticmethod
    def __binary_threshold(src: torch.Tensor, thresh: float, maxval: float) -> Tuple[float, torch.Tensor]:
        dst = torch.where(src > thresh, torch.tensor(maxval, dtype=src.dtype, device=src.device),
                          torch.tensor(0.0, dtype=src.dtype, device=src.device))
        return float(thresh), dst

    def __channel_conditioning(self, t_channel: torch.Tensor, s_channel: torch.Tensor) -> torch.Tensor:
        # t_channel, s_channel: (B,H,W,2) for α,β
        wval = 0.25

        # --- Source stats for U (s > 0) and L (s <= 0) ---
        _, mask = self.__binary_threshold(s_channel, 0.0, 1.0)
        mask = mask.to(dtype=torch.float32)
        inv_mask = 1.0 - mask

        def weighted_fourth_power_mean(x, m):
            # x: (B,H,W,2), m: 0/1 mask same shape
            # compute means per-batch, per-channel over H,W
            # First simple mean for scale in weights
            denom = torch.clamp(m.sum(dim=(1, 2), keepdim=True), min=1.0)
            mean1 = (x * m).sum(dim=(1, 2), keepdim=True) / denom
            w = torch.exp(-x * (wval / (mean1 + 1e-6)))
            w = (1.0 - w) * (1.0 - w)
            w_mean = (w * m).sum(dim=(1, 2), keepdim=True) / torch.clamp(denom, min=1.0)
            x4 = torch.pow(x, 4.0)
            mean4 = (x4 * w * m).sum(dim=(1, 2), keepdim=True) / (torch.clamp(denom, min=1.0) * (w_mean + 1e-6))
            return mean4, w

        s_mean_u, w_u_s = weighted_fourth_power_mean(s_channel, mask)
        s_mean_l, w_l_s = weighted_fourth_power_mean(s_channel, inv_mask)

        # --- Target stats (t_channel) ---
        _, tmask = self.__binary_threshold(t_channel, 0.0, 1.0)
        tmask = tmask.to(dtype=torch.float32)
        tinv = 1.0 - tmask

        t_mean_u, w_u_t = weighted_fourth_power_mean(t_channel, tmask)
        t_mean_l, w_l_t = weighted_fourth_power_mean(t_channel, tinv)

        # Re-shape with k factors
        k_u = torch.sqrt(torch.sqrt((s_mean_u + 1e-6) / (t_mean_u + 1e-6)))
        k_l = torch.sqrt(torch.sqrt((s_mean_l + 1e-6) / (t_mean_l + 1e-6)))

        t_channel_u = (1.0 + w_u_t * (k_u - 1.0)) * t_channel
        t_channel_l = (1.0 + w_l_t * (k_l - 1.0)) * t_channel

        out = t_channel_u * tmask + t_channel_l * tinv

        # Normalize to zero-mean, unit-std over H,W per-batch per-channel
        mean = out.mean(dim=(1, 2), keepdim=True)
        std = out.std(dim=(1, 2), keepdim=True) + 1e-6
        out = (out - mean) / std
        return out

    @staticmethod
    def __adjust_covariance(t_lab: torch.Tensor, s_lab: torch.Tensor,
                            cross_covariance_limit: float = 0.5) -> torch.Tensor:
        out = t_lab.clone()
        if cross_covariance_limit != 0.0:
            tcross = (out[..., 1] * out[..., 2]).mean(dim=(1, 2), keepdim=True)
            scross = (s_lab[..., 1] * s_lab[..., 2]).mean(dim=(1, 2), keepdim=True)

            w1 = 0.5 * torch.sqrt((1 + scross) / (1 + tcross) + 1e-6) + \
                 0.5 * torch.sqrt((1 - scross) / (1 - tcross) + 1e-6)
            w2 = 0.5 * torch.sqrt((1 + scross) / (1 + tcross) + 1e-6) - \
                 0.5 * torch.sqrt((1 - scross) / (1 - tcross) + 1e-6)

            cond = torch.abs(w2) > (abs(cross_covariance_limit) * torch.abs(w1))
            if cond.any():
                w2 = torch.where(cond, torch.sign(w2) * (abs(cross_covariance_limit) * w1), w2)
                norm = 1.0 / torch.sqrt(w1 ** 2 + w2 ** 2 + 2.0 * w1 * w2 * tcross + 1e-8)
                w1 = w1 * norm
                w2 = w2 * norm

            z1 = out[..., 1].clone()
            out[..., 1] = w1 * z1 + w2 * out[..., 2]
            out[..., 2] = w1 * out[..., 2] + w2 * z1
        return out

    @staticmethod
    def __rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
        # img: (B,H,W,3)
        weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=img.dtype, device=img.device)
        return (img * weights.view(1, 1, 1, 3)).sum(dim=-1)

    def __adjust_saturation(self, img: torch.Tensor, origin_img: torch.Tensor,
                            saturation_val: float = -1.0) -> torch.Tensor:
        # Convert to HSV
        img_hsv = self.__rgb2hsv(img)
        ori_hsv = self.__rgb2hsv(origin_img)

        if saturation_val < 0:
            amax1 = img_hsv[..., 1].amax(dim=(1, 2), keepdim=True) + 1e-6
            amax2 = ori_hsv[..., 1].amax(dim=(1, 2), keepdim=True)
            saturation_val = (amax2 / amax1).clamp(min=0.0, max=10.0)  # broadcastable tensor
        else:
            saturation_val = torch.tensor(saturation_val, dtype=img.dtype, device=img.device)

        if not torch.allclose(saturation_val, torch.ones_like(img_hsv[..., 1]).mean() * 1.0):
            ori_hsv[..., 1] = img_hsv[..., 1] * saturation_val + ori_hsv[..., 1] * (1.0 - saturation_val)

            _, mask = self.__binary_threshold(img_hsv[..., 1] - ori_hsv[..., 1], 0.0, 1.0)
            mask = mask.to(dtype=torch.float32)

            ori_hsv[..., 1] = ori_hsv[..., 1] * mask + img_hsv[..., 1] * (1.0 - mask)

            tmean = img_hsv[..., 1].mean(dim=(1, 2), keepdim=True)
            tdev = img_hsv[..., 1].std(dim=(1, 2), keepdim=True) + 1e-6
            tmpmean = ori_hsv[..., 1].mean(dim=(1, 2), keepdim=True)
            tmpdev = ori_hsv[..., 1].std(dim=(1, 2), keepdim=True) + 1e-6

            img_hsv[..., 1] = (img_hsv[..., 1] - tmean) / tdev * tmpdev + tmpmean
        else:
            # keep img_hsv saturation
            pass

        img_rgb = self.__hsv2rgb(img_hsv)
        return img_rgb

    def __full_shading(self, img: torch.Tensor, ori_img: torch.Tensor, src_img: torch.Tensor,
                       extra_shading: bool = True, shader_val: float = 0.5) -> torch.Tensor:
        if extra_shading:
            greyt = self.__rgb_to_gray(ori_img)  # (B,H,W)
            greyp = self.__rgb_to_gray(img)
            greys = self.__rgb_to_gray(src_img)

            smean = greys.mean(dim=(1, 2), keepdim=True)
            sdev = greys.std(dim=(1, 2), keepdim=True) + 1e-6
            tmean = greyt.mean(dim=(1, 2), keepdim=True)
            tdev = greyt.std(dim=(1, 2), keepdim=True) + 1e-6

            greyt_n = (greyt - tmean) / tdev
            greyt_r = greyt_n * (shader_val * sdev + (1.0 - shader_val) * tdev) + shader_val * smean + (
                        1.0 - shader_val) * tmean

            greyp = torch.clamp(greyp, min=1.0)
            greyt_r = torch.clamp(greyt_r, min=0.0)

            img = img / greyp.unsqueeze(-1) * greyt_r.unsqueeze(-1)
        return img

    @staticmethod
    def __normalize_array(arr: torch.Tensor, desired_min: float, desired_max: float) -> torch.Tensor:
        arr_min = arr.amin(dim=(1, 2, 3), keepdim=True)
        arr_max = arr.amax(dim=(1, 2, 3), keepdim=True)
        scale = (arr - arr_min) / (arr_max - arr_min + 1e-6) * (desired_max - desired_min) + desired_min
        return scale

    def __final_adjustment(self, img: torch.Tensor, ori_img: torch.Tensor, tint_val: float = 1.0,
                           modified_val: float = 1.0, scale_vs_clip: float = 1.0) -> torch.Tensor:
        if tint_val != 1.0:
            grey = self.__rgb_to_gray(img).unsqueeze(-1)
            img = tint_val * img + (1.0 - tint_val) * grey

        if modified_val != 1.0:
            img = modified_val * img + (1.0 - modified_val) * ori_img

        scale = self.__normalize_array(img, 0.0, 255.0)
        clip = img.clamp(0.0, 255.0)
        img = scale * (1.0 - scale_vs_clip) + clip * scale_vs_clip
        img = img.clamp(0.0, 255.0)
        return img

    # -------------------- generic math helpers ---------------------

    @staticmethod
    def __transform(src: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # src: (B,H,W,3); m: (3,3)
        return torch.matmul(src, m.t())

    @staticmethod
    def __mean_stddev(src: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # src: (B,H,W,3)
        mean = src.mean(dim=(1, 2), keepdim=True)  # (B,1,1,3)
        std = src.std(dim=(1, 2), keepdim=True)  # (B,1,1,3)
        return mean, std + 1e-6

    # ----------------------- HSV converters ------------------------

    @staticmethod
    def __rgb2hsv(src: torch.Tensor) -> torch.Tensor:
        # src: (B,H,W,3) in 0..255
        x = (src / 255.0).clamp(0.0, 1.0)
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        cmax = torch.max(x, dim=-1).values
        cmin = torch.min(x, dim=-1).values
        delta = cmax - cmin

        hue = torch.zeros_like(cmax)
        mask = delta > 1e-6
        # Cases
        r_is_max = (cmax == r) & mask
        g_is_max = (cmax == g) & mask
        b_is_max = (cmax == b) & mask

        hue[r_is_max] = ((g - b)[r_is_max] / delta[r_is_max]) % 6.0
        hue[g_is_max] = ((b - r)[g_is_max] / delta[g_is_max]) + 2.0
        hue[b_is_max] = ((r - g)[b_is_max] / delta[b_is_max]) + 4.0
        hue = hue * 30.0  # [0,180] like OpenCV

        sat = torch.zeros_like(cmax)
        nonzero = cmax > 0
        sat[nonzero] = (delta[nonzero] / cmax[nonzero])
        sat = sat * 255.0

        val = cmax * 255.0
        return torch.stack([hue, sat, val], dim=-1)

    @staticmethod
    def __hsv2rgb(src: torch.Tensor) -> torch.Tensor:
        # src: (B,H,W,3) with H in [0,180], S,V in [0,255]
        h = src[..., 0] * 2.0
        s = src[..., 1] / 255.0
        v = src[..., 2] / 255.0

        c = v * s
        h_ = h / 60.0
        x = c * (1.0 - torch.abs(h_ % 2 - 1.0))
        m = v - c

        zeros = torch.zeros_like(h)
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        conds = [
            (h >= 0) & (h < 60),
            (h >= 60) & (h < 120),
            (h >= 120) & (h < 180),
            (h >= 180) & (h < 240),
            (h >= 240) & (h < 300),
            (h >= 300) & (h < 360),
        ]
        rgbs = [
            (c, x, zeros),
            (x, c, zeros),
            (zeros, c, x),
            (zeros, x, c),
            (x, zeros, c),
            (c, zeros, x),
        ]
        for cond, (rc, gc, bc) in zip(conds, rgbs):
            r = torch.where(cond, rc, r)
            g = torch.where(cond, gc, g)
            b = torch.where(cond, bc, b)

        rgb = torch.stack([r + m, g + m, b + m], dim=-1) * 255.0
        return rgb

    # ------------------- utility -------------------
    def __move_constants(self, device: torch.device):
        # move class-level constant tensors to the right device lazily
        self.RGB2LMS = self.RGB2LMS.to(device)
        self.LMS2LAB = self.LMS2LAB.to(device)
        self.INV_LMS2LAB = self.INV_LMS2LAB.to(device)
        self.INV_RGB2LMS = self.INV_RGB2LMS.to(device)

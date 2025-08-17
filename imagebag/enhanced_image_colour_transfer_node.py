import math
import torch
from typing import Tuple


class EnhancedImageColourTransferNode:
    """
    Further enhanced adaptation of the Reinhard colour transfer method.

    Inputs:
      - source_image: torch.Tensor (B,H,W,3) float32 in [0,1]
      - target_image: torch.Tensor (B,H,W,3) float32 in [0,1]

    Internal processing uses float32 Torch tensors. Colour space here is the
    Reinhard Lαβ (derived from log-LMS), NOT CIELAB.
    """

    # --- Constant transforms (cached per device/dtype) ---
    _RGB2LMS_base = torch.tensor(
        [[0.3811, 0.5783, 0.0402],
         [0.1967, 0.7244, 0.0782],
         [0.0241, 0.1288, 0.8444]], dtype=torch.float32
    )
    _LMS2LAB_base = torch.tensor(
        [[1.0 / math.sqrt(3.0),  1.0 / math.sqrt(3.0),  1.0 / math.sqrt(3.0)],
         [1.0 / math.sqrt(6.0),  1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)],
         [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0),  0.0]], dtype=torch.float32
    )
    _INV_LMS2LAB_base = torch.linalg.inv(_LMS2LAB_base)
    _INV_RGB2LMS_base = torch.linalg.inv(_RGB2LMS_base)
    _gray_weights_base = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)

    # caches
    _cached_device = None
    _RGB2LMS = None
    _LMS2LAB = None
    _INV_LMS2LAB = None
    _INV_RGB2LMS = None
    _gray_weights = None

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
                "modified_val": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extra_shading": ("BOOLEAN", {"default": True}),
                "shader_val": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_vs_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tint_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "saturation_val": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01})
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

        assert source_image.ndim == 4 and source_image.shape[-1] == 3
        assert target_image.ndim == 4 and target_image.shape[-1] == 3
        device = target_image.device
        self.__move_constants(device, target_image.dtype)

        # 0..1 → 0..255 float
        src255 = (source_image.clamp(0, 1) * 255.0).to(torch.float32).contiguous(memory_format=torch.channels_last)
        tgt255 = (target_image.clamp(0, 1) * 255.0).to(torch.float32).contiguous(memory_format=torch.channels_last)

        core = self.__core_processing(tgt255, src255,
                                      cross_covariance_limit,
                                      reshaping_iteration,
                                      shader_val)
        res = self.__adjust_saturation(core, tgt255, saturation_val)
        shaded = self.__full_shading(res, tgt255, src255, extra_shading, shader_val)
        final = self.__final_adjustment(shaded, tgt255, tint_val, modified_val, scale_vs_clip)

        out = (final.clamp(0, 255) / 255.0).to(target_image.dtype)
        return (out,)

    # ----------------------- Color conversions ----------------------

    def __rgb_to_lab(self, img: torch.Tensor) -> torch.Tensor:
        lms = torch.matmul(img, self._RGB2LMS.T).clamp_min(1.0)
        return torch.matmul(torch.log10(lms), self._LMS2LAB.T)

    def __lab_to_rgb(self, img: torch.Tensor) -> torch.Tensor:
        lms = torch.pow(10.0, torch.matmul(img, self._INV_LMS2LAB.T))
        return torch.matmul(lms, self._INV_RGB2LMS.T)

    # -------------------- Core transfer pipeline --------------------

    def __core_processing(self, tgt, src, limit, iters, shader_val):
        tgtf, srcf = self.__rgb_to_lab(tgt), self.__rgb_to_lab(src)
        tgt_mean, tgt_std = self.__mean_stddev(tgtf)
        src_mean, src_std = self.__mean_stddev(srcf)

        t_lab = (tgtf - tgt_mean) / tgt_std
        s_lab = (srcf - src_mean) / src_std

        half = (iters + 1) // 2
        for _ in range(max(0, iters - half)):
            t_lab[..., 1:] = self.__channel_conditioning(t_lab[..., 1:], s_lab[..., 1:])
        t_lab = self.__adjust_covariance(t_lab, s_lab, limit)
        for _ in range(half):
            t_lab[..., 1:] = self.__channel_conditioning(t_lab[..., 1:], s_lab[..., 1:])

        # blend luminance stats
        src_mean[..., 0:1] = shader_val * src_mean[..., 0:1] + (1 - shader_val) * tgt_mean[..., 0:1]
        src_std[..., 0:1] = shader_val * src_std[..., 0:1] + (1 - shader_val) * tgt_std[..., 0:1]

        return self.__lab_to_rgb(t_lab * src_std + src_mean)

    # ----------------------- Helper operations ----------------------

    def __channel_conditioning(self, t, s):
        wval = 0.25

        def mean4(x, m):
            denom = m.sum((1, 2), keepdim=True).clamp_min(1.0)
            mean1 = (x * m).sum((1, 2), keepdim=True) / denom
            w = (1 - torch.exp(-x * wval / (mean1 + 1e-6))) ** 2
            w_mean = (w * m).sum((1, 2), keepdim=True) / denom
            return ((x ** 4) * w * m).sum((1, 2), keepdim=True) / (denom * (w_mean + 1e-6)), w

        mask = (s > 0).float()
        sU, _ = mean4(s, mask)
        sL, _ = mean4(s, 1 - mask)

        maskt = (t > 0).float()
        tU, wU = mean4(t, maskt)
        tL, wL = mean4(t, 1 - maskt)

        kU = ((sU + 1e-6) / (tU + 1e-6)).sqrt().sqrt()
        kL = ((sL + 1e-6) / (tL + 1e-6)).sqrt().sqrt()

        out = (1 + wU * (kU - 1)) * t * maskt + (1 + wL * (kL - 1)) * t * (1 - maskt)
        mean = out.mean((1, 2), keepdim=True)
        std = out.std((1, 2), keepdim=True) + 1e-6
        return (out - mean) / std

    def __adjust_covariance(self, t, s, limit):
        if limit == 0:
            return t
        tc = (t[..., 1] * t[..., 2]).mean((1, 2), keepdim=True)
        sc = (s[..., 1] * s[..., 2]).mean((1, 2), keepdim=True)
        W1 = 0.5 * ((1 + sc) / (1 + tc)).sqrt() + 0.5 * ((1 - sc) / (1 - tc)).sqrt()
        W2 = 0.5 * ((1 + sc) / (1 + tc)).sqrt() - 0.5 * ((1 - sc) / (1 - tc)).sqrt()
        cond = W2.abs() > limit * W1.abs()
        if cond.any():
            W2 = torch.where(cond, W2.sign() * limit * W1, W2)
            norm = 1.0 / ((W1 ** 2 + W2 ** 2 + 2 * W1 * W2 * tc).sqrt() + 1e-8)
            W1 *= norm
            W2 *= norm
        z1 = t[..., 1].clone()
        t[..., 1] = W1 * z1 + W2 * t[..., 2]
        t[..., 2] = W1 * t[..., 2] + W2 * z1
        return t

    def __adjust_saturation(self, img, ori, val):
        hsv = self.__rgb2hsv(img)
        ori_hsv = self.__rgb2hsv(ori)
        if val < 0:
            val = (ori_hsv[..., 1].amax((1, 2), keepdim=True) /
                   (hsv[..., 1].amax((1, 2), keepdim=True) + 1e-6)).clamp(0, 10)
        else:
            val = torch.tensor(val, dtype=img.dtype, device=img.device)
        if not torch.allclose(val, torch.ones_like(val)):
            ori_hsv[..., 1] = hsv[..., 1] * val + ori_hsv[..., 1] * (1 - val)
            mask = (hsv[..., 1] - ori_hsv[..., 1] > 0).float()
            ori_hsv[..., 1] = ori_hsv[..., 1] * mask + hsv[..., 1] * (1 - mask)
            tm, td = hsv[..., 1].mean((1, 2), keepdim=True), hsv[..., 1].std((1, 2), keepdim=True) + 1e-6
            om, od = ori_hsv[..., 1].mean((1, 2), keepdim=True), ori_hsv[..., 1].std((1, 2), keepdim=True) + 1e-6
            hsv[..., 1] = (hsv[..., 1] - tm) / td * od + om
        return self.__hsv2rgb(hsv)

    def __full_shading(self, img, ori, src, extra=True, shader_val=0.5):
        if not extra:
            return img
        gt, gp, gs = self.__rgb_to_gray(ori), self.__rgb_to_gray(img), self.__rgb_to_gray(src)
        sm, sd = gs.mean((1, 2), keepdim=True), gs.std((1, 2), keepdim=True) + 1e-6
        tm, td = gt.mean((1, 2), keepdim=True), gt.std((1, 2), keepdim=True) + 1e-6
        gtr = ((gt - tm) / td) * (shader_val * sd + (1 - shader_val) * td) + shader_val * sm + (1 - shader_val) * tm
        return img / gp.clamp_min(1).unsqueeze(-1) * gtr.clamp_min(0).unsqueeze(-1)

    def __final_adjustment(self, img, ori, tint, mod, svc):
        if tint != 1:
            img = tint * img + (1 - tint) * self.__rgb_to_gray(img).unsqueeze(-1)
        if mod != 1:
            img = mod * img + (1 - mod) * ori
        scale = self.__normalize_array(img, 0, 255)
        clip = img.clamp(0, 255)
        return scale * (1 - svc) + clip * svc

    # -------------------- math helpers ---------------------

    def __rgb_to_gray(self, img): 
        return (img * self._gray_weights.view(1, 1, 1, 3)).sum(-1)

    @staticmethod
    def __normalize_array(arr, amin, amax):
        a0, a1 = arr.amin((1, 2, 3), keepdim=True), arr.amax((1, 2, 3), keepdim=True)
        return (arr - a0) / (a1 - a0 + 1e-6) * (amax - amin) + amin

    @staticmethod
    def __mean_stddev(src, mask=None):
        return src.mean((1, 2), keepdim=True), src.std((1, 2), keepdim=True) + 1e-6

    # ---------------- HSV converters ----------------------

    def __rgb2hsv(self, x):
        x = (x / 255).clamp(0, 1)
        r, g, b = x.unbind(-1)
        cmax, cmin = x.max(-1).values, x.min(-1).values
        d = cmax - cmin
        h = torch.zeros_like(cmax)
        m = d > 1e-6
        idx = (cmax == r) & m
        h[idx] = ((g - b)[idx] / d[idx]) % 6
        idx = (cmax == g) & m
        h[idx] = ((b - r)[idx] / d[idx]) + 2
        idx = (cmax == b) & m
        h[idx] = ((r - g)[idx] / d[idx]) + 4
        h = h * 30
        s = torch.where(cmax > 0, d / cmax, torch.zeros_like(cmax)) * 255
        v = cmax * 255
        return torch.stack([h, s, v], -1)

    def __hsv2rgb(self, x):
        h, s, v = x.unbind(-1)
        h = h * 2
        s, v = s / 255, v / 255
        c, hp = v * s, h / 60
        x_ = c * (1 - (hp % 2 - 1).abs())
        m = v - c
        z = torch.zeros_like(h)
        r = g = b = z.clone()
        conds = [(h < 60), (h < 120), (h < 180), (h < 240), (h < 300), (h <= 360)]
        rgbs = [(c, x_, z), (x_, c, z), (z, c, x_), (z, x_, c), (x_, z, c), (c, z, x_)]
        for cond, (rc, gc, bc) in zip(conds, rgbs):
            r = torch.where(cond, rc, r)
            g = torch.where(cond, gc, g)
            b = torch.where(cond, bc, b)
        return torch.stack([r + m, g + m, b + m], -1) * 255

    # ---------------- constants device mgmt ----------------

    def __move_constants(self, device, dtype):
        if self._cached_device == (device, dtype):
            return
        self._RGB2LMS = self._RGB2LMS_base.to(device=device, dtype=dtype)
        self._LMS2LAB = self._LMS2LAB_base.to(device=device, dtype=dtype)
        self._INV_LMS2LAB = self._INV_LMS2LAB_base.to(device=device, dtype=dtype)
        self._INV_RGB2LMS = self._INV_RGB2LMS_base.to(device=device, dtype=dtype)
        self._gray_weights = self._gray_weights_base.to(device=device, dtype=dtype)
        self._cached_device = (device, dtype)

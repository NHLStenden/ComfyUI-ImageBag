# IMPLEMENTATION OF A FURTHER ENHANCED ADAPTATION
# OF THE REINHARD COLOUR TRANSFER METHOD.
#
# Python coding by Minh Nguyen Hoang
# https://github.com/minh-nguyenhoang
# Original C++ implementation by Terry Johnson
# https://github.com/TJCoding/Enhanced-Image-Colour-Transfer-2

import numpy as np
import torch
from typing import Optional, Tuple
from numpy.typing import ArrayLike, NDArray

class EnhancedImageColourTransferNode:
    """
    Further enhanced adaptation of the Reinhard colour transfer method.

    Inputs:
      - source_image: torch.Tensor (B,H,W,3) float32 in [0,1]
      - target_image: torch.Tensor (B,H,W,3) float32 in [0,1]

    Internal processing uses float32. Colour space here is the
    Reinhard Lαβ (derived from log-LMS), NOT CIELAB.
    """

    # --- Constant transforms (class-level to avoid reallocation) ---
    RGB2LMS = np.array([[0.3811, 0.5783, 0.0402],
                        [0.1967, 0.7244, 0.0782],
                        [0.0241, 0.1288, 0.8444]]).astype(np.float32)
    LMS2LAB = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                        [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
                        [1/np.sqrt(2), -1/np.sqrt(2), 0]]).astype(np.float32)

    INV_LMS2LAB = np.linalg.inv(LMS2LAB)
    INV_RGB2LMS = np.linalg.inv(RGB2LMS)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
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

    def transfer(self, source_image: torch.Tensor, target_image: torch.Tensor, cross_covariance_limit: float, reshaping_iteration: int, modified_val: float, extra_shading: bool, shader_val: float, scale_vs_clip: float, tint_val: float, saturation_val: float) -> Tuple[torch.Tensor]:
        _src = EnhancedImageColourTransferNode.__tensor_to_rgb255(source_image)
        _tgt = EnhancedImageColourTransferNode.__tensor_to_rgb255(target_image)

        # For a full explanation of parameter choices and the associated 
        # methodology see the comments in the original C++ coding at
        # https://github.com/TJCoding/Enhanced-Image-Colour-Transfer-2

        _core = self.__core_processing(_tgt,_src, reshaping_iteration= reshaping_iteration, cross_covariance_limit= cross_covariance_limit, shader_val= shader_val)
        _res = EnhancedImageColourTransferNode.__adjust_saturation(_core, _tgt, saturation_val= saturation_val)
        _shaded = EnhancedImageColourTransferNode.__full_shading(_res, _tgt, _src, extra_shading= extra_shading, shader_val = shader_val)
        _final = EnhancedImageColourTransferNode.__final_adjustment(_shaded, _tgt, tint_val= tint_val, modified_val= modified_val, scale_vs_clip= scale_vs_clip)
        _final2 = EnhancedImageColourTransferNode.__rgb255_to_tensor(_final)
        return (_final2,)

    # ----------------------- Color conversions ----------------------

    # Here LAB refers to the L-alpha-beta colour space rather than CIELAB.
    def __rgb_to_lab(self, img: np.ndarray) -> np.ndarray:
        img = np.maximum(img, 1.)
        img_lms = EnhancedImageColourTransferNode.__transform(img, self.RGB2LMS)
        img = np.maximum(img, 1.)
        img_lms = np.log(img_lms).astype(np.float32)/np.log(10)
        img_lab = EnhancedImageColourTransferNode.__transform(img_lms, self.LMS2LAB)

        return img_lab

    def __lab_to_rgb(self, img: np.ndarray) -> np.ndarray:
        img_lms = EnhancedImageColourTransferNode.__transform(img, self.INV_LMS2LAB)
        img_lms = np.exp(img_lms).astype(np.float32)
        img_lms = np.power(img_lms, np.log(10.0))
        img_rgb = EnhancedImageColourTransferNode.__transform(img_lms, self.INV_RGB2LMS)
        return img_rgb

    # -------------------- Core transfer pipeline --------------------

    def __core_processing(self, tgt: np.ndarray, src: np.ndarray, cross_covariance_limit: float = 0.5, reshaping_iteration: int = 1, shader_val: float = 0.5 ) -> np.ndarray:
        tgtf = self.__rgb_to_lab(tgt)
        srcf = self.__rgb_to_lab(src)

        tgt_mean, tgt_std = EnhancedImageColourTransferNode.__mean_stddev(tgtf)
        src_mean, src_std = EnhancedImageColourTransferNode.__mean_stddev(srcf)

        tgt_mean, tgt_std = tgt_mean.reshape(-1).astype(np.float32), tgt_std.reshape(-1).astype(np.float32)
        src_mean, src_std = src_mean.reshape(-1).astype(np.float32), src_std.reshape(-1).astype(np.float32)

        t_lab = (tgtf - tgt_mean)/tgt_std
        s_lab = (srcf - src_mean)/src_std

        for j in range(reshaping_iteration,(reshaping_iteration + 1)//2, -1):
            t_lab[...,1:] = EnhancedImageColourTransferNode.__channel_conditioning(t_lab[...,1:], s_lab[...,1:])
        

        t_lab = EnhancedImageColourTransferNode.__adjust_covariance(t_lab, s_lab, cross_covariance_limit)

        for j in range((reshaping_iteration + 1)//2):
            t_lab[...,1:] = EnhancedImageColourTransferNode.__channel_conditioning(t_lab[...,1:], s_lab[...,1:])
        

        src_mean, src_std = src_mean.copy(), src_std.copy()
        src_mean[0] = shader_val*src_mean[0] + (1-shader_val)*tgt_mean[0]
        src_std[0] = shader_val*src_std[0] + (1-shader_val)*tgt_std[0]

        t_lab = t_lab * src_std + src_mean

        res_rgb = self.__lab_to_rgb(t_lab)
        return res_rgb

    # ----------------------- Helper operations ----------------------

    @staticmethod
    def __channel_conditioning(t_channel: np.ndarray, s_channel: np.ndarray) -> np.ndarray:
        wval = float(0.25)

        _, mask = EnhancedImageColourTransferNode.__binary_threshold(s_channel, 0, 1)
        mask = mask.astype(np.uint8)

        s_mean_U = np.sum(s_channel*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) ## mean for element above 0
        w_U = np.exp(-s_channel * wval/s_mean_U)
        w_U = (1 - w_U) * (1 - w_U) 
        w_mean = np.sum(w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) ## mean for element above 0
        channel_U = np.power(s_channel, 4)
        s_mean_U = np.sum(channel_U*w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1)/w_mean

        inv_mask = 1 - mask
        s_mean_L = np.sum(s_channel*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0

        w_L = np.exp(-s_channel * wval/s_mean_L)
        w_L = (1 - w_L) * (1 - w_L) 
        w_mean = np.sum(w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0
        channel_L = np.power(s_channel, 4)
        s_mean_L = np.sum(channel_L*w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1)/w_mean


        _, mask = EnhancedImageColourTransferNode.__binary_threshold(t_channel, 0, 1)
        mask = mask.astype(np.uint8)

        t_mean_U = np.sum(t_channel*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) ## mean for element above 0
        w_U = np.exp(-t_channel * wval/t_mean_U)
        w_U = (1 - w_U) * (1 - w_U) 
        w_mean = np.sum(w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1) + 1e-6 ## mean for element above 0
        channel_U = np.power(t_channel, 4)
        t_mean_U = np.sum(channel_U*w_U*mask, axis= (0,1))/np.maximum(np.sum(mask, axis= (0,1)), 1)/w_mean

        inv_mask = 1 - mask
        t_mean_L = np.sum(t_channel*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0
        w_L = np.exp(-t_channel * wval/t_mean_L)
        w_L = (1 - w_L) * (1 - w_L) 
        w_mean = np.sum(w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1) ## mean for element above 0
        channel_L = np.power(t_channel, 4)
        t_mean_L = np.sum(channel_L*w_L*inv_mask, axis= (0,1))/np.maximum(np.sum(inv_mask, axis= (0,1)), 1)/w_mean

        k = np.sqrt(np.sqrt(s_mean_U/t_mean_U))
        t_channel_U = (1 + w_U*(k-1))*t_channel

        k = np.sqrt(np.sqrt(s_mean_L/t_mean_L))
        t_channel_L = (1 + w_L*(k-1))*t_channel

        t_chanel_normalized: np.ndarray = t_channel_U * mask + t_channel_L * inv_mask
        t_mean, t_std = t_chanel_normalized.mean((0,1)), t_chanel_normalized.std((0,1))

        t_chanel_normalized = (t_chanel_normalized - t_mean)/t_std

        return t_chanel_normalized

    @staticmethod
    def __adjust_covariance(t_lab: np.ndarray, s_lab: np.ndarray, cross_covariance_limit: float = 0.5) -> np.ndarray:
        t_lab = t_lab.copy()
        if cross_covariance_limit != 0:
            tcrosscorr = np.mean(t_lab[...,1] * t_lab[...,2], axis= (0,1))
            scrosscorr = np.mean(s_lab[...,1] * s_lab[...,2], axis= (0,1))

            W1 = 0.5 * np.sqrt((1 + scrosscorr) / (1 + tcrosscorr)) + \
                0.5 * np.sqrt((1 - scrosscorr) / (1 - tcrosscorr))
            W2 = 0.5 * np.sqrt((1 + scrosscorr) / (1 + tcrosscorr)) - \
                0.5 * np.sqrt((1 - scrosscorr) / (1 - tcrosscorr))
            
            if abs(W2) > cross_covariance_limit * abs(W1):
                W2 = np.copysign(cross_covariance_limit * W1, W2)
                norm = 1.0 / np.sqrt(W1**2 + W2**2 + 2 * W1 * W2 * tcrosscorr)
                W1 *= norm
                W2 *= norm

            z1 = t_lab[...,1].copy()
            t_lab[...,1] = W1 * z1 + W2 * t_lab[...,2]
            t_lab[...,2] = W1 * t_lab[...,2] + W2 * z1

        return t_lab

    @staticmethod
    def __adjust_saturation(img: np.ndarray, origin_img: np.ndarray, saturation_val: float = -1) -> np.ndarray:
        # Convert images from RGB to HSV color space
        img: np.ndarray = EnhancedImageColourTransferNode.__rgb2hsv(img.astype(np.float32))
        origin_img: np.ndarray = EnhancedImageColourTransferNode.__rgb2hsv(origin_img.astype(np.float32))
        if saturation_val < 0:
            # Calculate saturation_val as the ratio of the max saturation value
            # in the original image to the max value in the processed image
            amax1 = np.max(img[...,1])
            amax2 = np.max(origin_img[...,1])
            saturation_val = amax2 / amax1

        if saturation_val != 1.:
            # Compute weighted mix of the processed target and original saturation channels
            origin_img[...,1] = img[...,1]*(saturation_val) + origin_img[...,1]*(1-saturation_val)

            # Create a mask where the processed image's saturation exceeds the original target image's saturation
            mask = EnhancedImageColourTransferNode.__binary_threshold(img[...,1] - origin_img[...,1], 0, 1)[1]
            mask = mask.astype(np.uint8)

            # Create the modified reference saturation channel
            origin_img[...,1] = origin_img[...,1] * mask + img[...,1]*(1-mask)

            # Match the mean and standard deviation of the processed image's saturation channel
            # to the modified reference saturation channel
            tmean, tdev = img[...,1].mean((0,1)), img[...,1].std((0,1))
            tmpmean, tmpdev = origin_img[...,1].mean((0,1)), origin_img[...,1].std((0,1))

            tmp_ = img[...,1].copy().astype(np.float32)
            img[...,1] = (tmp_ - tmean) / tdev * tmpdev + tmpmean
        img = EnhancedImageColourTransferNode.__hsv2rgb(img)

        return img

    @staticmethod
    def __full_shading(img: np.ndarray, ori_img: np.ndarray, src_img: np.ndarray, extra_shading: bool = True, shader_val: float = 0.5) -> np.ndarray:
        # Matches the grey shade distribution of the modified target image
        # to that of a notional shader image which is a linear combination
        # of the original target and source image as determined by 'shader_val'.

        if extra_shading:
            # Convert images to grayscale
            greyt = np.dot(ori_img[..., :3], [0.2989, 0.5870, 0.1140])
            greyp = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            greys = np.dot(src_img[..., :3], [0.2989, 0.5870, 0.1140])

            # Standardize the grayscale images for the source and target
            smean, sdev = np.mean(greys), np.std(greys)
            tmean, tdev = np.mean(greyt), np.std(greyt)
            greyt = (greyt - tmean) / tdev

            # Rescale the standardized grayscale target image
            greyt = greyt * (shader_val * sdev + (1.0 - shader_val) * tdev) \
                    + shader_val * smean + (1.0 - shader_val) * tmean

            # Ensure there are no zero or negative values in the grayscale images
            min_val = 1.
            greyp = np.maximum(greyp, min_val)  # Guard against zero divide
            greyt = np.maximum(greyt, 0.0)      # Guard against negative values

            # Rescale each color channel of the processed image
            img = img / greyp[..., None] * greyt[..., None]

        return img

    @staticmethod
    def __normalize_array(array: np.ndarray, desired_min: float, desired_max: float) -> np.ndarray:
        array_min = np.min(array)
        array_max = np.max(array)
        normalized_array = (array - array_min) / (array_max - array_min) * (desired_max - desired_min) + desired_min
        return normalized_array

    @staticmethod
    def __final_adjustment(img: np.ndarray, ori_img: np.ndarray, tint_val: float = 1., modified_val: float = 1., scale_vs_clip: float = 1.) -> np.ndarray:
        # Implements a change to the tint of the final image and
        # its degree of modification if a change is specified.

        # If 100% tint is not specified, compute a weighted average
        # of the processed image and its grayscale representation.
        if tint_val != 1.0:
            # Convert to grayscale
            grey = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])[..., None]

            # Apply the tint by adjusting each channel
            img = tint_val * img + (1.0 - tint_val) * grey

        # If 100% image modification is not specified, compute a weighted average
        # of the processed image and the original target image.
        if modified_val != 1.0:
            img = modified_val * img + (1.0 - modified_val) * ori_img

        scale = EnhancedImageColourTransferNode.__normalize_array(img, 0., 255.)
        clip = np.clip(img, 0., 255.)
        img = (scale * (1. - scale_vs_clip)) + (clip * scale_vs_clip)
        img = np.clip(img, 0., 255.).astype(np.uint8)
        return img

    @staticmethod
    def __tensor_to_rgb255(image: torch.Tensor) -> np.ndarray:
        return (np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def __rgb255_to_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def __transform(src: ArrayLike, m: ArrayLike) -> NDArray[np.float32]:
        """
        Pure NumPy replacement for cv2.transform.

        Parameters
        ----------
        src : ArrayLike
            Input array with shape (..., N), e.g. (H, W, N).
        m : ArrayLike
            Transformation matrix of shape (M, N) or (M, N+1).

        Returns
        -------
        dst : NDArray[np.float32]
            Transformed array with shape (..., M).
        """
        src_arr: NDArray[np.float32] = np.asarray(src, dtype=np.float32)
        m_arr: NDArray[np.float32] = np.asarray(m, dtype=np.float32)

        N = src_arr.shape[-1]
        if m_arr.shape[1] == N:  # simple linear transform
            dst = np.tensordot(src_arr, m_arr.T, axes=1)
        elif m_arr.shape[1] == N + 1:  # affine transform
            dst = np.tensordot(src_arr, m_arr[:, :-1].T, axes=1) + m_arr[:, -1]
        else:
            raise ValueError(
                f"Matrix shape {m_arr.shape} incompatible with src shape {src_arr.shape}"
            )

        return dst.astype(np.float32)

    def __mean_stddev(src: ArrayLike, mask: Optional[ArrayLike] = None) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Pure NumPy replacement for cv2.meanStdDev.

        Parameters
        ----------
        src : ArrayLike
            Input array of shape (H, W[, C]) with 1 to 4 channels.
        mask : Optional[ArrayLike], default=None
            Optional mask of shape (H, W). Nonzero values indicate which
            pixels are included in the calculation.

        Returns
        -------
        mean : NDArray[np.float32]
            Mean values per channel, shape (C, 1).
        stddev : NDArray[np.float32]
            Standard deviation per channel, shape (C, 1).
        """
        src_arr: NDArray[np.float32] = np.asarray(src, dtype=np.float32)

        # Handle grayscale as single-channel
        if src_arr.ndim == 2:
            src_arr = src_arr[..., None]  # shape (H, W, 1)

        H, W, C = src_arr.shape

        if mask is not None:
            mask_arr: NDArray[np.bool_] = np.asarray(mask, dtype=bool)
            if mask_arr.shape != (H, W):
                raise ValueError(f"Mask shape {mask_arr.shape} does not match src shape {(H, W)}")

            # Apply mask to each channel
            values = [src_arr[..., c][mask_arr] for c in range(C)]
        else:
            # Flatten all pixels per channel
            values = [src_arr[..., c].ravel() for c in range(C)]

        # Compute stats per channel
        means = np.array([np.mean(v) if v.size > 0 else 0.0 for v in values], dtype=np.float32)
        stds  = np.array([np.std(v)  if v.size > 0 else 0.0 for v in values], dtype=np.float32)

        # Match OpenCV shape: (C, 1)
        return means[:, None], stds[:, None]

    def __binary_threshold(
        src: ArrayLike,
        thresh: float,
        maxval: float
    ) -> Tuple[float, NDArray[np.float32]]:
        """
        Pure NumPy replacement for cv2.threshold with THRESH_BINARY.

        Parameters
        ----------
        src : ArrayLike
            Input array (grayscale or multi-channel).
        thresh : float
            Threshold value.
        maxval : float
            Maximum value for THRESH_BINARY.

        Returns
        -------
        retval : float
            The used threshold value (same as OpenCV).
        dst : NDArray[np.float32]
            Thresholded output array, same shape as src.
        """
        src_arr: NDArray[np.float32] = np.asarray(src, dtype=np.float32)
        dst = np.where(src_arr > thresh, maxval, 0).astype(np.float32)
        return float(thresh), dst    

    def __rgb2hsv(src: ArrayLike) -> NDArray[np.float32]:
        """
        Convert RGB image to HSV (OpenCV convention: H∈[0,180], S,V∈[0,255]).

        Parameters
        ----------
        src : ArrayLike
            Input RGB image, dtype uint8, shape (H, W, 3).

        Returns
        -------
        dst : NDArray[np.uint8]
            HSV image, same shape (H, W, 3).
        """
        src_arr = np.asarray(src, dtype=np.float32).astype(np.float32) / 255.0
        r, g, b = src_arr[..., 0], src_arr[..., 1], src_arr[..., 2]

        cmax = np.max(src_arr, axis=-1)
        cmin = np.min(src_arr, axis=-1)
        delta = cmax - cmin

        # Hue
        hue = np.zeros_like(cmax)
        mask = delta > 1e-6
        idx = (cmax == r) & mask
        hue[idx] = ((g - b)[idx] / delta[idx]) % 6
        idx = (cmax == g) & mask
        hue[idx] = (b - r)[idx] / delta[idx] + 2
        idx = (cmax == b) & mask
        hue[idx] = (r - g)[idx] / delta[idx] + 4
        hue = hue * 30  # scale to [0,180]

        # Saturation
        sat = np.zeros_like(cmax)
        sat[cmax > 0] = (delta[cmax > 0] / cmax[cmax > 0]) * 255

        # Value
        val = cmax * 255

        hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
        return hsv


    def __hsv2rgb(src: ArrayLike) -> NDArray[np.float32]:
        """
        Convert HSV image (OpenCV convention) to RGB.

        Parameters
        ----------
        src : ArrayLike
            HSV image, dtype uint8, shape (H, W, 3).
            H∈[0,180], S,V∈[0,255].

        Returns
        -------
        dst : NDArray[np.uint8]
            RGB image, same shape (H, W, 3).
        """
        hsv_arr = np.asarray(src, dtype=np.float32).astype(np.float32)
        h = hsv_arr[..., 0] * 2.0  # back to [0,360)
        s = hsv_arr[..., 1] / 255.0
        v = hsv_arr[..., 2] / 255.0

        c = v * s
        x = c * (1 - np.abs((h / 60.0) % 2 - 1))
        m = v - c

        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)

        idx = (0 <= h) & (h < 60)
        r[idx], g[idx], b[idx] = c[idx], x[idx], 0
        idx = (60 <= h) & (h < 120)
        r[idx], g[idx], b[idx] = x[idx], c[idx], 0
        idx = (120 <= h) & (h < 180)
        r[idx], g[idx], b[idx] = 0, c[idx], x[idx]
        idx = (180 <= h) & (h < 240)
        r[idx], g[idx], b[idx] = 0, x[idx], c[idx]
        idx = (240 <= h) & (h < 300)
        r[idx], g[idx], b[idx] = x[idx], 0, c[idx]
        idx = (300 <= h) & (h < 360)
        r[idx], g[idx], b[idx] = c[idx], 0, x[idx]

        rgb = np.stack([(r + m), (g + m), (b + m)], axis=-1)
        return (rgb * 255).astype(np.float32)
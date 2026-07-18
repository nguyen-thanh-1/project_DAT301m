import numpy as np

def rgb_to_gray_np(img_float):
    # img_float: HxWx3, giá trị [0,1]
    return 0.2989 * img_float[..., 0] + 0.5870 * img_float[..., 1] + 0.1140 * img_float[..., 2]

def local_binary_pattern_np(gray, P=8, R=1):
    """Classic circular LBP, implemented from scratch with bilinear-sampled neighbors."""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.float32)
    for p in range(P):
        theta = 2 * np.pi * p / P
        dy, dx = -R * np.sin(theta), R * np.cos(theta)
        y0, x0 = np.floor(dy).astype(int), np.floor(dx).astype(int)
        fy, fx = dy - y0, dx - x0

        shifted = np.zeros_like(gray)
        for oy in (0, 1):
            for ox in (0, 1):
                wgt = (1 - abs(fy - oy)) * (1 - abs(fx - ox))
                if wgt <= 0:
                    continue
                shifted += wgt * np.roll(np.roll(gray, -(y0 + oy), axis=0), -(x0 + ox), axis=1)
        lbp += (shifted >= gray).astype(np.float32) * (2 ** p)
    return lbp / (2 ** P - 1)  # normalize to [0,1]

def lbp_numpy_wrapper(img_uint8, P, R):
    img_float = img_uint8.astype(np.float32) / 255.0
    gray = rgb_to_gray_np(img_float)
    lbp = local_binary_pattern_np(gray, P=P, R=R)
    return lbp[..., None].astype(np.float32)  # HxWx1

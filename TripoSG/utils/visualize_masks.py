"""
Segmentation visualization using RMBG background removal masks.
Generates colored overlays with contours and labels on the original image.
"""
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


def get_rmbg_mask(image_path: str, rmbg_net) -> np.ndarray:
    """
    Run RMBG on an image and return a binary foreground mask.
    Returns: uint8 numpy array (H x W), values 0 or 255.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Convert to RGB regardless of input channels
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = rgb.shape[:2]
    rgb_tensor = torch.from_numpy(rgb).cuda().float().permute(2, 0, 1) / 255.0

    # Resize to 1024 for RMBG inference
    resize_1024 = transforms.Resize((1024, 1024), antialias=True)
    rgb_1024 = resize_1024(rgb_tensor)

    norm = TF.normalize(rgb_1024, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).unsqueeze(0)
    with torch.no_grad():
        result = rmbg_net(norm)
    # result[0][0] is (1, H, W) — keep channel dim so Resize works correctly
    alpha = result[0][0]  # shape: (1, H, W)
    torch.cuda.synchronize()  # ensure RMBG kernel is fully done before next ops

    # Resize back to original resolution (needs ≥3D tensor)
    resize_back = transforms.Resize((orig_h, orig_w), antialias=True)
    alpha = resize_back(alpha)   # (1, H, W)
    alpha = alpha.squeeze(0)     # (H, W) — squeeze only after resize

    # Normalize to [0, 1]
    ma, mi = alpha.max(), alpha.min()
    alpha = (alpha - mi) / (ma - mi + 1e-8)

    # Squeeze ALL size-1 dimensions so the result is exactly (H, W)
    alpha = alpha.squeeze()
    # If still >2D (e.g. model returned multi-channel output), take first channel
    if alpha.dim() > 2:
        alpha = alpha[0]

    # Move to CPU before numpy conversion
    alpha_np = (alpha * 255).to(torch.uint8).cpu().numpy()
    # Ensure 2D going into cv2 (handles any lingering extra dims)
    if alpha_np.ndim > 2:
        alpha_np = alpha_np[0]
    _, alpha_np = cv2.threshold(alpha_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small noise blobs
    labeled = label(alpha_np)
    cleaned = remove_small_objects(labeled, min_size=200)
    mask = (cleaned > 0).astype(np.uint8) * 255

    # Guarantee exactly 2D — this is the array shape everything downstream expects
    return mask.reshape(orig_h, orig_w)


def create_segmentation_overlay(image_path: str, rmbg_net, output_path: str) -> str:
    """
    Create a multi-component colored segmentation overlay on the original image.

    Erodes the binary RMBG mask to separate touching parts, then uses connected-component
    analysis so each distinct jewelry region gets its own color and label.

    Args:
        image_path: Path to the input image.
        rmbg_net: Loaded RMBG model (already on CUDA).
        output_path: Where to save the overlay PNG.

    Returns:
        output_path (for chaining).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    mask = get_rmbg_mask(image_path, rmbg_net)
    # Resize mask to match img_bgr in case of any rounding differences
    if mask.shape != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # ── Multi-component segmentation ────────────────────────────────────────────
    # Erode the mask to separate touching parts (e.g. gemstone touching the band),
    # label each resulting blob independently, then dilate back to restore outlines.
    img_area      = img_bgr.shape[0] * img_bgr.shape[1]
    kernel_size   = max(5, min(img_bgr.shape[:2]) // 45)
    kernel        = np.ones((kernel_size, kernel_size), np.uint8)
    eroded        = cv2.erode(mask, kernel, iterations=3)

    # Connected-component stats on the eroded mask
    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(eroded)

    min_area = img_area * 0.0015
    valid    = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    # Sort by area descending so the largest part gets the "primary" label
    valid.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)

    # Colour + label palette for up to 4 components (BGR)
    PALETTE = [
        ([30,  180,  80],  "Band"),       # green  — main body / band
        ([180,  60,  40],  "Gemstone"),   # blue   — primary stone
        ([40,  160, 220],  "Setting"),    # gold   — prong / bezel setting
        ([30,  100, 200],  "Prong"),      # orange — individual prongs / details
    ]

    overlay = img_bgr.copy()
    blend_alpha = 0.38

    for idx, comp_id in enumerate(valid[:len(PALETTE)]):
        color_bgr, comp_label = PALETTE[idx]

        # Dilate the eroded component back to approximate original size
        comp_mask  = (label_map == comp_id).astype(np.uint8) * 255
        comp_dilated = cv2.dilate(comp_mask, kernel, iterations=3)
        # Clamp to original RMBG mask to avoid colouring outside the foreground
        comp_dilated = cv2.bitwise_and(comp_dilated, mask)

        color_layer = np.full_like(img_bgr, color_bgr)
        blended     = cv2.addWeighted(overlay, 1.0 - blend_alpha, color_layer, blend_alpha, 0)
        m3          = (comp_dilated > 0)[:, :, np.newaxis]
        overlay     = np.where(m3, blended, overlay).astype(np.uint8)

        # Contour outline
        contours, _ = cv2.findContours(comp_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(color_bgr), 2)

        # Label near the bounding box
        if contours:
            largest    = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            label_y    = max(y - 10, 20)
            cv2.putText(
                overlay, comp_label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, tuple(color_bgr), 2, cv2.LINE_AA
            )

    # Fallback: if erosion removed everything, draw the plain RMBG mask as before
    if not valid:
        color_layer = np.full_like(img_bgr, [30, 180, 80])
        blended     = cv2.addWeighted(img_bgr, 1.0 - blend_alpha, color_layer, blend_alpha, 0)
        m3          = (mask > 0)[:, :, np.newaxis]
        overlay     = np.where(m3, blended, img_bgr).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 220, 60), 2)
        if contours:
            largest    = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.putText(overlay, "Jewelry", (x, max(y - 12, 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 60), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)
    return output_path

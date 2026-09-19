"""Direct context editing with a single, late, soft pixel-space composition."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox


def edit_context_crop(
    pipe,
    source,
    row,
    mask,
    generator,
    steps=40,
    true_cfg_scale=4.0,
    guidance_scale=1.0,
    negative_prompt=" ",
    remove_context_window=False,
    diagnostics_dir=None,
    target_guide=False,
    attribute_mask_composition=False,
    preserve_attribute_texture=False,
):
    """Keep surrounding evidence during diffusion and protect distant pixels.

    Unlike exact-mask latent reinjection, the old target silhouette is not
    restored at each denoising step. The final editable rectangle includes a
    collar around the whole old footprint, admitting a new object silhouette.
    Add masks identify an anchor, so their full context window is writable.
    This is not a quality guarantee: collateral edits and seams still need audit.
    """
    bbox = padded_mask_bbox(mask, padding_fraction=0.75, min_padding=128)
    crop = source.crop(bbox)
    image_input = crop
    prompt = row["editing_instruction"]
    if target_guide:
        # Official Qwen-Image-Edit supports multiple reference images. The first
        # stays clean; the second supplies non-semantic region geometry only.
        from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs

        local_mask = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        guide, _ = audit_two_image_inputs(
            crop, crop, local_mask, longest_side=max(crop.size), scope="full"
        )
        guide = guide.crop((0, 40, guide.width, guide.height))
        image_input = [crop, guide]
        prompt = (
            "Edit Image 1. Image 2 is the same photo with a black/white outline "
            "identifying the exact target; the outline is a guide, not an object. "
            + prompt
            + " Apply this action to that target only. Return one natural "
            "edited version of Image 1 without any guide outline or annotation."
        )
    if preserve_attribute_texture:
        prompt += (
            " Make the requested property change clearly visible while preserving the "
            "target's original geometry, photographic texture, local shading and fine "
            "surface detail. A color change must follow the existing lighting and folds, "
            "not become a flat painted shape. Match the original level of focus."
        )
        negative_prompt = (
            "sticker border, black and white annotation outline, flat painted fill, "
            "posterization, cartoon rendering, ghost contour, pasted patch"
        )
    edited = (
        pipe(
            image=image_input,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )
        .images[0]
        .resize(crop.size, Image.Resampling.LANCZOS)
    )
    result, alpha = compose_context_crop(
        source, edited, mask, row["task_type"], bbox, remove_context_window
    )
    if attribute_mask_composition:
        if row["task_type"] != "attribute":
            raise ValueError(
                "Exact late composition is only intended for same-geometry attributes"
            )
        result, alpha = compose_attribute_crop(source, edited, mask, bbox)
    if diagnostics_dir is not None:
        import json
        from pathlib import Path

        directory = Path(diagnostics_dir)
        directory.mkdir(parents=True, exist_ok=True)
        crop.save(directory / "source_crop.png")
        edited.save(directory / "raw_edited_crop.png")
        alpha.save(directory / "composition_alpha.png")
        if target_guide:
            guide.save(directory / "target_guide.png")
        (directory / "generation_request.json").write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "task_type": row["task_type"],
                    "steps": steps,
                    "true_cfg_scale": true_cfg_scale,
                    "guidance_scale": guidance_scale,
                    "crop_bbox": bbox,
                    "source_size": source.size,
                    "reference_images": 2 if target_guide else 1,
                    "attribute_mask_composition": attribute_mask_composition,
                    "preserve_attribute_texture": preserve_attribute_texture,
                },
                indent=2,
            )
        )
    return result


def compose_attribute_crop(source, edited, mask, bbox):
    """Restrict a same-geometry property edit to original target pixels.

    This is a late pixel-space operation, not reference-latent reinjection.
    External guide strokes and unrelated instances are excluded. New object
    shapes must NOT use this policy: they would be clipped by the old shape.
    """
    crop = source.crop(bbox)
    if mask.shape != (source.height, source.width) or edited.size != crop.size:
        raise ValueError("Source, mask and edited crop must be aligned")
    inside = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype(bool)
    if not inside.any():
        raise ValueError("Attribute target mask must be nonempty")
    binary = Image.fromarray(inside.astype(np.uint8) * 255)
    soft = binary.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5))
    alpha = Image.fromarray(np.where(inside, np.asarray(soft), 0).astype(np.uint8))
    result = source.copy()
    result.paste(Image.composite(edited, crop, alpha), bbox[:2])
    return result, alpha


def compose_context_crop(
    source, edited, mask, task_type, bbox, remove_context_window=False
):
    """Feather only internal crop boundaries, never a true image boundary."""
    if mask.shape != (source.height, source.width):
        raise ValueError("Mask shape must match the original image")
    if task_type not in {"add", "remove", "replace", "attribute"}:
        raise ValueError(f"Unsupported task type: {task_type}")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height):
        raise ValueError("Crop box must lie within the original image")
    crop = source.crop(bbox)
    if edited.size != crop.size:
        raise ValueError("Edited crop must be aligned with the source crop")
    local_mask = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    alpha = Image.new("L", crop.size, 0)
    ys, xs = np.nonzero(local_mask)
    if not len(xs):
        raise ValueError("Crop must contain a nonempty target mask")
    pad = max(32, round(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.2))
    ImageDraw.Draw(alpha).rectangle(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(crop.width - 1, int(xs.max()) + pad),
            min(crop.height - 1, int(ys.max()) + pad),
        ),
        fill=255,
    )
    if task_type == "add" or (task_type == "remove" and remove_context_window):
        edge = min(24, min(crop.size) // 8)
        arr = np.ones((crop.height, crop.width), np.float32)
        for i in range(edge):
            v = (i + 1) / (edge + 1)
            if bbox[1] > 0:
                arr[i, :] *= v
            if bbox[3] < source.height:
                arr[-i - 1, :] *= v
            if bbox[0] > 0:
                arr[:, i] *= v
            if bbox[2] < source.width:
                arr[:, -i - 1] *= v
        alpha = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        alpha = alpha.filter(ImageFilter.GaussianBlur(10))
    result = source.copy()
    result.paste(Image.composite(edited, crop, alpha), bbox[:2])
    return result, alpha

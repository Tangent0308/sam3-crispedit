"""Build annotation-safe visual panels for instruction generation and audit.

The source pixels are never painted with a semantic-looking mask color.  A
clean crop supplies appearance, while a separate black/white mask supplies
geometry only.  The audit panel also magnifies the same area before and after
editing so that small and partial edits remain judgeable by a VLM.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def padded_mask_bbox(
    mask: np.ndarray, padding_fraction: float = 0.18, min_padding: int = 24
) -> tuple[int, int, int, int]:
    """Return an image-clipped mask box with enough surrounding context."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot build a visual prompt panel from an empty mask")
    height, width = mask.shape
    object_width = int(xs.max() - xs.min() + 1)
    object_height = int(ys.max() - ys.min() + 1)
    pad_x = max(min_padding, round(object_width * padding_fraction))
    pad_y = max(min_padding, round(object_height * padding_fraction))
    return (
        max(0, int(xs.min()) - pad_x),
        max(0, int(ys.min()) - pad_y),
        min(width, int(xs.max()) + 1 + pad_x),
        min(height, int(ys.max()) + 1 + pad_y),
    )


def _letterbox(
    image: Image.Image,
    size: tuple[int, int],
    *,
    magnify: bool = False,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    image = image.convert("RGB")
    if magnify:
        scale = min(size[0] / image.width, size[1] / image.height)
        fitted = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            resample,
        )
    else:
        fitted = image.copy()
        fitted.thumbnail(size, resample)
    canvas = Image.new("RGB", size, (238, 238, 238))
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _panel(
    images_and_labels: list[tuple[Image.Image, str]], tile_size: int, *, magnify: bool = False
) -> Image.Image:
    header_height = 36
    gap = 4
    width = len(images_and_labels) * tile_size + (len(images_and_labels) - 1) * gap
    canvas = Image.new("RGB", (width, tile_size + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(images_and_labels):
        x = index * (tile_size + gap)
        draw.text((x + 8, 11), label, fill="black")
        resample = (
            Image.Resampling.NEAREST
            if label == "BINARY TARGET MASK"
            else Image.Resampling.LANCZOS
        )
        canvas.paste(
            _letterbox(
                image,
                (tile_size, tile_size),
                magnify=magnify,
                resample=resample,
            ),
            (x, header_height),
        )
    return canvas


def binary_mask_crop(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    binary = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return binary.crop(bbox).convert("RGB")


def clean_target_cutout(
    source: Image.Image, mask: np.ndarray, bbox: tuple[int, int, int, int]
) -> Image.Image:
    """Show only unmodified target pixels on a neutral checkerboard."""
    source_crop = np.asarray(source.convert("RGB").crop(bbox), dtype=np.uint8)
    mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    yy, xx = np.indices(mask_crop.shape)
    checker = ((xx // 16 + yy // 16) % 2).astype(np.uint8)
    background = np.where(checker[..., None] == 0, 210, 242).astype(np.uint8)
    cutout = np.where(mask_crop[..., None], source_crop, background)
    return Image.fromarray(cutout, mode="RGB")


def isolated_target_image(
    source: Image.Image, mask: np.ndarray, tile_size: int = 768
) -> Image.Image:
    """Magnify only pixels inside one mask as a standalone model input."""
    bbox = padded_mask_bbox(mask)
    return _letterbox(clean_target_cutout(source, mask, bbox), (tile_size, tile_size))


def target_footprint_comparison(
    source: Image.Image,
    edited: Image.Image,
    mask: np.ndarray,
    tile_size: int = 640,
) -> Image.Image:
    """Compare a tight aligned crop before and after with an exact mask edge.

    Keeping a rectangular photographic crop avoids the misleading effect of
    clipping edited background through the original object's silhouette.  A
    thin black/white contour identifies the exact footprint while leaving
    nearby same-category instances visible as explicit distractors.
    """
    bbox = padded_mask_bbox(mask, padding_fraction=0.05, min_padding=8)

    def edged_crop(image: Image.Image) -> Image.Image:
        pixels = np.asarray(image.convert("RGB").crop(bbox), dtype=np.uint8).copy()
        mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype(bool)
        padded = np.pad(mask_crop, 2, mode="constant", constant_values=False)
        dilated = np.zeros_like(mask_crop, dtype=bool)
        eroded = np.ones_like(mask_crop, dtype=bool)
        for dy in range(5):
            for dx in range(5):
                shifted = padded[dy : dy + mask_crop.shape[0], dx : dx + mask_crop.shape[1]]
                dilated |= shifted
                eroded &= shifted
        band = dilated & ~eroded
        pixels[band] = 0
        inner = mask_crop & ~eroded
        pixels[inner] = 255
        return Image.fromarray(pixels, mode="RGB")

    return _panel(
        [
            (edged_crop(source), "SOURCE; TARGET INSIDE B/W EDGE"),
            (edged_crop(edited), "EDITED; SAME TARGET EDGE"),
        ],
        tile_size,
        magnify=True,
    )


def instruction_target_panel(
    source: Image.Image, mask: np.ndarray, tile_size: int = 512
) -> Image.Image:
    """Return clean target context beside its separate binary geometry mask."""
    bbox = padded_mask_bbox(mask)
    return _panel(
        [
            (source.convert("RGB").crop(bbox), "CLEAN TARGET CROP"),
            (clean_target_cutout(source, mask, bbox), "CLEAN TARGET PIXELS"),
            (binary_mask_crop(mask, bbox), "BINARY TARGET MASK"),
        ],
        tile_size,
    )


def instruction_target_crop(
    source: Image.Image, mask: np.ndarray, tile_size: int = 768, target_pointer: bool = False,
    excluded_mask: np.ndarray | None = None,
    context_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Magnify a photographic context crop and mark the exact mask externally.

    There is no semantic fill or cutout.  The black/white contour lies just
    outside the mask; a tiny explicitly labelled blue point disambiguates the
    selected side while leaving essentially all target appearance visible.
    """
    bbox = context_bbox or padded_mask_bbox(mask, padding_fraction=0.4, min_padding=48)
    crop = source.convert("RGB").crop(bbox)
    mask_crop = Image.fromarray(
        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]].astype(np.uint8) * 255,
        mode="L",
    )
    # Leave room OUTSIDE the photograph for contours touching the image edge.
    # Without this margin an edge-connected region can look like its complement.
    margin = 8
    scale = min((tile_size - 2 * margin) / crop.width, (tile_size - 2 * margin) / crop.height)
    display_size = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    displayed = np.asarray(
        crop.resize(display_size, Image.Resampling.LANCZOS), dtype=np.uint8
    ).copy()
    displayed = np.pad(displayed, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)
    displayed_mask = Image.fromarray(np.pad(
        np.asarray(mask_crop.resize(display_size, Image.Resampling.NEAREST)), margin
    ))
    inside = np.asarray(displayed_mask, dtype=np.uint8) > 0
    near = np.asarray(displayed_mask.filter(ImageFilter.MaxFilter(5))) > 0
    far = np.asarray(displayed_mask.filter(ImageFilter.MaxFilter(11))) > 0
    displayed[far & ~near] = (255, 255, 255)
    displayed[near & ~inside] = (0, 0, 0)

    header_height = 42
    canvas = Image.new("RGB", (tile_size, tile_size + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 13), "MASK REGION: INSIDE BLACK/WHITE OUTLINE", fill="black")
    offset = (
        (tile_size - displayed.shape[1]) // 2,
        header_height + (tile_size - displayed.shape[0]) // 2,
    )
    canvas.paste(Image.fromarray(displayed, mode="RGB"), offset)
    if target_pointer:
        # Put labels in a new external gutter and point to each substantial
        # visible fragment.  Disconnected fragments can belong to one occluded
        # instance, so this is membership evidence rather than object counting.
        import cv2
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            inside.astype(np.uint8), connectivity=8
        )
        total = int(inside.sum())
        minimum = max(16, int(round(total * 0.02)))
        selected_y, selected_x = np.nonzero(inside)
        selected_span = max(
            int(selected_x.max() - selected_x.min() + 1),
            int(selected_y.max() - selected_y.min() + 1),
        )
        components = [
            index for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum
            or (
                int(stats[index, cv2.CC_STAT_AREA]) >= 16
                and max(
                    int(stats[index, cv2.CC_STAT_WIDTH]),
                    int(stats[index, cv2.CC_STAT_HEIGHT]),
                ) >= max(24, round(0.22 * selected_span))
            )
        ]
        if not components and count > 1:
            components = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
        components = sorted(
            components,
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
            reverse=True,
        )[:3]
        # Enclosed complement components are true excluded holes.  Label only
        # meaningful ones; tiny antialiasing islands remain visible but do not
        # add prompt clutter.
        outside_count, outside_labels, outside_stats, _ = cv2.connectedComponentsWithStats(
            (~inside).astype(np.uint8), connectivity=8
        )
        border_labels = set(np.concatenate([
            outside_labels[0], outside_labels[-1],
            outside_labels[:, 0], outside_labels[:, -1],
        ]).tolist())
        hole_minimum = max(32, int(round(total * 0.002)))
        holes = [
            index for index in range(1, outside_count)
            if index not in border_labels
            and int(outside_stats[index, cv2.CC_STAT_AREA]) >= hole_minimum
        ]
        holes = sorted(
            holes,
            key=lambda index: int(outside_stats[index, cv2.CC_STAT_AREA]),
            reverse=True,
        )[:3]
        left_gutter = 112
        right_gutter = 128 if holes or excluded_mask is not None else 0
        pointed = Image.new(
            'RGB', (canvas.width + left_gutter + right_gutter, canvas.height), 'white'
        )
        pointed.paste(canvas, (left_gutter, 0))
        painter = ImageDraw.Draw(pointed)
        target_points = []
        for component_number, component in enumerate(components):
            component_mask = labels == component
            distance = cv2.distanceTransform(
                component_mask.astype(np.uint8), cv2.DIST_L2, 5
            )
            py, px = np.unravel_index(distance.argmax(), distance.shape)
            target_points.append((int(px), int(py),
                                  'TARGET INTERIOR' if component_number == 0 else 'TARGET FRAGMENT'))

        # One connected component can contain a large owner plus a long thin
        # selected extension (a bat, rope, handle, branch, or limb). A single
        # deepest-interior point tends to hide that extension from the VLM.
        # Add at most two sparse membership landmarks to the largest component,
        # selected by farthest-point coverage. These are points, never a fill,
        # so source appearance remains readable and cannot be mistaken for an
        # object color.
        largest_width = int(stats[components[0], cv2.CC_STAT_WIDTH]) if components else 0
        largest_height = int(stats[components[0], cv2.CC_STAT_HEIGHT]) if components else 0
        largest_area = int(stats[components[0], cv2.CC_STAT_AREA]) if components else 0
        fill_ratio = largest_area / max(1, largest_width * largest_height)
        aspect_ratio = max(largest_width, largest_height) / max(
            1, min(largest_width, largest_height)
        )
        needs_extension_landmarks = fill_ratio < 0.65 or aspect_ratio > 2.4
        if components and len(target_points) < 3 and needs_extension_landmarks:
            largest_mask = labels == components[0]
            coordinates = np.argwhere(largest_mask)
            diagonal = float(np.hypot(largest_height, largest_width))
            chosen = [(target_points[0][1], target_points[0][0])]
            while len(target_points) < 3 and len(coordinates):
                squared = np.full(len(coordinates), np.inf)
                for chosen_y, chosen_x in chosen:
                    squared = np.minimum(
                        squared,
                        (coordinates[:, 0] - chosen_y) ** 2
                        + (coordinates[:, 1] - chosen_x) ** 2,
                    )
                candidate = int(np.argmax(squared))
                if float(np.sqrt(squared[candidate])) < max(32.0, 0.28 * diagonal):
                    break
                py, px = map(int, coordinates[candidate])
                chosen.append((py, px))
                target_points.append((px, py, 'TARGET EXTENSION'))

        target_centers = []
        for marker_number, (px, py, label) in enumerate(target_points, start=1):
            target_centers.append((px, py, marker_number))
            component_mask = inside
            edge = int(px)
            while edge > 0 and component_mask[py, edge - 1]:
                edge -= 1
            end_x = left_gutter + offset[0] + edge - 1
            end_y = offset[1] + int(py)
            label_y = max(0, min(pointed.height - 18, end_y - 17))
            painter.text((6, label_y), f'TARGET {marker_number} {label[7:]}', fill='black')
            painter.line((8, end_y, end_x, end_y), fill=(0, 110, 210), width=3)
            painter.line(
                (end_x - 9, end_y - 6, end_x, end_y, end_x - 9, end_y + 6),
                fill=(0, 110, 210), width=3,
            )
        for hole_number, hole in enumerate(holes):
            hole_mask = outside_labels == hole
            distance = cv2.distanceTransform(
                hole_mask.astype(np.uint8), cv2.DIST_L2, 5
            )
            py, px = np.unravel_index(distance.argmax(), distance.shape)
            edge = int(px)
            while edge + 1 < hole_mask.shape[1] and hole_mask[py, edge + 1]:
                edge += 1
            end_x = left_gutter + offset[0] + edge + 1
            end_y = offset[1] + int(py)
            label_x = left_gutter + canvas.width + 6
            label_y = max(0, min(pointed.height - 18, end_y - 17))
            label = 'EXCLUDED HOLE' if hole_number == 0 else 'EXCLUDED HOLE'
            painter.text((label_x, label_y), label, fill='black')
            painter.line((label_x, end_y, end_x, end_y), fill=(215, 125, 0), width=3)
            painter.line(
                (end_x + 9, end_y - 6, end_x, end_y, end_x + 9, end_y + 6),
                fill=(215, 125, 0), width=3,
            )
        # Preserve every selected pixel exactly even for concave shapes whose
        # other component intersects the leader's path.
        pixels = np.asarray(pointed).copy()
        original = np.asarray(canvas)
        selected = np.zeros((canvas.height, canvas.width), dtype=bool)
        selected[offset[1]:offset[1]+inside.shape[0], offset[0]:offset[0]+inside.shape[1]] = inside
        pixels[:, left_gutter:left_gutter+canvas.width][selected] = original[selected]
        result = Image.fromarray(pixels)
        # Keep only a tiny marker inside the selected pixels.  The long leader
        # is restored over the target, so source texture remains readable.
        marker = ImageDraw.Draw(result)
        for px, py, marker_number in target_centers:
            cx = left_gutter + offset[0] + px
            cy = offset[1] + py
            marker.ellipse((cx - 5, cy - 5, cx + 5, cy + 5),
                           fill=(0, 110, 210), outline='white', width=2)
            marker.text((cx - 3, cy - 5), str(marker_number), fill='white')
        if excluded_mask is not None:
            if excluded_mask.shape != mask.shape:
                raise ValueError('Excluded mask dimensions differ')
            excluded = excluded_mask.astype(bool) & ~mask.astype(bool)
            local = Image.fromarray(excluded[bbox[1]:bbox[3],bbox[0]:bbox[2]].astype(np.uint8)*255)
            local = np.pad(np.asarray(local.resize(display_size,Image.Resampling.NEAREST)),margin)>0
            _, exlabels, exstats, _ = cv2.connectedComponentsWithStats(local.astype(np.uint8),8)
            components = sorted(range(1,len(exstats)), key=lambda i:int(exstats[i,cv2.CC_STAT_AREA]),reverse=True)[:2]
            for number, component in enumerate(components,1):
                distance=cv2.distanceTransform((exlabels==component).astype(np.uint8),cv2.DIST_L2,5)
                py,px=np.unravel_index(distance.argmax(),distance.shape)
                cx=left_gutter+offset[0]+int(px);cy=offset[1]+int(py)
                marker.ellipse((cx-5,cy-5,cx+5,cy+5),fill=(215,125,0),outline='white',width=2)
                marker.text((cx-3,cy-5),str(number),fill='white')
                marker.text((left_gutter+canvas.width+4,cy-8),f'OUTSIDE {number}',fill='black')
            marker.text((left_gutter+8,30),'ORANGE DOTS: OUTSIDE ORIGINAL TARGET',fill='black')
        return result
    return canvas


def _difference_map(
    source_crop: Image.Image, edited_crop: Image.Image, mask_crop: np.ndarray
) -> Image.Image:
    """Return a false-color absolute difference map with a white mask contour."""
    source_array = np.asarray(source_crop.convert("RGB"), dtype=np.float32)
    edited_array = np.asarray(edited_crop.convert("RGB"), dtype=np.float32)
    delta = np.abs(source_array - edited_array).mean(axis=2) / 255.0
    strength = np.clip(delta / 0.35, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.round(255.0 * strength).astype(np.uint8)
    heat[..., 2] = np.round(255.0 * (1.0 - strength)).astype(np.uint8)

    padded = np.pad(mask_crop.astype(bool), 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask_crop, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + mask_crop.shape[0], dx : dx + mask_crop.shape[1]]
    contour = mask_crop.astype(bool) & ~eroded
    heat[contour] = 255
    return Image.fromarray(heat, mode="RGB")


def audit_visual_inputs(
    source: Image.Image,
    edited: Image.Image,
    mask: np.ndarray,
    full_tile_size: int = 640,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Return source localization, crop-level edit evidence, and full comparison."""
    if edited.size != source.size:
        edited = edited.resize(source.size, Image.Resampling.LANCZOS)
    bbox = padded_mask_bbox(mask)
    source_crop = source.convert("RGB").crop(bbox)
    edited_crop = edited.convert("RGB").crop(bbox)
    source_localization = target_footprint_comparison(source, edited, mask)
    mask_crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    detail_comparison = _panel(
        [
            (source_crop, "ALIGNED SOURCE CROP"),
            (binary_mask_crop(mask, bbox), "BINARY TARGET MASK"),
            (edited_crop, "ALIGNED EDITED CROP"),
            (
                _difference_map(source_crop, edited_crop, mask_crop),
                "ABS DIFF; WHITE = MASK EDGE",
            ),
        ],
        448,
        magnify=True,
    )
    full_comparison = _panel(
        [
            (source.convert("RGB"), "FULL SOURCE"),
            (edited.convert("RGB"), "FULL EDITED"),
        ],
        full_tile_size,
    )
    return source_localization, detail_comparison, full_comparison


def audit_two_image_inputs(
    source: Image.Image,
    edited: Image.Image,
    mask: np.ndarray,
    longest_side: int = 1280,
    scope: str = "full",
) -> tuple[Image.Image, Image.Image]:
    """Show only the aligned pair, marking the source mask without hiding its pixels.

    The contour is outside the target.  Both images use identical scaling so
    the viewer can locate the corresponding region in the unmarked edit.
    """
    if mask.shape != (source.height, source.width):
        raise ValueError("The source mask and image dimensions must match")
    if not mask.any():
        raise ValueError("Cannot audit an empty target mask")
    if edited.size != source.size:
        edited = edited.resize(source.size, Image.Resampling.LANCZOS)
    if scope == "context_crop":
        bbox = padded_mask_bbox(mask, padding_fraction=0.75, min_padding=80)
        source = source.crop(bbox)
        edited = edited.crop(bbox)
        mask = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    elif scope != "full":
        raise ValueError(f"Unsupported two-image audit scope: {scope!r}")

    scale = longest_side / max(source.size)
    display_size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    source_pixels = np.asarray(
        source.convert("RGB").resize(display_size, Image.Resampling.LANCZOS),
        dtype=np.uint8,
    ).copy()
    resized_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        display_size, Image.Resampling.NEAREST
    )
    inside = np.asarray(resized_mask, dtype=np.uint8) > 0
    near = np.asarray(resized_mask.filter(ImageFilter.MaxFilter(7))) > 0
    far = np.asarray(resized_mask.filter(ImageFilter.MaxFilter(13))) > 0
    source_pixels[far & ~near] = (255, 255, 255)
    source_pixels[near & ~inside] = (0, 0, 0)
    edited_display = edited.convert("RGB").resize(
        display_size, Image.Resampling.LANCZOS
    )

    def with_label(photo: Image.Image, label: str) -> Image.Image:
        canvas = Image.new("RGB", (display_size[0], display_size[1] + 40), "white")
        ImageDraw.Draw(canvas).text((10, 13), label, fill="black")
        canvas.paste(photo, (0, 40))
        return canvas

    return (
        with_label(
            Image.fromarray(source_pixels, mode="RGB"),
            "SOURCE: BLACK/WHITE OUTLINE = TARGET MASK",
        ),
        with_label(edited_display, "EDITED: SAME IMAGE COORDINATES"),
    )

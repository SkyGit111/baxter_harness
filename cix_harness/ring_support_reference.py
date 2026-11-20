"""Compare the current gripper approach area with a stationary RGB-D survey."""
import base64
import zlib

import numpy as np


def _pack(image):
    return base64.b64encode(zlib.compress(image.tobytes(order='C'), 6)).decode('ascii')


def _unpack(data, dtype, shape):
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    stream = zlib.decompressobj()
    raw = stream.decompress(base64.b64decode(data, validate=True), expected + 1)
    if len(raw) != expected or not stream.eof:
        raise ValueError('Invalid surveyed image patch')
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def make_reference(bgr, depths, origin, footprint, ring, frame_ids):
    """Build the reference from registered static images and a measured footprint mask."""
    depth = np.asarray(depths)
    rgb = np.asarray(bgr)
    mask = np.asarray(footprint, bool)
    if len(depth) < 4 or len(set(frame_ids)) != len(depth) or rgb.shape != (*depth.shape, 3):
        raise ValueError('Need four or more distinct, aligned RGB-D observations')
    if mask.shape != depth.shape[1:] or np.count_nonzero(mask) < 64 or max(mask.shape) > 121:
        raise ValueError('Invalid surveyed footprint')
    good = (depth > .1) & (depth < 3.) & np.isfinite(depth)
    # Masked median avoids allowing missing depths to pull a reference toward the camera.
    med = np.ma.median(np.ma.array(depth, mask=~good), axis=0).filled(0.)
    mask = mask & (good.sum(axis=0) >= len(depth) - 1)
    return {
        'version': 1, 'origin_px': list(map(int, origin)), 'shape': list(mask.shape),
        'center_uv_px': ring['center_uv_px'], 'center_base_xy_m': ring['center_base_m'][:2],
        'frame_ids': list(map(int, frame_ids)), 'bare_mask': _pack(mask.astype(np.uint8)),
        'depth_mm': _pack(np.rint(med*1000).astype('<u2')), 'bgr': _pack(np.median(rgb, axis=0).astype(np.uint8))
    }


def check_reference(geometry, bgr, ring, reference):
    """Check placement, visibility, global height, and connected local changes in this frame."""
    import cv2
    report = {'verified': False}
    if reference.get('version') != 1 or len(set(reference['frame_ids'])) < 4:
        return dict(report, reason='invalid_survey_version_or_count')
    uv_error = np.linalg.norm(np.asarray(ring['center_uv_px'])-reference['center_uv_px'])
    xy_error = np.linalg.norm(np.asarray(ring['center_base_m'][:2])-reference['center_base_xy_m'])
    report.update(ring_shift_px=float(uv_error), ring_shift_xy_m=float(xy_error),
                  survey_frame_ids=reference['frame_ids'])
    if 'jaw_axis_base_xy' in reference:
        report['jaw_axis_base_xy'] = reference['jaw_axis_base_xy']
    if uv_error > 2.5 or xy_error > .007:
        return dict(report, reason='ring_moved_from_surveyed_site')
    h, w = reference['shape']
    x, y = reference['origin_px']
    if not 5 <= min(w, h) <= max(w, h) <= 121 or min(x, y) < 0:
        return dict(report, reason='invalid_survey_footprint')
    rgb = bgr[y:y+h, x:x+w]
    depth = geometry.depth[y:y+h, x:x+w]
    if rgb.shape[:2] != (h, w):
        return dict(report, reason='survey_outside_image')
    original = _unpack(reference['bgr'], np.uint8, (h, w, 3))
    baseline = _unpack(reference['depth_mm'], '<u2', (h, w)) * .001
    mask = _unpack(reference['bare_mask'], np.uint8, (h, w)) > 0
    if mask.sum() < 64:
        return dict(report, reason='missing_survey_coverage')
    valid = mask & (depth > .1) & (depth < 3.) & np.isfinite(depth)
    coverage = valid.sum() / mask.sum()
    report.update(depth_coverage=float(coverage), surveyed_pixels=int(mask.sum()),
                  current_depth_pixels=int(valid.sum()))
    if coverage < .90:
        return dict(report, reason='insufficient_current_depth_coverage')
    py, px = np.nonzero(valid)
    rays = geometry.rays(np.column_stack((px+x, py+y)))
    heights = (depth[valid]-baseline[valid]) * (rays @ geometry.n)
    shift = float(np.median(heights))
    report.update(current_surface_shift_m=shift, current_p90_shift_m=float(np.percentile(heights,90)))
    if abs(shift) > .003:
        return dict(report, reason='support_height_changed')
    # Remove global exposure change, retain per-channel deviations and localized shadows.
    difference = rgb.astype(float) - original
    difference -= np.median(difference[mask], axis=0)
    photometric = np.max(abs(difference), axis=2)
    image_change = mask & (photometric > 25)
    raised = np.zeros(mask.shape, dtype=bool)
    raised[valid] = heights > .006
    correlated = raised & (photometric > 12)
    def largest(flag):
        count, _, stat, _ = cv2.connectedComponentsWithStats(flag.astype(np.uint8), connectivity=8)
        return int(stat[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    values = [largest(flag) for flag in (image_change, raised, correlated, mask & ~valid)]
    report.update(zip(('appearance_component_px','elevated_component_px',
                       'correlated_component_px','missing_component_px'), values))
    if any(value >= limit for value, limit in zip(values, (6, 6, 4, 6))):
        return dict(report, reason='changed_or_occluded_fingertip_approach')
    return dict(report, verified=True, reason='surveyed_support_reobserved')

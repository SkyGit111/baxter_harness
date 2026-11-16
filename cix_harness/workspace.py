"""Read-only workspace geometry for visual planning; never an actuator gate."""
import math

from .contract import vector
from .pi05_contract import quaternion, rotate
from .primitives import table_height, workspace_polygon, validate_table_geometry


def workspace_metadata(profile, intrinsics, project_pixel):
    """Project the allowed convex footprint onto the actual support plane.

    project_pixel(intrinsics, camera_point) uses the camera's distortion model.
    UVs refer to the unmodified color frame, matching ground_pixel normalization.
    Elevated objects can project outside this plane polygon through parallax.
    """
    low = vector(profile['tcp_workspace_min_m'], 3, 'workspace minimum')
    high = vector(profile['tcp_workspace_max_m'], 3, 'workspace maximum')
    validate_table_geometry(profile)
    table_z = float(profile.get('table_z_m',table_height(0,0,profile)))
    if any(a >= b for a, b in zip(low, high)) or not math.isfinite(table_z):
        raise ValueError('Invalid workspace projection bounds')
    result = {
        'frame': 'base', 'tcp_min_m': low, 'tcp_max_m': high, 'table_z_m': table_z,
        'profile_calibrated': profile.get('calibrated') is True,
        'calibration_note': str(profile.get('notes', 'No calibration accuracy note supplied.'))[:300],
        'table_reference': {'base_z_m': table_z if profile.get('table_plane') is None else None,
                            'plane': profile.get('table_plane',{'normal':[0,0,1],'offset':-table_z}),
                            'polygon_uv': None, 'status': 'unavailable'},
    }
    q = quaternion(profile['camera_to_base_xyzw'])
    inverse = [-q[0], -q[1], -q[2], q[3]]
    translation = vector(profile['camera_to_base_translation_m'], 3, 'camera translation')
    width, height = intrinsics.width, intrinsics.height
    if width <= 1 or height <= 1:
        raise ValueError('Camera projection requires full color frame dimensions')
    corners = [[x,y,table_height(x,y,profile)] for x,y in workspace_polygon(profile)]
    result['table_reference']['polygon_base_m']=corners
    camera_points = [rotate(inverse, [v - t for v, t in zip(point, translation)]) for point in corners]
    if any(point[2] <= 1e-6 for point in camera_points):
        result['table_reference']['status'] = 'not_in_front_of_camera'
        return result
    polygon = []
    for point in camera_points:
        pixel = vector(project_pixel(intrinsics, point), 2, 'projected table pixel')
        polygon.append([round(pixel[0] / (width - 1), 5), round(pixel[1] / (height - 1), 5)])
    # Preserve off-image coordinates. Clamping each corner would change the region.
    status = 'in_image' if all(0 <= value <= 1 for point in polygon for value in point) else 'extends_outside_image'
    result['table_reference'].update(polygon_uv=polygon, status=status)
    return result

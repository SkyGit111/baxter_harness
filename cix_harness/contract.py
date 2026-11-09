import base64
import math

ACTION_SPEC = "bridge_orig.xyz_m.rpy_rad.gripper_open01.v1"
MAX_IMAGE = 4 * 1024 * 1024


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("{} must have {} elements".format(name, length))
    if any(type(n) not in (int, float) or not math.isfinite(n) for n in value):
        raise ValueError("{} must contain finite numbers".format(name))
    return list(value)


def action7(values):
    values = vector(values, 7, "action")
    if not 0 <= values[6] <= 1:
        raise ValueError("Expected normalized open gripper command in [0,1]")
    return values


def image_data(raw):
    if not 0 < len(raw) <= MAX_IMAGE:
        raise ValueError("Empty or oversized camera image")
    if raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        raise ValueError("Expected JPEG or PNG image")
    return "data:{};base64,".format(mime) + base64.b64encode(raw).decode("ascii")


def image_bytes(url):
    if not isinstance(url, str) or len(url) > MAX_IMAGE * 4 // 3 + 32:
        raise ValueError("Invalid image data URL")
    header, _, payload = url.partition(",")
    if header not in ("data:image/jpeg;base64", "data:image/png;base64"):
        raise ValueError("Expected embedded JPEG or PNG")
    raw = base64.b64decode(payload, validate=True)
    if not 0 < len(raw) <= MAX_IMAGE:
        raise ValueError("Empty or oversized image")
    return raw


def instruction(text):
    if not isinstance(text, str) or not 0 < len(text.strip()) <= 512:
        raise ValueError("Expected a nonempty instruction of at most 512 characters")
    return text.strip()

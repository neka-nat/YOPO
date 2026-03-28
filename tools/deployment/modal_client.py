import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Send a pose inference request to the YOPO Modal endpoint.')
    parser.add_argument('endpoint', help='Modal endpoint URL')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--fx', type=float, required=True, help='Camera fx')
    parser.add_argument('--fy', type=float, required=True, help='Camera fy')
    parser.add_argument('--cx', type=float, required=True, help='Camera cx')
    parser.add_argument('--cy', type=float, required=True, help='Camera cy')
    parser.add_argument(
        '--score-thr', type=float, default=0.2, help='Score threshold')
    parser.add_argument(
        '--json-out', type=str, default=None, help='Optional JSON output path')
    parser.add_argument(
        '--render-out',
        type=str,
        default=None,
        help='Optional rendered image output path')
    parser.add_argument(
        '--draw-axis',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Draw pose axes on the rendered image')
    parser.add_argument(
        '--draw-cuboid',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Draw size-based 3D cuboids on the rendered image')
    return parser.parse_args()


def _request_prediction(endpoint: str, payload: dict) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(
            f'HTTP {exc.code} from endpoint:\n{error_body}') from exc
    return json.loads(body)


def _load_intrinsic_matrix(result: dict, fallback_intrinsic: list[float]):
    import numpy as np

    intrinsic = result.get('intrinsic', fallback_intrinsic)
    if isinstance(intrinsic, list):
        if len(intrinsic) == 4:
            fx, fy, cx, cy = intrinsic
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32)
        if len(intrinsic) == 9:
            return np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
        if len(intrinsic) == 3 and all(isinstance(row, list) for row in intrinsic):
            return np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
    raise ValueError(f'Unsupported intrinsic format: {intrinsic!r}')


def _color_for_label(label: int) -> tuple[int, int, int]:
    palette = [
        (80, 110, 255),
        (80, 200, 120),
        (255, 170, 70),
        (200, 120, 255),
        (110, 220, 220),
        (120, 120, 255),
        (255, 200, 120),
        (150, 230, 120),
        (255, 120, 180),
        (220, 220, 120),
    ]
    return palette[label % len(palette)]


def _project_points(points_3d, intrinsic):
    import numpy as np

    points_3d = np.asarray(points_3d, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)

    z = points_3d[:, 2]
    valid = z > 1e-6
    projected = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float32)
    if not valid.any():
        return projected, valid

    camera = (intrinsic @ points_3d[valid].T).T
    projected_valid = camera[:, :2] / camera[:, 2:3]
    projected[valid] = projected_valid
    return projected, valid


def _transform_points(points_local, transform):
    import numpy as np

    points_local = np.asarray(points_local, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (rotation @ points_local.T).T + translation


def _draw_label(cv2, image, text: str, origin: tuple[int, int],
                color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(y - height - baseline - 6, 0)
    cv2.rectangle(
        image,
        (x, top),
        (x + width + 6, top + height + baseline + 6),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, top + height + 1),
        font,
        scale,
        (15, 15, 15),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_bbox(cv2, image, detection: dict) -> None:
    bbox = detection.get('bbox')
    if bbox is None:
        return
    label = int(detection.get('label', 0))
    color = _color_for_label(label)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

    class_name = detection.get('class_name', f'class {label}')
    score = float(detection.get('score', 0.0))
    text = f'{class_name}: {score:.2f}'
    _draw_label(cv2, image, text, (max(x1, 0), max(y1, 0)), color)


def _draw_axis(cv2, image, detection: dict, intrinsic) -> None:
    import numpy as np

    transform = detection.get('transform')
    if transform is None:
        return

    size = detection.get('size')
    if size is not None and len(size) == 3:
        width, height, depth = [float(v) for v in size]
        local_axis = np.array(
            [[0.0, 0.0, 0.0], [width / 2, 0.0, 0.0], [0.0, height / 2, 0.0],
             [0.0, 0.0, depth / 2]],
            dtype=np.float32,
        )
    else:
        transform_arr = np.asarray(transform, dtype=np.float32).reshape(4, 4)
        distance = float(np.linalg.norm(transform_arr[:3, 3]))
        axis_length = 0.05 * (1000.0 if distance > 100.0 else 1.0)
        local_axis = np.array(
            [[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], [0.0, axis_length, 0.0],
             [0.0, 0.0, axis_length]],
            dtype=np.float32,
        )

    axis_cam = _transform_points(local_axis, transform)
    axis_2d, valid = _project_points(axis_cam, intrinsic)
    if not valid[0]:
        return

    origin = tuple(np.round(axis_2d[0]).astype(int))
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for idx, color in zip(range(1, 4), axis_colors):
        if not valid[idx]:
            continue
        endpoint = tuple(np.round(axis_2d[idx]).astype(int))
        cv2.line(image, origin, endpoint, color, thickness=2, lineType=cv2.LINE_AA)


def _draw_cuboid(cv2, image, detection: dict, intrinsic) -> None:
    import numpy as np

    transform = detection.get('transform')
    size = detection.get('size')
    if transform is None or size is None or len(size) != 3:
        return

    width, height, depth = [float(v) for v in size]
    corners_local = np.array(
        [
            [-width / 2, -height / 2, -depth / 2],
            [width / 2, -height / 2, -depth / 2],
            [width / 2, height / 2, -depth / 2],
            [-width / 2, height / 2, -depth / 2],
            [-width / 2, -height / 2, depth / 2],
            [width / 2, -height / 2, depth / 2],
            [width / 2, height / 2, depth / 2],
            [-width / 2, height / 2, depth / 2],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    corners_cam = _transform_points(corners_local, transform)
    corners_2d, valid = _project_points(corners_cam, intrinsic)
    color = (85, 221, 85)
    for start, end in edges:
        if not (valid[start] and valid[end]):
            continue
        p1 = tuple(np.round(corners_2d[start]).astype(int))
        p2 = tuple(np.round(corners_2d[end]).astype(int))
        cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)


def render_prediction(image_path: str, result: dict, out_path: str,
                      draw_axis: bool, draw_cuboid: bool) -> None:
    import cv2

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed to read image: {image_path}')

    intrinsic = _load_intrinsic_matrix(
        result, result.get('intrinsic', [0.0, 0.0, 0.0, 0.0]))
    detections = sorted(
        result.get('detections', []),
        key=lambda row: float(row.get('score', 0.0)),
        reverse=True,
    )

    for detection in detections:
        _draw_bbox(cv2, image, detection)
        if draw_cuboid:
            _draw_cuboid(cv2, image, detection, intrinsic)
        if draw_axis:
            _draw_axis(cv2, image, detection, intrinsic)

    if not detections:
        _draw_label(cv2, image, 'No detections', (12, 24), (220, 220, 220))

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f'Failed to write rendered image: {out_path}')


def main():
    args = parse_args()
    image_bytes = Path(args.image_path).read_bytes()
    payload = {
        'image_base64': base64.b64encode(image_bytes).decode('ascii'),
        'intrinsic': [args.fx, args.fy, args.cx, args.cy],
        'score_thr': args.score_thr,
    }
    result = _request_prediction(args.endpoint, payload)

    formatted = json.dumps(result, indent=2, ensure_ascii=False)
    print(formatted)

    if args.json_out is not None:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(formatted + '\n', encoding='utf-8')

    if args.render_out is not None:
        render_prediction(
            image_path=args.image_path,
            result=result,
            out_path=args.render_out,
            draw_axis=args.draw_axis,
            draw_cuboid=args.draw_cuboid,
        )
        print(f'Rendered image saved to {args.render_out}')


if __name__ == '__main__':
    main()

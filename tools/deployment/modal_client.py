import argparse
import base64
import json
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
    return parser.parse_args()


def main():
    args = parse_args()
    image_bytes = Path(args.image_path).read_bytes()
    payload = {
        'image_base64': base64.b64encode(image_bytes).decode('ascii'),
        'intrinsic': [args.fx, args.fy, args.cx, args.cy],
        'score_thr': args.score_thr,
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request) as response:
        body = response.read().decode('utf-8')
    print(body)


if __name__ == '__main__':
    main()

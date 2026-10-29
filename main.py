import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--initial-x', type=float, required=True)
    parser.add_argument('--initial-y', type=float, required=True)
    parser.add_argument('--start-ns', type=int)
    parser.add_argument('--output', type=Path, default=Path('predictions.jsonl'))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--context-workers', type=int, default=4)
    args = parser.parse_args()
    import torch
    from full_online import Stream
    from full_online import load_assets
    torch.set_num_threads(args.threads)
    os.environ['YOMO_MOBILE_ONLINE_CONTEXT_WORKERS'] = str(args.context_workers)
    assets = load_assets(args.assets.resolve())
    stream = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.events.open() as source, args.output.open('w') as output:
            for line in source:
                if not line.strip():
                    continue
                event = json.loads(line)
                if stream is None:
                    start = args.start_ns if args.start_ns is not None else int(event['monotonic_timestamp_ns'])
                    stream = Stream(assets, [args.initial_x, args.initial_y], start)
                prediction = stream.consume(event)
                if prediction is not None:
                    output.write(json.dumps(prediction) + '\n')
                    output.flush()
    finally:
        if stream is not None:
            stream.close()
if __name__ == '__main__':
    main()

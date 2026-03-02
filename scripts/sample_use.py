"""CLI wrapper for the reusable small-molecule design API."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocketxmol import SamplingRequest, run_sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_task",
        type=str,
        default="configs/sample/examples/sbdd.yml",
        help="task config",
    )
    parser.add_argument(
        "--config_model",
        type=str,
        default="configs/sample/pxm.yml",
        help="model config",
    )
    parser.add_argument("--outdir", type=str, default="./outputs_use")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="batch size; by default use the value in the config file",
    )
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="num_workers for dataloader; by default use the value in the train config file",
    )
    args = parser.parse_args()

    config_request = SamplingRequest(
        config_task=args.config_task,
        config_model=args.config_model,
        outdir=args.outdir,
        device=args.device,
        batch_size=None if args.batch_size == 0 else args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )
    run_sampling(request=config_request)

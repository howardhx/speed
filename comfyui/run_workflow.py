#!/usr/bin/env python
"""Headless runner for the SPEED ComfyUI node — no UI building required.

Submits an API-format workflow JSON to a running ComfyUI server's ``/prompt``
endpoint, waits for it to finish, and reports the saved image path(s).

Usage
-----
    # 1. Start ComfyUI (in a ComfyUI env), e.g.:
    #    CUDA_VISIBLE_DEVICES=0 python /path/to/ComfyUI/main.py --port 8188
    # 2. Run this against it:
    python run_workflow.py --server http://127.0.0.1:8188 \
                           --workflow workflow_flux_api.json \
                           --prompt "a corgi puppy sitting in a field" --seed 42

The default workflow is ``workflow_flux_api.json`` (FLUX.1-dev +
SamplerSPEED, dct, scales 0.5,1.0, delta 0.01). ``--prompt``/``--seed`` patch
the matching nodes; everything else comes from the JSON.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path


def _post(server: str, path: str, payload: dict) -> dict:
    """POST ``payload`` as JSON to ``server+path`` and return the parsed reply.

    Args:
    - server: Base URL of the ComfyUI server, e.g. ``http://127.0.0.1:8188``.
    - path: Endpoint path, e.g. ``/prompt``.
    - payload: JSON-serialisable request body.

    Returns:
    - The parsed JSON response.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(server + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def _get(server: str, path: str) -> dict:
    """GET ``server+path`` and return the parsed JSON reply.

    Args:
    - server: Base URL of the ComfyUI server.
    - path: Endpoint path, e.g. ``/history/<id>``.

    Returns:
    - The parsed JSON response.
    """
    return json.load(urllib.request.urlopen(server + path, timeout=60))


def main() -> None:
    """Submit the workflow, poll until done, and print the output image paths.

    Args:
    - None (reads from ``sys.argv``).

    Returns:
    - None; prints the saved image filenames.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--workflow", default=str(Path(__file__).with_name("workflow_flux_api.json")))
    p.add_argument("--prompt", default=None, help="Override the text prompt.")
    p.add_argument("--seed", type=int, default=None, help="Override noise + spectral seed.")
    p.add_argument("--timeout", type=int, default=600, help="Seconds to wait for completion.")
    args = p.parse_args()

    wf = json.loads(Path(args.workflow).read_text())
    if args.prompt is not None:
        wf["pos"]["inputs"]["text"] = args.prompt
    if args.seed is not None:
        wf["noise"]["inputs"]["noise_seed"] = args.seed
        wf["speed"]["inputs"]["seed"] = args.seed

    client_id = uuid.uuid4().hex
    resp = _post(args.server, "/prompt", {"prompt": wf, "client_id": client_id})
    prompt_id = resp["prompt_id"]
    print(f"submitted prompt_id={prompt_id}")

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        hist = _get(args.server, f"/history/{prompt_id}")
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            imgs = [im for node in outputs.values() for im in node.get("images", [])]
            if imgs:
                print(f"done — {len(imgs)} image(s):")
                for im in imgs:
                    print(f"  output/{im.get('subfolder','')}/{im['filename']}".replace("//", "/"))
                return
            status = hist[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                raise SystemExit(f"workflow errored: {json.dumps(status)[:500]}")
        time.sleep(2)
    raise SystemExit("timed out waiting for the workflow to finish.")


if __name__ == "__main__":
    main()

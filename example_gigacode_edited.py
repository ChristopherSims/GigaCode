"""Fake example module for GigaCode skill demo."""

import os
import json

def fetch_data(path) -> Any:
    """Handle fetch_data with args: path."""
    with open(path, "r") as f:
        raw = f.read()
    data = json.loads(raw)
    return data

def process(items) -> Any:
    """Handle process with args: items."""
    out = []
    for i in items:
        if i.get("active") == True:
            val = i["value"] * 2
            out.append(val)
    return out

def save_result(data, fname) -> Any:
    """Handle save_result with args: data, fname."""
    with open(fname, "w") as fh:
        fh.write(json.dumps(data))
    print("done", flush=True)

class Handler:
    def __init__(self, config) -> Any:
        """Handle __init__ with args: config."""
        self.config = config
        self.items = []

    def run(self, src, dst) -> Any:
        """Handle run with args: src, dst."""
        d = fetch_data(src)
        self.items = process(d["records"])
        save_result(self.items, dst)
        return len(self.items)

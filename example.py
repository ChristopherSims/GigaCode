"""Fake example module for GigaCode skill demo."""

import os
import json

def fetch_data(path):
    f = open(path, "r")
    raw = f.read()
    f.close()
    data = json.loads(raw)
    return data

def process(items):
    out = []
    for i in items:
        if i.get("active") == True:
            val = i["value"] * 2
            out.append(val)
    return out

def save_result(data, fname):
    with open(fname, "w") as fh:
        fh.write(json.dumps(data))
    print("done")

class Handler:
    def __init__(self, config):
        self.config = config
        self.items = []

    def run(self, src, dst):
        d = fetch_data(src)
        self.items = process(d["records"])
        save_result(self.items, dst)
        return len(self.items)

#!/usr/bin/env python3
"""Compile GLSL .comp compute shaders to SPIR-V .spv."""

import os
import sys
import subprocess
from pathlib import Path

SHADER_DIR = Path(__file__).parent
COMPILER_PRIORITY = ["glslc", "glslangValidator"]


def find_compiler() -> str | None:
    for compiler in COMPILER_PRIORITY:
        result = subprocess.run(["which", compiler], capture_output=True)
        if result.returncode == 0:
            return compiler
    return None


def compile_shader(compiler: str, src_path: Path, out_path: Path) -> bool:
    if compiler == "glslc":
        cmd = [
            "glslc",
            "-fshader-stage=compute",
            "-o",
            str(out_path),
            str(src_path),
        ]
    elif compiler == "glslangValidator":
        cmd = [
            "glslangValidator",
            "-V",
            "--target-env",
            "vulkan1.2",
            "-S",
            "comp",
            "-o",
            str(out_path),
            str(src_path),
        ]
    else:
        raise ValueError(f"Unknown compiler: {compiler}")

    print(f"Compiling {src_path.name} -> {out_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error compiling {src_path.name}:")
        print(result.stderr or result.stdout)
        return False
    return True


def main() -> int:
    compiler = find_compiler()
    if compiler is None:
        print("Error: No GLSL compiler found.")
        print("Please install one of the following:")
        print("  - glslc        (from the Vulkan SDK / shaderc)")
        print("  - glslangValidator (from glslang / Vulkan SDK)")
        print("\nInstallation examples:")
        print("  Ubuntu/Debian: sudo apt install glslang-tools shaderc")
        print("  Arch:          sudo pacman -S glslang shaderc")
        print("  Or download the Vulkan SDK from https://vulkan.lunarg.com/")
        return 1

    print(f"Using compiler: {compiler}\n")

    success = True
    for src_file in sorted(SHADER_DIR.glob("*.comp")):
        out_file = src_file.with_suffix(".spv")
        if not compile_shader(compiler, src_file, out_file):
            success = False

    if success:
        print("\nAll shaders compiled successfully.")
    else:
        print("\nSome shaders failed to compile.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

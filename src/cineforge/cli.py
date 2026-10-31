"""Offline package entry point."""

import argparse

from cineforge import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="Cineforge AI filmmaking scaffold")
    parser.add_argument("--version", action="version", version=f"cineforge {__version__}")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()

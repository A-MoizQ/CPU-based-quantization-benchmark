from pathlib import Path


def main():
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found. The checked-in notebooks are expected under notebooks/.")
    print("Notebook files already exist:")
    for notebook in notebooks:
        print(f"- {notebook}")


if __name__ == "__main__":
    main()

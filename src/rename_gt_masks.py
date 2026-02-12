import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename *_gt.nii.gz to *.nii.gz")
    parser.add_argument("--gt_dir", required=True, help="Directory containing *_gt.nii.gz files.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned renames without modifying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_dir = args.gt_dir
    renamed = 0

    for name in sorted(os.listdir(gt_dir)):
        if not name.endswith("_gt.nii.gz"):
            continue
        old_path = os.path.join(gt_dir, name)
        new_name = name.replace("_gt.nii.gz", ".nii.gz")
        new_path = os.path.join(gt_dir, new_name)
        if os.path.exists(new_path):
            print(f"Skip (target exists): {new_name}")
            continue
        print(f"{name} -> {new_name}")
        if not args.dry_run:
            os.rename(old_path, new_path)
            renamed += 1

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Renamed {renamed} files.")


if __name__ == "__main__":
    main()

#! /usr/bin/env python3

import os
from threading import Semaphore
from argparse import ArgumentParser
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor, wait
from urllib import request, error
from tqdm import tqdm


def download_hook(t):
    last_b = [0]

    def update(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update


def download_file(url: str, save_dir: str):
    filename = url.split("/")[-1]
    exist = os.path.exists(os.path.join(save_dir, filename))
    if exist:
        with tqdm(
            total=1,
            desc=f"File exists. Pass downloading {filename}",
            leave=False,
        ) as t:
            return

    os.makedirs(save_dir, exist_ok=True)
    with tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=f"Download {filename}",
        leave=False,
    ) as t:
        opener = request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        request.install_opener(opener)
        request.urlretrieve(
            url,
            filename=os.path.join(save_dir, filename),
            reporthook=download_hook(t),
            data=None,
        )


def extract_and_remove(filename, dst, args):
    with zipfile.ZipFile(filename, "r") as zf:
        for member in tqdm(
            zf.infolist(),
            desc=f"Extract file: {filename}",
        ):
            try:
                zf.extract(member, dst)
            except zipfile.error as e:
                pass

    if args.remove_zip:
        with tqdm(
            total=1,
            desc=f"Remove file: {filename}",
            leave=False,
        ) as t:
            os.remove(filename)
            t.update()


def download_all(bop, args, sem):
    with sem:
        try:
            # download and extract base file
            download_file(bop["base_url"], args.save_dir)
            filename = os.path.join(args.save_dir, bop["base"])
            extract_and_remove(filename, args.save_dir, args)

            # download and extract model file
            download_file(bop["model_url"], args.save_dir)
            filename = os.path.join(args.save_dir, bop["model"])
            extract_and_remove(
                filename, os.path.join(args.save_dir, bop["dataset_name"]), args
            )

            if args.download_images:
                for img, img_url in zip(bop["images"], bop["image_urls"]):
                    zip_path = os.path.join(args.save_dir, bop["dataset_name"])
                    download_file(img_url, zip_path)
                    filename = os.path.join(zip_path, img)
                    if args.extract_images:
                        extract_and_remove(
                            filename,
                            os.path.join(args.save_dir, bop["dataset_name"]),
                            args,
                        )

        except error.HTTPError:
            print(f"Download Failed: {bop['base']}")


def main():
    parser = ArgumentParser("download bop")
    parser.add_argument("save_dir", type=str, help="output directory")
    parser.add_argument(
        "--whitelist", nargs="+", help="An whitelist of dataset names for downloading"
    )
    parser.add_argument(
        "--remove_zip", action="store_true", help="remove zip files after extract"
    )
    parser.add_argument(
        "--download_images", action="store_true", help="download all image data"
    )
    parser.add_argument(
        "--extract_images",
        action="store_true",
        help="Extract downloaded images. not recommended. The contents might overwrite the files with the same name.",
    )
    parser.add_argument(
        "--num_thread", type=int, default=8, help="number of worker thread"
    )
    args = parser.parse_args()
    dirname = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dirname, "bop_all.json")

    with open(filename, "r") as f:
        bop_all = json.load(f)

    if args.whitelist:
        bop_all = [bop for bop in bop_all if bop["dataset_name"] in args.whitelist]

    with ThreadPoolExecutor() as executor:
        sem = Semaphore(args.num_thread)
        futures = []
        for bop in bop_all:
            future = executor.submit(download_all, bop, args, sem)
            futures.append(future)
        wait(futures)


if __name__ == "__main__":
    main()

"""Download datasets from public collections (currently the Broad Bioimage Benchmark Collection)."""

from __future__ import annotations

import re
import shutil
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import unquote, urljoin, urlparse

BBBC_PAGE_URL = "https://bbbc.broadinstitute.org/BBBC{number}"
BBBC_DATA_HOST = "data.broadinstitute.org"
DOWNLOAD_EXTENSIONS = (".zip", ".csv", ".tif", ".tiff", ".png", ".txt", ".xlsx", ".xls", ".mat", ".json", ".tgz", ".gz")
CHUNK_SIZE = 1 << 20
USER_AGENT = "image-fidelity/0.1"

Fetcher = Callable[[str], bytes]


def normalize_bbbc_id(dataset_id: str | int) -> str:
    """Return the canonical three-digit BBBC number, e.g. ``38`` / ``"bbbc038"`` -> ``"038"``."""
    text = str(dataset_id).strip().lower().removeprefix("bbbc")
    if not text.isdigit() or int(text) == 0:
        raise ValueError(f"Invalid BBBC dataset id {dataset_id!r}; expected a number such as 038")
    return f"{int(text):03d}"


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def list_bbbc_files(dataset_id: str | int, fetch: Fetcher = _fetch) -> list[str]:
    """List the downloadable file URLs advertised on the page of a BBBC dataset."""
    number = normalize_bbbc_id(dataset_id)
    page_url = BBBC_PAGE_URL.format(number=number)
    try:
        html = fetch(page_url).decode("utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"Could not fetch BBBC{number} page {page_url}: {exc}") from exc

    parser = _LinkParser()
    parser.feed(html)
    prefix = f"/bbbc/BBBC{number}/".lower()
    urls: list[str] = []
    for href in parser.links:
        url = urljoin(page_url, href)
        parsed = urlparse(url)
        if (
            parsed.hostname == BBBC_DATA_HOST
            and parsed.path.lower().startswith(prefix)
            and parsed.path.lower().endswith(DOWNLOAD_EXTENSIONS)
            and url not in urls
        ):
            urls.append(url)
    if not urls:
        raise RuntimeError(f"No downloadable files found for BBBC{number} at {page_url}")
    return urls


def _download_file(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, CHUNK_SIZE)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if not (target / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in {archive.name}: {member.filename}")
        zf.extractall(target)


def download_bbbc(
    dataset_id: str | int,
    output_dir: str | Path,
    files: Sequence[str] | None = None,
    extract: bool = True,
    keep_archives: bool = True,
    force: bool = False,
    downloader: Callable[[str, Path], None] = _download_file,
    fetch: Fetcher = _fetch,
) -> Path:
    """Download a BBBC dataset into ``<output_dir>/bbbc<NNN>/``.

    Every file listed on the dataset page is fetched, or only those whose name contains one
    of the ``files`` substrings. Zip archives are extracted into ``<dataset_dir>/<archive stem>/``.
    Existing files are skipped unless ``force`` is set. Returns the dataset directory.
    """
    number = normalize_bbbc_id(dataset_id)
    urls = list_bbbc_files(number, fetch=fetch)
    if files:
        wanted = [name.lower() for name in files]
        urls = [url for url in urls if any(name in Path(unquote(urlparse(url).path)).name.lower() for name in wanted)]
        if not urls:
            raise ValueError(f"No file of BBBC{number} matches {list(files)}")

    dataset_dir = Path(output_dir) / f"bbbc{number}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        name = Path(unquote(urlparse(url).path)).name
        destination = dataset_dir / name
        is_zip = name.lower().endswith(".zip")
        extracted = dataset_dir / destination.stem
        if is_zip and extract and extracted.is_dir() and not force:
            print(f"[bbbc{number}] {name}: already extracted, skipping")
            continue
        if force or not destination.exists():
            print(f"[bbbc{number}] downloading {url}")
            downloader(url, destination)
        else:
            print(f"[bbbc{number}] {name}: already downloaded")
        if is_zip and extract:
            if extracted.is_dir():
                shutil.rmtree(extracted)
            print(f"[bbbc{number}] extracting {name}")
            _extract_zip(destination, extracted)
            if not keep_archives:
                destination.unlink()
    return dataset_dir


COLLECTIONS: dict[str, Callable[..., Path]] = {"bbbc": download_bbbc}


def download_datasets(
    collection: str,
    dataset_ids: Iterable[str | int],
    output_dir: str | Path,
    **options,
) -> list[Path]:
    """Download several datasets of one collection; returns their directories."""
    try:
        downloader = COLLECTIONS[collection.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported collection {collection!r}; available: {', '.join(sorted(COLLECTIONS))}") from exc
    return [downloader(dataset_id, output_dir, **options) for dataset_id in dataset_ids]

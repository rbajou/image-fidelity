import io
import zipfile

import pytest
from image_fidelity.cli.cli import translate_arguments
from image_fidelity.datasets.download import download_bbbc, normalize_bbbc_id

PAGE = b"""<a href="https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip">z</a>
<a href="https://data.broadinstitute.org/bbbc/BBBC038/labels.csv">c</a>
<a href="https://example.com/other.zip">x</a>"""


def _zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("sample/images/a.png", b"x")
    return buffer.getvalue()


def _downloader(url, destination):
    destination.write_bytes(_zip_bytes() if url.endswith(".zip") else b"a,b")


def test_normalize_bbbc_id():
    assert normalize_bbbc_id("038") == "038"
    assert normalize_bbbc_id(38) == "038"
    assert normalize_bbbc_id("BBBC010") == "010"
    with pytest.raises(ValueError):
        normalize_bbbc_id("abc")


def test_download_bbbc_extracts_and_filters(tmp_path):
    root = download_bbbc("038", tmp_path, fetch=lambda url: PAGE, downloader=_downloader)
    assert root == tmp_path / "bbbc038"
    assert (root / "stage1_train" / "sample" / "images" / "a.png").is_file()
    assert (root / "labels.csv").is_file()

    only_csv = download_bbbc("38", tmp_path / "b", files=["labels"], fetch=lambda url: PAGE, downloader=_downloader)
    assert not (only_csv / "stage1_train.zip").exists()


def test_translate_download_arguments():
    assert translate_arguments(["download", "bbbc", "038", "038"][:3]) == [
        "task=download",
        "task.collection=bbbc",
        "task.datasets=['038']",
    ]

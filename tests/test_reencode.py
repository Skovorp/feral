import os
import shutil
import subprocess
import tempfile

import pytest

from feral.reencode_videos import build_scale_filter, process_file


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


_needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH")


def _ffprobe_dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def _make_test_video(path, width, height, seconds=1):
    """Create a short mp4 of the given dimensions using ffmpeg's testsrc."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size={width}x{height}:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "ultrafast", path],
        check=True, capture_output=True,
    )


# ===================================================================
# build_scale_filter
# ===================================================================

class TestBuildScaleFilter:
    def test_default_512(self):
        f = build_scale_filter(512)
        assert "min(512,iw)" in f
        assert "min(512,ih)" in f
        assert "lanczos" in f

    def test_custom_value(self):
        f = build_scale_filter(128)
        assert "min(128,iw)" in f
        assert "min(128,ih)" in f

    def test_branches_on_aspect(self):
        # The filter must pick which side gets capped depending on iw vs ih.
        f = build_scale_filter(256)
        assert "if(gt(iw,ih)" in f
        # Even dimension rounding for libx264.
        assert "-2" in f


# ===================================================================
# process_file (ffmpeg integration)
# ===================================================================

@_needs_ffmpeg
class TestProcessFileSmallestSide:
    """End-to-end test that process_file downsizes the smallest side correctly
    and preserves aspect ratio."""

    def test_downsize_landscape(self, tmp_path):
        src = tmp_path / "in" / "wide.mp4"
        src.parent.mkdir()
        _make_test_video(str(src), width=640, height=480)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        ok = process_file((str(src), str(out_dir), "ffmpeg", 240))
        assert ok
        out = out_dir / "wide.mp4"
        assert out.exists()
        w, h = _ffprobe_dims(str(out))
        # smallest side (height) becomes 240, width scales to ~320 (rounded to even)
        assert h == 240
        assert w in (320,)  # 640 * 240/480 = 320

    def test_downsize_portrait(self, tmp_path):
        src = tmp_path / "in" / "tall.mp4"
        src.parent.mkdir()
        _make_test_video(str(src), width=240, height=320)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        ok = process_file((str(src), str(out_dir), "ffmpeg", 120))
        assert ok
        w, h = _ffprobe_dims(str(out_dir / "tall.mp4"))
        # smallest side (width) becomes 120, height scales to ~160
        assert w == 120
        assert h == 160

    def test_no_upscale_when_already_smaller(self, tmp_path):
        src = tmp_path / "in" / "small.mp4"
        src.parent.mkdir()
        _make_test_video(str(src), width=320, height=240)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Request 512: video is already smaller on both sides, so it should stay.
        ok = process_file((str(src), str(out_dir), "ffmpeg", 512))
        assert ok
        w, h = _ffprobe_dims(str(out_dir / "small.mp4"))
        assert (w, h) == (320, 240)

    def test_square_stays_square(self, tmp_path):
        src = tmp_path / "in" / "sq.mp4"
        src.parent.mkdir()
        _make_test_video(str(src), width=256, height=256)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        ok = process_file((str(src), str(out_dir), "ffmpeg", 128))
        assert ok
        w, h = _ffprobe_dims(str(out_dir / "sq.mp4"))
        assert (w, h) == (128, 128)

    def test_default_smallest_side_when_omitted(self, tmp_path):
        """Calling process_file with a 3-tuple (legacy signature) should still work
        and use the default smallest side."""
        src = tmp_path / "in" / "v.mp4"
        src.parent.mkdir()
        _make_test_video(str(src), width=200, height=100)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        ok = process_file((str(src), str(out_dir), "ffmpeg"))
        assert ok
        w, h = _ffprobe_dims(str(out_dir / "v.mp4"))
        # 100 < 512, so no upscale, dims unchanged.
        assert (w, h) == (200, 100)

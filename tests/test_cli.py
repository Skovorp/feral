"""Tests for the `feral` CLI (feral/cli.py).

These cover argument parsing / cross-arg validation in ``main()`` and the four
subcommand handlers. The handlers import their heavy dependencies lazily
(``torch``, ``wandb``, ``feral.train`` …) *inside* the function body, so every
test here stubs those modules via ``sys.modules`` — the suite runs with no GPU
and without torch/wandb installed.
"""

import argparse
import sys
import types

import pytest
import yaml

import feral.cli as cli


# ── _load_default_config ──────────────────────────────────────────────────────

class TestLoadDefaultConfig:
    def test_returns_dict_with_core_sections(self):
        cfg = cli._load_default_config()
        assert isinstance(cfg, dict)
        for section in ("data", "model", "training"):
            assert section in cfg, f"missing {section!r} in packaged default_config.yaml"

    def test_returns_independent_copies(self):
        # Each call re-reads the packaged file, so mutating one result must not
        # leak into the next (the handlers mutate cfg in place).
        first = cli._load_default_config()
        first["data"]["prefix"] = "MUTATED"
        second = cli._load_default_config()
        assert second["data"].get("prefix") != "MUTATED"


# ── main(): parsing + validation ──────────────────────────────────────────────

@pytest.fixture
def dispatch(monkeypatch):
    """Replace every subcommand handler with a recorder so main() parses/validates
    argv and dispatches without running the real (torch-heavy) handlers."""
    captured = {}

    def make(name):
        def _stub(args):
            captured["name"] = name
            captured["args"] = args
        return _stub

    for handler in ("_cmd_train", "_cmd_train_config", "_cmd_infer", "_cmd_reencode"):
        monkeypatch.setattr(cli, handler, make(handler))
    return captured


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["feral", *argv])
    cli.main()


class TestMainParsing:
    def test_no_command_exits(self, monkeypatch, dispatch):
        # subparsers required=True -> argparse exits (code 2) when no command given.
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, [])

    def test_unknown_command_exits(self, monkeypatch, dispatch):
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["frobnicate"])

    def test_train_valid_dispatch_defaults(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        _run_main(monkeypatch, ["train", str(vids), str(labels)])
        assert dispatch["name"] == "_cmd_train"
        args = dispatch["args"]
        assert args.video_folder == str(vids)
        assert args.label_json_path == str(labels)
        assert args.mode is None
        assert args.no_wandb is False
        assert args.public_wandb is False
        assert args.gradient_checkpointing is False

    def test_train_parses_flags(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_text("x")
        _run_main(monkeypatch, [
            "train", str(vids), str(labels),
            "--mode", "max", "--resolution", "512", "--no-wandb",
            "-c", str(ckpt), "--part_subsample", "0.5",
            "--subsample_keep_rare_threshold", "0.1",
            "--gradient-checkpointing",
        ])
        args = dispatch["args"]
        assert args.mode == "max"
        assert args.resolution == 512
        assert args.no_wandb is True
        assert args.checkpoint == str(ckpt)
        assert args.part_subsample == 0.5
        assert args.subsample_keep_rare_threshold == 0.1
        assert args.gradient_checkpointing is True

    def test_train_bad_mode_choice_exits(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["train", str(vids), str(labels), "--mode", "turbo"])

    def test_train_missing_video_folder_exits(self, monkeypatch, dispatch, tmp_path):
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["train", str(tmp_path / "nope"), str(labels)])

    def test_train_missing_label_json_exits(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["train", str(vids), str(tmp_path / "nope.json")])

    def test_train_checkpoint_not_a_file_exits(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, [
                "train", str(vids), str(labels), "-c", str(tmp_path / "missing.pt"),
            ])

    def test_train_part_subsample_out_of_range_exits(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, [
                "train", str(vids), str(labels), "--part_subsample", "1.5",
            ])

    def test_keep_rare_threshold_requires_part_subsample(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, [
                "train", str(vids), str(labels), "--subsample_keep_rare_threshold", "0.1",
            ])

    def test_keep_rare_threshold_out_of_range_exits(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, [
                "train", str(vids), str(labels),
                "--part_subsample", "0.5", "--subsample_keep_rare_threshold", "9",
            ])

    def test_infer_valid_dispatch(self, monkeypatch, dispatch, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.write_text("x")
        vids = tmp_path / "vids"
        vids.mkdir()
        _run_main(monkeypatch, ["infer", str(ckpt), str(vids), "-b", "16", "-w", "2"])
        assert dispatch["name"] == "_cmd_infer"
        args = dispatch["args"]
        assert args.checkpoint == str(ckpt)
        assert args.video_folder == str(vids)
        assert args.batch_size == 16
        assert args.num_workers == 2
        assert args.compile is False

    def test_infer_missing_checkpoint_exits(self, monkeypatch, dispatch, tmp_path):
        vids = tmp_path / "vids"
        vids.mkdir()
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["infer", str(tmp_path / "nope.pt"), str(vids)])

    def test_infer_missing_video_folder_exits(self, monkeypatch, dispatch, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.write_text("x")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["infer", str(ckpt), str(tmp_path / "nope")])

    def test_train_config_valid_dispatch(self, monkeypatch, dispatch, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("a: 1\n")
        _run_main(monkeypatch, ["train-config", str(cfg_file)])
        assert dispatch["name"] == "_cmd_train_config"
        assert dispatch["args"].config == str(cfg_file)

    def test_train_config_missing_file_exits(self, monkeypatch, dispatch, tmp_path):
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["train-config", str(tmp_path / "nope.yaml")])

    def test_reencode_dispatch_defaults(self, monkeypatch, dispatch, tmp_path):
        # main() does no filesystem validation for reencode; that lives in the handler.
        _run_main(monkeypatch, ["reencode", str(tmp_path / "in"), str(tmp_path / "out")])
        assert dispatch["name"] == "_cmd_reencode"
        args = dispatch["args"]
        assert args.processes == 4
        assert args.smallest_side == 512


# ── _cmd_infer ────────────────────────────────────────────────────────────────

class TestCmdInfer:
    def _stub_inference(self, monkeypatch):
        recorded = {}
        fake = types.ModuleType("feral.inference_folder")
        fake.run_inference_folder = lambda **kw: recorded.update(kw)
        monkeypatch.setitem(sys.modules, "feral.inference_folder", fake)
        return recorded

    def test_maps_all_args(self, monkeypatch):
        recorded = self._stub_inference(monkeypatch)
        args = argparse.Namespace(
            checkpoint="ck.pt", video_folder="vf", output="out.json",
            batch_size=16, num_workers=2, compile=True, mode="max", resolution=384,
        )
        cli._cmd_infer(args)
        assert recorded == dict(
            checkpoint_path="ck.pt", video_folder="vf", output="out.json",
            batch_size=16, num_workers=2, compile=True, mode="max", resolution=384,
        )

    def test_compile_defaults_false_when_absent(self, monkeypatch):
        # _cmd_infer uses getattr(args, 'compile', False) defensively.
        recorded = self._stub_inference(monkeypatch)
        args = argparse.Namespace(
            checkpoint="ck.pt", video_folder="vf", output=None,
            batch_size=8, num_workers=4, mode=None, resolution=None,
        )
        cli._cmd_infer(args)
        assert recorded["compile"] is False


# ── _cmd_train_config ─────────────────────────────────────────────────────────

class TestCmdTrainConfig:
    @pytest.fixture
    def stubs(self, monkeypatch):
        state = types.SimpleNamespace(trained=[], logged=[])
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.login = lambda **kw: state.logged.append(kw)
        fake_train = types.ModuleType("feral.train")
        fake_train.main = lambda cfg: state.trained.append(cfg)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
        monkeypatch.setitem(sys.modules, "feral.train", fake_train)
        return state

    def test_logs_in_when_key_present(self, tmp_path, stubs):
        cfg = {"wandb": {"key": "abc123"}, "foo": 1}
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        cli._cmd_train_config(argparse.Namespace(config=str(cfg_file)))
        assert stubs.logged == [{"key": "abc123"}]
        assert stubs.trained == [cfg]

    def test_no_login_when_no_wandb_key(self, tmp_path, stubs):
        cfg = {"foo": 1}
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        cli._cmd_train_config(argparse.Namespace(config=str(cfg_file)))
        assert stubs.logged == []
        assert stubs.trained == [cfg]


# ── _cmd_train ────────────────────────────────────────────────────────────────

def _train_args(**overrides):
    base = dict(
        mode=None, video_folder="vids", label_json_path="labels.json",
        resolution=None, checkpoint=None, part_subsample=None,
        subsample_keep_rare_threshold=None, gradient_checkpointing=False,
        no_wandb=False, public_wandb=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCmdTrain:
    @pytest.fixture
    def stubs(self, monkeypatch):
        state = types.SimpleNamespace(trained=[], logged=[])
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.login = lambda **kw: state.logged.append(kw)
        fake_train = types.ModuleType("feral.train")
        fake_train.main = lambda cfg: state.trained.append(cfg)
        fake_utils = types.ModuleType("feral.utils")
        fake_utils.get_random_run_name = lambda: "run-xyz"
        fake_presets = types.ModuleType("feral.presets")
        fake_presets.apply_mode = lambda cfg, mode: {**cfg, "_mode": mode}
        fake_presets.MODE_HELP = {"lite": "l", "max": "m", "rare": "r"}
        for name, mod in [
            ("wandb", fake_wandb), ("feral.train", fake_train),
            ("feral.utils", fake_utils), ("feral.presets", fake_presets),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        return state

    def test_no_wandb_skips_login_and_trains(self, stubs):
        cli._cmd_train(_train_args(no_wandb=True))
        assert stubs.logged == []
        assert len(stubs.trained) == 1
        cfg = stubs.trained[0]
        assert "wandb" not in cfg
        assert cfg["data"]["prefix"] == "vids"
        assert cfg["data"]["label_json"] == "labels.json"
        assert cfg["run_name"] == "run-xyz"

    def test_public_wandb_uses_shared_account(self, stubs):
        cli._cmd_train(_train_args(public_wandb=True))
        assert stubs.logged == [{"key": "dde17687b4b84ba8171dfede64d865243be41a0e"}]
        cfg = stubs.trained[0]
        assert cfg["wandb"] == {"entity": "sposiboh", "project": "feral_public"}

    def test_interactive_open(self, monkeypatch, stubs):
        monkeypatch.setattr("builtins.input", lambda *a, **k: "open")
        cli._cmd_train(_train_args())
        assert stubs.logged == [{"key": "dde17687b4b84ba8171dfede64d865243be41a0e"}]
        assert stubs.trained[0]["wandb"] == {"entity": "sposiboh", "project": "feral_public"}

    def test_interactive_skip(self, monkeypatch, stubs):
        monkeypatch.setattr("builtins.input", lambda *a, **k: "skip")
        cli._cmd_train(_train_args())
        assert stubs.logged == []
        assert "wandb" not in stubs.trained[0]

    def test_interactive_personal(self, monkeypatch, stubs):
        answers = iter(["personal", "my-api-key", "https://wandb.ai/myent/myproj/runs"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
        cli._cmd_train(_train_args())
        assert stubs.logged == [{"key": "my-api-key"}]
        assert stubs.trained[0]["wandb"] == {"entity": "myent", "project": "myproj"}

    def test_interactive_personal_bad_url_raises(self, monkeypatch, stubs):
        answers = iter(["personal", "my-api-key", "https://example.com/a/b"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
        with pytest.raises(AssertionError):
            cli._cmd_train(_train_args())

    def test_interactive_invalid_choice_exits(self, monkeypatch, stubs):
        monkeypatch.setattr("builtins.input", lambda *a, **k: "maybe")
        with pytest.raises(SystemExit):
            cli._cmd_train(_train_args())
        assert stubs.trained == []

    def test_mode_overlay_applied(self, stubs):
        cli._cmd_train(_train_args(mode="max", no_wandb=True))
        assert stubs.trained[0]["_mode"] == "max"

    def test_optional_overrides_propagate(self, stubs):
        cli._cmd_train(_train_args(
            no_wandb=True, resolution=512, checkpoint="ck.pt",
            part_subsample=0.25, subsample_keep_rare_threshold=0.1,
            gradient_checkpointing=True,
        ))
        cfg = stubs.trained[0]
        assert cfg["data"]["resize_to"] == 512
        assert cfg["starting_checkpoint"] == "ck.pt"
        assert cfg["data"]["part_sample"] == 0.25
        assert cfg["data"]["subsample_keep_rare_threshold"] == 0.1
        assert cfg["model"]["gradient_checkpointing"] is True


# ── _cmd_reencode ─────────────────────────────────────────────────────────────

class _FakePool:
    """Stand-in for multiprocessing.Pool: map() returns a preset result list."""
    results = [1]

    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, items):
        return type(self).results[:len(items)]


class TestCmdReencode:
    @pytest.fixture
    def reencode_stubs(self, monkeypatch):
        """Stub feral.reencode_videos so no ffmpeg download / real work happens."""
        fake = types.ModuleType("feral.reencode_videos")
        fake.is_video_file = lambda path: str(path).endswith(".mp4")
        fake.setup_ffmpeg = lambda: "/usr/bin/ffmpeg"
        fake.process_file = lambda spec: 1
        monkeypatch.setitem(sys.modules, "feral.reencode_videos", fake)
        return fake

    def _args(self, in_dir, out_dir, **over):
        base = dict(input_dir=str(in_dir), output_dir=str(out_dir),
                    processes=2, smallest_side=512)
        base.update(over)
        return argparse.Namespace(**base)

    def test_missing_input_dir_exits(self, reencode_stubs, tmp_path):
        with pytest.raises(SystemExit) as exc:
            cli._cmd_reencode(self._args(tmp_path / "nope", tmp_path / "out"))
        assert exc.value.code == 1

    def test_non_video_file_in_input_exits(self, reencode_stubs, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "notes.txt").write_text("x")
        with pytest.raises(SystemExit) as exc:
            cli._cmd_reencode(self._args(in_dir, tmp_path / "out"))
        assert exc.value.code == 1

    def test_no_videos_found_exits(self, reencode_stubs, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        with pytest.raises(SystemExit) as exc:
            cli._cmd_reencode(self._args(in_dir, tmp_path / "out"))
        assert exc.value.code == 1

    def test_nonempty_output_dir_exits(self, reencode_stubs, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "a.mp4").write_text("x")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "stale.mp4").write_text("x")
        with pytest.raises(SystemExit) as exc:
            cli._cmd_reencode(self._args(in_dir, out_dir))
        assert exc.value.code == 1

    def test_success_creates_output_and_returns(self, reencode_stubs, monkeypatch, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "a.mp4").write_text("x")
        out_dir = tmp_path / "out"  # does not exist yet -> handler mkdirs it
        _FakePool.results = [1]
        monkeypatch.setattr("multiprocessing.Pool", _FakePool)
        # Should complete without raising SystemExit.
        cli._cmd_reencode(self._args(in_dir, out_dir))
        assert out_dir.is_dir()

    def test_partial_failure_exits(self, reencode_stubs, monkeypatch, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "a.mp4").write_text("x")
        (in_dir / "b.mp4").write_text("x")
        out_dir = tmp_path / "out"
        _FakePool.results = [1, 0]  # one of two files failed
        monkeypatch.setattr("multiprocessing.Pool", _FakePool)
        with pytest.raises(SystemExit) as exc:
            cli._cmd_reencode(self._args(in_dir, out_dir))
        assert exc.value.code == 1


def test_train_output_dir_flag(monkeypatch, dispatch, tmp_path):
    labels = tmp_path / 'labels.json'
    labels.write_text('{}')
    _run_main(monkeypatch, ['train', str(tmp_path), str(labels),
                           '--output-dir', str(tmp_path / 'run'), '--no-wandb'])
    assert dispatch['args'].output_dir == str(tmp_path / 'run')


def test_conflicting_wandb_flags_rejected(monkeypatch, dispatch, tmp_path):
    labels = tmp_path / 'labels.json'
    labels.write_text('{}')
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ['train', str(tmp_path), str(labels), '--no-wandb', '--public-wandb'])
    assert exc.value.code == 2
    assert dispatch == {}

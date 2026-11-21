"""批次103: 应用日志按日轮转 + 总量上限。

覆盖:
- trim_rotated_logs: 超总量时从最旧轮转文件开始删, 当前写入文件永不删
- DailyRotatingFileHandler: doRollover 生成 ``<name>.YYYY-MM-DD`` 且继续写回本体
"""
import logging
import time

from adr.core.logging import DailyRotatingFileHandler, trim_rotated_logs


def _mk(path, size):
    path.write_bytes(b"x" * size)
    return path


def test_trim_removes_oldest_until_under_limit(tmp_path):
    base = _mk(tmp_path / "adr.log", 100)
    old = []
    for i, day in enumerate(("2026-11-01", "2026-11-02", "2026-11-03")):
        p = _mk(tmp_path / f"adr.log.{day}", 100)
        # 显式拉开 mtime, 避免同秒创建导致排序不确定
        import os
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
        old.append(p)

    freed = trim_rotated_logs(str(base), 250)

    # 400 → 删最早一份 300 → 仍超 → 再删一份 200 (<=250)
    assert freed == 200
    assert base.exists()
    assert not old[0].exists() and not old[1].exists()
    assert old[2].exists()


def test_trim_never_deletes_active_file(tmp_path):
    base = _mk(tmp_path / "adr.log", 300)
    freed = trim_rotated_logs(str(base), 100)
    assert freed == 0
    assert base.exists()


def test_daily_handler_rolls_to_dated_file(tmp_path):
    base = tmp_path / "adr.log"
    handler = DailyRotatingFileHandler(str(base), max_total_bytes=10 ** 9)
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        handler.emit(logging.LogRecord(
            "adr", logging.INFO, __file__, 1, "hello", None, None))
        handler.doRollover()
        names = [p.name for p in tmp_path.iterdir()]
        assert "adr.log" in names
        assert any(n.startswith("adr.log.") for n in names), names
        handler.emit(logging.LogRecord(
            "adr", logging.INFO, __file__, 1, "world", None, None))
    finally:
        handler.close()
    assert "world" in base.read_text(encoding="utf-8")

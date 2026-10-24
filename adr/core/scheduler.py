"""合成并发调度器 (批次63 第三子批, m10 ④ 调度层 Runtime — N=1 硬预算)。

并发落地顺序 (纪要): N=1 硬预算 → N=2 固定份额 → 动态份额 + 隔离 → 准入控制;
v1.3.0 交付 N=1: 显式并发预算 + 条件变量排队 + 排队可观测 + 调度报告。

- 批次23 以来 GSVEngine._lock 事实上就是 N=1 串行, 但无显式预算、无统计;
  本模块把并发预算变成一等公民 (stats 可测可报告), N=2 固定份额 (1.3.x)
  起改预算值即可插桩。
- 准入控制 (队列上限 / 等待超时 / 拒绝语义) 属第四阶段, 本版刻意不引入 —
  排队无界、不拒绝, queue_depth/synth_busy 观测语义与批次23 完全一致。
- 不可重入: 同一线程嵌套 slot() 会自锁死等 (引擎每个请求恰好取一个槽)。
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import NamedTuple


class AdmissionTicket(NamedTuple):
    """一次准入的观察记录 (供调度报告 / 审计)。"""

    wait_ms: int   # 从进入到拿到槽的等待毫秒
    waiting: int   # 拿到槽瞬间的排队数 (不含自己)
    busy: int      # 拿到槽瞬间的占用数 (含自己)


class SynthesisScheduler:
    """N 预算并发槽 (默认 N=1): 满额时条件变量排队, 释放时唤醒一个。"""

    def __init__(self, max_concurrent: int = 1):
        self._max = max(1, int(max_concurrent))
        self._cond = threading.Condition()
        self._busy = 0
        self._waiting = 0
        # 调度报告累计 (进程生命周期)
        self._admitted = 0
        self._last_wait_ms = 0
        self._max_wait_ms = 0
        self._max_waiting = 0

    @property
    def max_concurrent(self) -> int:
        return self._max

    @property
    def busy(self) -> int:
        with self._cond:
            return self._busy

    @property
    def waiting(self) -> int:
        with self._cond:
            return self._waiting

    def stats(self) -> dict:
        """调度报告快照 (console / 诊断用; 数字为取值瞬间的一致性视图)。"""
        with self._cond:
            return {
                "max_concurrent": self._max,
                "busy": self._busy,
                "waiting": self._waiting,
                "admitted_total": self._admitted,
                "max_waiting_observed": self._max_waiting,
                "last_wait_ms": self._last_wait_ms,
                "max_wait_ms": self._max_wait_ms,
            }

    @contextlib.contextmanager
    def slot(self):
        """取一个并发槽; 预算满则排队 (无界, 不超时 — 准入控制属第四阶段)。

        yield AdmissionTicket (wait_ms / waiting / busy) 供调用方记调度报告。
        """
        with self._cond:
            self._waiting += 1
            if self._waiting > self._max_waiting:
                self._max_waiting = self._waiting
            start = time.monotonic()
            try:
                while self._busy >= self._max:
                    self._cond.wait()
            except BaseException:
                self._waiting -= 1
                raise
            self._waiting -= 1
            self._busy += 1
            self._admitted += 1
            wait_ms = int((time.monotonic() - start) * 1000)
            self._last_wait_ms = wait_ms
            if wait_ms > self._max_wait_ms:
                self._max_wait_ms = wait_ms
            ticket = AdmissionTicket(wait_ms, self._waiting, self._busy)
        try:
            yield ticket
        finally:
            with self._cond:
                self._busy -= 1
                self._cond.notify()

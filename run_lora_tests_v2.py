"""LoRA 测试运行器 (写到文件)。"""
import sys
import io
import importlib.util
from pathlib import Path

REPO = Path(r"e:\新创意构思\新建文件夹\ADR")
sys.path.insert(0, str(REPO))

# 捕获 stdout
out_buf = io.StringIO()
err_buf = io.StringIO()

class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, s):
        for w in self.writers:
            w.write(s)
    def flush(self):
        for w in self.writers:
            try:
                w.flush()
            except Exception:
                pass

sys.stdout = TeeWriter(sys.__stdout__, out_buf)
sys.stderr = TeeWriter(sys.__stderr__, err_buf)

# 加载测试模块 (force fresh)
import sys as _sys
# 移除可能的缓存
for k in list(_sys.modules.keys()):
    if k.startswith("test_lora"):
        del _sys.modules[k]
spec = importlib.util.spec_from_file_location("test_lora", REPO / "tests" / "test_lora.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 跑
passed = 0
failed = 0
for name in sorted(dir(m)):
    if not name.startswith("test_"):
        continue
    func = getattr(m, name)
    try:
        func()
        print(f"[PASS] {name}")
        passed += 1
    except Exception as e:
        import traceback
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1

print(f"\n{'='*60}")
print(f"结果: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*60}")

# 还原
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# 写文件
log_path = REPO / "lora_test_results.txt"
log_path.write_text(out_buf.getvalue() + "\n" + err_buf.getvalue(), encoding="utf-8")
print(f"\n日志: {log_path}")
sys.exit(0 if failed == 0 else 1)

import sys
import os
sys.path.append(os.path.abspath('../../ext/Q-Align'))

print("sys.path:", sys.path)

# 尝试仅导入 q_align（不导入子模块）
import q_align
print("q_align loaded from:", q_align.__file__)

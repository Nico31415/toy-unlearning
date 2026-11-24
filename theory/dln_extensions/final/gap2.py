"""Static formula: fresh samples in stage 2, a relearning attack (a third stage), and label noise. Forget coordinates only."""
import numpy as np
from multi_test import compare
bl = [(1.0, 0.2)]; gr = [(0.2, [0]), (0.1, [0]), (0.2, [0]), (0.15, [0])]            # 0 forget, 1 kept, 2 fresh (same distribution), 3 attacker's samples
for k in (1.0, 1e-2):
    c = np.sqrt(k) / 2
    sy = [dict(parent=None, c=c, use={0: 1, 1: 1}), dict(parent=1, c=c, use={0: 0.0}), dict(parent=1, c=c, use={2: 0.0}), dict(parent=None, c=c, use={1: 1}),
          dict(parent=2, c=c, use={3: 1}), dict(parent=3, c=c, use={3: 1}), dict(parent=4, c=c, use={3: 1}), dict(parent=1, c=c, use={3: 1})]
    pr = [("reused forget samples: distance to retrained", 2, 4), ("fresh samples: distance to retrained", 3, 4), ("reused: norm left", 2, None), ("fresh: norm left", 3, None),
          ("relearning error after reused-sample unlearning", 5, 0), ("relearning error after fresh-sample unlearning", 6, 0), ("relearning error of the retrained model", 7, 0), ("relearning error of the pretrained model", 8, 0)]
    compare(f"fresh samples and relearning attack, k={k:g}", bl, gr, sy, pr, d=1500, ns=8, N=60000, iters=1500)
    sy = [dict(parent=None, c=c, use={0: 1, 1: 1}), dict(parent=1, c=c, use={0: 0.0}), dict(parent=None, c=c, use={1: 1}), dict(parent=1, c=c, use={0: 0.0, 1: 1})]
    compare(f"label noise 0.5, k={k:g}", bl, gr, sy, [("pretrained to teacher", 1, 0), ("unlearned to retrained", 2, 3), ("unlearned with anchor to retrained", 4, 3), ("do nothing to retrained", 1, 3)], d=1500, ns=8, N=60000, iters=1500, noise=[0.5] * 4)

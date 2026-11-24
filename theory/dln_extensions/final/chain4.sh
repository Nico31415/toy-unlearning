python -W ignore -c "
import numpy as np
from multi_test import compare
for k in (1e-6, 1e-4, 100.0):
    c = np.sqrt(k) / 2
    compare(f'blocks R/Sh/F with retain anchor, k={k:g}', [(0.4, 0.2), (0.2, 0.2), (0.4, 0.2)], [(0.3, [0, 1]), (0.12, [1, 2]), (0.08, [1, 2])],
        [dict(parent=None, c=c, use={0: 1, 1: 1, 2: 1}), dict(parent=1, c=c, use={1: 0.0, 0: 1}), dict(parent=None, c=c, use={0: 1, 2: 1}), dict(parent=None, c=c, use={0: 1}), dict(parent=1, c=0.3 * c, use={1: 0.0})],
        [('pretrained to teacher', 1, 0), ('unlearned (anchor) to untraining ref', 2, 3), ('unlearned (anchor) to unlearning ref', 2, 4), ('do nothing to untraining ref', 1, 3), ('do nothing to unlearning ref', 1, 4), ('kappa=0.3, no anchor, to unlearning ref', 5, 4)], d=2000, ns=8, N=100000, iters=1500)
"
python -W ignore sgd_dmft.py 100 0.01 0.2 150 8
python -W ignore dmft_nl2.py npo 100 1 8 1 1.5 0.6 0.5 0.025 3000
python -W ignore es_cov.py 100 1 0.01 0.8 4 6 2000
python -W ignore sgd_dmft.py 1e-6 0.04 0.1 350 6 0 1500
python -W ignore dmft_nl2.py npo 1e-6 1 8 8 3 0.6 0.5 0.05 2000
python -W ignore es_cov.py 1e-6 10 0.05 3 4 6 1500
echo CHAIN4_DONE

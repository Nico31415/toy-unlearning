python -W ignore sgd_dmft.py 1.0 0.04 0.1 250 8
python -W ignore sgd_dmft.py 1.0 0.04 1.0 250 8
python -W ignore sgd_dmft.py 0.01 0.04 0.1 300 8
python -W ignore es_cov.py 0.01 6
python -W ignore dmft_nl2.py npo 0.01 1 12 4 3 1.5 0.2
python -W ignore dmft_nl2.py flip 0.01 0 12 4 1.5 1.5 0.2
python -W ignore -c "
import numpy as np
from multi_test import compare
c=0.5
compare('blocks, k=1, d=3000, 12 seeds', [(0.4, 0.2), (0.2, 0.2), (0.4, 0.2)], [(0.3, [0, 1]), (0.12, [1, 2]), (0.08, [1, 2])],
 [dict(parent=None, c=c, use={0: 1, 1: 1, 2: 1}), dict(parent=1, c=c, use={1: 0.0, 0: 1}), dict(parent=None, c=c, use={0: 1, 2: 1}), dict(parent=None, c=c, use={0: 1}), dict(parent=1, c=0.3 * c, use={1: 0.0})],
 [('pretrained to teacher', 1, 0), ('unlearned (anchor) to untraining ref', 2, 3), ('unlearned (anchor) to unlearning ref', 2, 4), ('do nothing to untraining ref', 1, 3), ('do nothing to unlearning ref', 1, 4), ('kappa=0.3, no anchor, to unlearning ref', 5, 4)], d=3000, ns=12, N=100000)
"
echo CHAIN_DONE

from __future__ import annotations

import sys

import compute_gd_recovery_worker as full_worker


SMALL_RECOVERY_ALPHAS_JSON = "[0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]"


def main() -> None:
    # 3 methods x 3 teachers x 3 seeds x 2 post selections x 2 targets
    # x 7 recovery alphas x 1 variant = 756 tasks.
    extra_args = [
        "--variants",
        "full_keep_w",
        "--recovery-alphas-json",
        SMALL_RECOVERY_ALPHAS_JSON,
    ]
    sys.argv = [sys.argv[0], *sys.argv[1:], *extra_args]
    full_worker.main()


if __name__ == "__main__":
    main()

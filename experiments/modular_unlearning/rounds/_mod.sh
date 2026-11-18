# Shared settings of the modular rounds (the transformer plan). Sourced by the mod_*.sh rounds; submits nothing.
# Forget addition mod 60; retain mod 59, mod 61 and one partner q. Conditions (max shared frequencies gcd(60, q) - 1):
#   p60-q49 none (0), p60-q36 low (11), p60-q40 medium (19), p60-q30 high (29).
ROOT=${ROOT:-data/modular_unlearning/main_wd}
SEEDS=${SEEDS:-"0 1 2"}
NONE=p60-q49
ALL="p60-q49 p60-q36 p60-q40 p60-q30"
A="--out_root $ROOT"
# the family of the None condition: a grid, and L_full - 50 and L_full - 400 for every L_full used by the cells
LFULLS="400 800 1800"
FAM_LS=$(for x in 0 50 100 200 400 800 1200 1800 $(for f in $LFULLS; do echo $((f - 50)) $((f - 400)); done); do
  (( x >= 0 )) && echo $x; done | sort -n | uniq | tr '\n' ' ')
ns_for() { local f=$1 x; for x in $FAM_LS; do (( x < f )) && echo $((f - x)); done | sort -n | tr '\n' ' '; }
nseeds() { echo $SEEDS | wc -w; }

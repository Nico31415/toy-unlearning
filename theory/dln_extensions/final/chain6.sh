python -W ignore dmft_blocks2.py flow 1.0 60 40 5 6000 labels
python -W ignore dmft_blocks2.py sgd 1.0 100 60 6 4000 squared 0.06 0.2
python -W ignore dmft_blocks2.py sgd 1.0 60 60 6 4000 "NPO + retain term on all kept" 0.06 0.2
python -W ignore dmft_blocks2.py flow 0.01 80 40 5 6000 labels
python -W ignore dmft_blocks2.py sgd 0.01 130 60 8 3000 squared 0.06 0.2
echo CHAIN6_DONE

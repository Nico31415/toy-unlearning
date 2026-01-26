Experiment Setup


c_pt = [1e-6, 1e-3, 1]
lambda_pt = [-c_pt, -0.99 * c_pt, 0, 0.99 * c_pt]
gamma_reinit = [0, 1, 10]
omega = [0, 0.5, 1]
alpha = np.linspace(0.01, 0.5, 11)
rho_pt = [0.1]
rho_ft = [0.1, 0.04, 0.01]
seeds = [i for i in range(6, 20)]

EXPERIMENT 1 : 
DO WE BENEFIT FROM EXISTING FEATURES?
TO DIFFERENTIATE BETWEEEN PRETRAINING DEPENDENCE AND INDEPENDENCE
- vary omega, fixed sparsity of 0.1 for PT+FT
- vary c_pt, fixed lambda_pt = 0, gamma_reinit = 0
- vary lambda_pt, fixed c_pt = 1e-3, gamma_reinit = 0
- vary gamma_reinit, fixed c_pt = 1e-3, lambda_pt = 0

EXPERIMENT 2 (2x(8x 14 seeds) = 256 experiments): 
CAN WE LEARN NEW FEATURES?
TO DIFFERENTIATE BETWEEN RICH AND LAZY LEARNING IN NEW FEATURES
- omega = 0, vary rho_ft to be 0.1 or 0.9 
- vary c_pt, fixed lambda_pt = 0, gamma_reinit = 0
- vary lambda_pt, fixed c_pt = 1e-3, gamma_reinit = 0
- vary gamma_reinit, fixed c_pt = 1e-3, lambda_pt = 0

EXPERIMENT 3 (total seeds = 336 experiments):
CAN WE GET INTO THE NESTED FEATURE REGIME?
TO SHOW RICH AND LAZY LEARNING ON PRETRAINED FEATURES 
- omega = {1, 0}, rho_pt = 0.1, rho_ft = {0.01, 0.04}
- vary c_pt, fixed lambda_pt = 0, gamma_reinit = 0
- vary lambda_pt, fixed c_pt = 1e-3, gamma_reinit = 0
- vary gamma_reinit, fixed c_pt = 1e-3, lambda_pt = 0

EXPERIMENT 4 
SINGLE TASK LEARNING 
TO COMPARE TO THE OTHER CURVES, TO SHOW LAMBDA_PT IS IRRELEVANT FOR SINGLE TASK LEARNING, 
TO SHOW INFLUENCE OF C_PT ON SLT
- rho_pt = {0.9, 0.04, 0.01, 0.1}
- c_pt = {1e-6, 1e-3, 1}
- lambda_pt = {0, -c_pt, -0.99 * c_pt, 0.99 * c_pt}

(MAYBE DO -0.9999 C INSTEAD OF -C)
BY CHOOSING THESE PARAM COMBINATIONS, WE GET OUR 4 REGIMES: 
c_pt = 1e-6, lambda_pt = -c_pt, gamma_reinit = 0 = RICH PRETRAINING INDEPENDENT REGIME (NOT SURE)
c_pt = 1e-6, lambda_pt = -0.99*c_pt, gamma_reinit = 0 = RICH PRETRAINING DEPENDENT REGIME 
c_pt = 1e-6, lambda_pt = 0.99*c_pt, gamma_reinit = 0 = LAZY PRETRAINING DEPENDENT REGIME 
c_pt = 1e-6, lambda_pt = 0, gamma_reinit = 1 = LAZY PRETRAINING INDEPENDENT REGIME 

RICH PRETRAINING INDEPENDENT REGIME
- omega = 0, rho_ft = 0.01

RICH PRETRAINING DEPENDENT REGIME 
- omega = 1, rho_ft = 0.01 

LAZY PRETRAINING DEPENDENT REGIME
- omega = 1, rho_ft = 0.1

LAZY PRETRAINING INDEPENDENT REGIME
- omega = 0, rho_ft = 0.9



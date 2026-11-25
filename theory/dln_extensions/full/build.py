import re
L=open('../lossopts/losses.tex').read(); D=open('../mse/derivations.tex').read()
Lb=L[L.index(r"\section{Model}"):L.index(r"\end{document}")]
Db=D[D.index(r"\section{The flow in one variable}"):D.index(r"\section{Relation to \texttt{losses.pdf}}")]
symtab=D[D.index(r"\textbf{Symbols.}"):D.index(r"\section{The flow in one variable}")]
status=D[D.index(r"\section{What is and is not derived}"):D.index(r"\end{document}")]
# ---------- Part I: adapt the setup text to the common notation
def R(s,a,b,need=True):
    assert (a in s) or not need, a[:60]
    return s.replace(a,b)
Lb=R(Lb,r"\beta_g",r"\tilde\beta"); Lb=R(Lb,r"g_i",r"\tilde y_i")
Lb=re.sub(r"q_j",r"\\vartheta_j",Lb)
Lb=R(Lb,r"\lambda\|\theta\|^2",r"\lambda\|w\|^2"); Lb=R(Lb,r"$\theta-\lambda(\theta_F-\theta)$",r"$w-\lambda(w_F-w)$")
Lb=R(Lb,r"come within $\eta$ of each target? With distances divided by $\|\beta^\star_{d_F}\|$ ($\eta^2=0.15$)",r"come within $\delta$ of each target? With distances divided by $\|\beta^\star_{d_F}\|$ ($\delta^2=0.15$)")
Lb=Lb.replace(r"\le\eta\big\}",r"\le\delta\big\}").replace(r"\le2\eta$",r"\le2\delta$")
assert r"\eta" not in Lb, [m.start() for m in re.finditer(r"\\eta",Lb)]
Lb=R(Lb,r"""\operatorname*{argmin}_\beta\ D_{Q_k}(\beta,\beta_0)\quad\text{subject to}\quad X^\top\beta=t,
\qquad Q_k'(\beta)=\tfrac12\operatorname{arcsinh}\tfrac{2\beta}{\sqrt k},""",r"""\operatorname*{argmin}_\beta\ D_{Q_k}(\beta,\beta_0)\quad\text{subject to}\quad X^\top\beta=t,
\qquad Q_k(\beta)=\textstyle\sum_jq_k(\beta_j),\quad q_k'(\beta)=\tfrac12\operatorname{arcsinh}\tfrac{2\beta}{\sqrt k},""")
Lb=R(Lb,r"where $D_{Q_k}$ is the Bregman divergence of $Q_k$ (Woodworth et al.\ 2020; Azulay et al.\ 2021).",r"where $D_{Q_k}$ is the Bregman divergence of $Q_k$ (Woodworth et al.\ 2020; Azulay et al.\ 2021). Derived in Section~\ref{sec:ends}.")
Lb=R(Lb,r"The network computes $f(x)=x^\top\beta$ with $\beta_j=u_j^+v_j^+-u_j^-v_j^-$; all weights start at the same value, which sets the initialisation scale $k$.",r"The network computes $f(x)=x^\top\beta$ with $\beta_j=u_j^+v_j^+-u_j^-v_j^-$; all weights start at the same value, which sets the initialisation scale $k$. Since $u_j^\pm=v_j^\pm$ at the start and this is preserved, $\beta=u^2-v^2$ entrywise with $u=u^+$, $v=u^-$; the derivations use this form.")
Lb=R(Lb,r"\textbf{Rescaling.} Multiplying all weights by $\sqrt\kappa$ maps $\beta\to\kappa\beta$ and $k\to\kappa^2k$.",r"\textbf{Rescaling.} Multiplying all weights by $\sqrt\kappa$ maps $\beta\to\kappa\beta$ and $k\to\kappa^2k$ (Result 4).")
Lb=R(Lb,r"and $y=x^\top\beta^\star$ (no noise).",r"and $y=x^\top\beta^\star$ (no noise; the derivations allow $y=x^\top\beta^\star+\sigma\xi$, $\xi\sim\mathcal N(0,1)$). Block $B$ is a fraction $p_B$ of the coordinates.")
Lb=R(Lb,r"Derivations, checks and proofs are in \texttt{extensions\_maths.pdf}; this is the list.",r"Checks and proofs for the cross-entropy items are in \texttt{extensions\_maths.pdf}; this is the list.")
# readiness definition: tie to theta
Lb=R(Lb,r"Its state is not $\beta_j$ but its \emph{readiness} $\vartheta_j$: minus the gradient it has accumulated over all training so far, divided by $\frac12\log\frac{2}{\kappa\sqrt k}$",r"Its state is not $\beta_j$ but its \emph{readiness} $\vartheta_j$: minus the gradient it has accumulated over all training so far (the mirror variable $\theta_j=q_k'(\beta_j)$ of Result 1), divided by $\frac12\log\frac{2}{\kappa\sqrt k}$")
i1=Lb.index(r"\section{Exact curves vs simulation}"); i2=Lb.index(r"\section{Extensions (24 November)}"); i3=Lb.index(r"\section{Plan}")
setup=Lb[:i1]; curves=Lb[i1:i2]; ext=Lb[i2:i3]; plan=Lb[i3:]
# metrics + symbols inserted after the Data section (before Unlearning losses)
j=setup.index(r"\section{Unlearning losses}")
metrics=r"""\textbf{Metrics.} \emph{Retain error} and \emph{forget error}: $\mathbb E(x^\top\beta-x^\top\beta^\star)^2$ over fresh $x\sim P_R$ and $x\sim P_F$. \emph{Error on $F$} and \emph{on the kept forget samples}: mean of $(f_i-y_i)^2$ over $F$ and over $\Ff\setminus F$, with $f_i=x_i^\top\beta$. Distance to a reference: $\|\beta-\beta_{\rm ref}\|^2/d$, which equals $\|\cdot\|^2/\|\beta^\star\|^2$ since $\mathbb E(\beta^\star_j)^2=1$.

"""+symtab.replace(r"potential and its divergence (Section 3)",r"potential and its divergence (Section~\ref{sec:ends})")+"\n"
setup=setup[:j]+metrics+setup[j:]
# ---------- Part II: derivations
Db=Db.replace(r"\section{Where a stage ends}",r"\section{Where a stage ends}\label{sec:ends}").replace(r"\section{Replica method: blocks and groups}",r"\section{Replica method: blocks and groups}\label{sec:blocks}")
Db=Db.replace(r"\section{Replica method: the references and the metrics}",r"\section{Replica method: the references and the metrics}\label{sec:refs}").replace(r"\section{Unbalanced layers (Azulay et al.\ 2021)}",r"\section{Unbalanced layers (Azulay et al.\ 2021)}\label{sec:unbal}")
Db=R(Db,r"Combine with Section 9:",r"Combine with Section~\ref{sec:blocks}:"); Db=R(Db,r"(the workshop loss, Section 15)",r"(the workshop loss, Section~\ref{sec:losses})")
Db=R(Db,r"(\texttt{losses.pdf} writes $Q_k'$ for $q_k'$.)",""); Db=R(Db,r"This is not the rescaling $\kappa$ and is not used in \texttt{losses.pdf}.",r"This is not the rescaling $\kappa$ and is not used in this project.")
assert "losses.pdf" not in Db, Db[Db.index("losses.pdf")-80:Db.index("losses.pdf")+40]
setup=setup.replace(r"\section{Unlearning losses}",r"\section{Unlearning losses}\label{sec:losses}").replace(r"\section{Data}",r"\section{Data}\label{sec:data}",1)
mapping=r"""\section{Where each item of Part I is derived}
\begin{center}\small\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.6cm}>{\raggedright\arraybackslash}p{10.6cm}@{}}\toprule
Part I & Part II\\\midrule
Implicit bias; rescaling & Results 1--2; Result 4.\\
Key fact: every loss is $\frac{c_i}2(f_i-t_i)^2$ and the weights do not move $U$ & Result 2, Step 3.\\
Relabel, $\tilde\beta=0$ & targets $0$ on $F$: $b=0$ in Results 13--16.\\
Relabel, oracle $\tilde\beta=A(\Rf)$ & exact end point by Result 2; the targets $x_i^\top A(\Rf)$ are not of the form $b\,x_i^\top\beta^\star$, so Results 13--16 do not cover it. Not derived here.\\
Anchored ascent & targets from $\frac{\mu_d\tilde\beta-\mu_a\beta^\star}{\mu_d-\mu_a}$; with $\tilde\beta=0$ this is $b=-\frac{\mu_a}{\mu_d-\mu_a}$ ($b=-0.5$: $\mu_a/\mu_d=\frac13$; $b=-1$: $\frac12$). Results 11, 13--16.\\
Workshop paper & multiples $\frac{\mu_r}{\mu_r-\mu_a}$ on $d_R\cup\dSh$ and $-\frac{\mu_a}{\mu_r-\mu_a}$ on $d_F$, on every sample. Result 14 with a block-dependent target vector.\\
Gradient difference & no end point; Result 10 (no motion from a fit); otherwise the trajectory, Result 18, until it diverges.\\
Weight decay $\lambda\|w\|^2$ & Result 5; large $d$: Result 13, ``weight decay kept on''.\\
Early stopping, $U(\tau)$ & Result 18. The lazy and rich limits stated in Part I are not rederived.\\
$\mathrm{mem}(F)$, $\mathrm{xinf}_C(F)$ & $\mathrm{mem}(F)=\sum_{B\in\{\dSh,d_F\}}p_B\E_B(A(D\setminus F)-\beta^\star)^2$, since $A$ fits $F$ and $F$ is fresh for $A(D\setminus F)$; $\mathrm{xinf}_C=\sum_Bp_B\E_B(A-A(D\setminus F))^2$ over the blocks of class $C$, from Result 16.\\
Question 1 (distances to the two references) & Results 15--16 and the metrics of Section~\ref{sec:refs}.\\
Question 2 (the attacker's stage) & a third stage started from $U$: Section~\ref{rep2} applied again with $U$ in the role of $A$. Implemented in the solver, not written out.\\
Exact curves of Section~\ref{sec:curves} (Gordon's comparison conditioned on stage 1) & the same problem as Result 15. Both agree with simulation; the two sets of equations have not been matched term by term.\\\bottomrule
\end{tabular}\end{center}
The industry losses with hard labels (Section~\ref{sec:industry}) are outside Part II; see \texttt{extensions\_maths.pdf}.

"""
runs=r"""\section{Trajectory runs with squared loss (25 November)}\label{sec:runs}
Pretraining to convergence, then unlearning for a time $\tau\le40$ by gradient descent with step $0.01$; every metric recorded along the way, with theory curves (Result 18) for $\tau\le6$. Fixed: $k=10^{-5}$, $\rho=0.2$, $d=3000$, 5 seeds. Baseline: $|d_R|:|\dSh|:|d_F|=1:0.4:1$; $0.3$ samples per coordinate seen, in each class; $|F|=\frac12|\Ff|$; $R=\Rf$; $\kappa=1$; no noise, no weight decay, full batch.

\textbf{Losses run} (names used in the figures, in the notation of Section~\ref{sec:losses}). ``Retain only'': no term on $F$. ``Forget to zero'': relabel with $\tilde\beta=0$. ``Forget to noise'': targets $\mathcal N(0,1)$ on $F$. ``Negated target $b$'': anchored ascent with $b=-\mu_a/(\mu_d-\mu_a)=-0.5,-1$. ``Signed'': gradient difference. ``Pure ascent'': the same without the retain term. In the figures ``ideal untraining'' is $A(D\setminus F)$, ``ideal unlearning'' is $A(\Rf)$, ``left-over'' samples are $\Ff\setminus F$, and the rescaling $\kappa$ is written $S$.

\textbf{Status.} Finished on all four block settings ($1:0.4:1$, $0:1:0$, $1:0:1$, $1:1:1$): baseline; $|F|/|\Ff|=0.1,0.9,1$; label noise $\sigma^2=0.1,0.5$ in pretraining; fresh forget samples; $\kappa=0.1,0.3,3$. Not run yet: loss weights, $R$ a quarter of $\Rf$, SGD, weight decay, and $0.8$ or $1.5$ samples per coordinate.

\textbf{Theory against simulation.} Relabel-to-zero and the two anchored-ascent targets: 2868 comparisons, $94\%$ within 2 standard errors, median gap $3.0\%$ (Figure~\ref{fig:base}). Targets of noise are much noisier (the outcome follows the empirical variance of the $\approx260$ noise targets); with 12 runs per side they agree, except a $\approx10\%$ excess of theory on the kept forget samples for $\tau\ge4$ (Figure~\ref{fig:noise}). Gradient difference: theory predicts no motion (Result 10); in simulation it diverges at $\tau\approx2$--$4$ from the $2\times10^{-4}$ training error left by pretraining.

\begin{figure}[!ht]\centering\includegraphics[width=\linewidth]{figs/base__C.png}
\caption{\small Baseline. Top: theory (lines) and simulation (markers) for $\tau\le6$. Bottom: simulation to $\tau=40$. The orange theory curve (targets of noise) is three unlucky runs; see Figure~\ref{fig:noise}.}\label{fig:base}\end{figure}
\begin{figure}[!ht]\centering\includegraphics[width=\linewidth]{figs/noise_more_seeds.png}
\caption{\small Targets of noise at baseline with 12 runs per side.}\label{fig:noise}\end{figure}

\textbf{Baseline, end of unlearning} (simulation).
\begin{center}\small\begin{tabular}{@{}lcccccc@{}}\toprule
& retain error & error on $F$ & on $\Ff\setminus F$ & forget error & dist.\ to $A(D\setminus F)$ & dist.\ to $A(\Rf)$\\\midrule
$A(D)$ & 0.387 & 0.000 & 0.000 & 0.381 & 0.207 & 0.290\\
$A(D\setminus F)$ & 0.429 & 0.507 & 0.000 & 0.499 & 0 & \\
$A(\Rf)$ & 0.452 & 0.555 & 0.553 & 0.547 & & 0\\\midrule
relabel to $0$ & 0.430 & 0.573 & 0.231 & 0.499 & 0.140 & 0.096\\
anchored ascent, $b=-0.5$ & 0.489 & 1.311 & 0.360 & 0.674 & 0.260 & 0.249\\
anchored ascent, $b=-1$ & 0.530 & 2.341 & 0.749 & 0.998 & 0.530 & 0.567\\
targets of noise & 0.486 & 1.499 & 0.727 & 0.955 & 0.657 & 0.736\\\bottomrule
\end{tabular}\end{center}
Smallest distance along the path (time reached), to $A(D\setminus F)$ and to $A(\Rf)$: relabel to $0$, $0.119$ ($\tau=2.3$) and $0.090$ ($11.7$); $b=-0.5$: $0.125$ ($0.8$) and $0.112$ ($2.2$); $b=-1$: $0.129$ ($0.4$) and $0.127$ ($0.9$). Anchored ascent passes near the same point 3--5 times sooner, then overshoots.

\textbf{Blocks} (relabel to $0$).
\begin{center}\small\begin{tabular}{@{}lcccc@{}}\toprule
$|d_R|:|\dSh|:|d_F|$ & $1:0:1$ & $1:0.4:1$ & $1:1:1$ & $0:1:0$\\\midrule
retain error, $A(D)\to$ end & $0.374\to0.374$ & $0.387\to0.430$ & $0.356\to0.472$ & $0.068\to0.704$\\
distance to $A(\Rf)$, start $\to$ end & $0.222\to0.012$ & $0.290\to0.096$ & $0.339\to0.187$ & $0.523\to0.395$\\
distance to $A(D\setminus F)$, start $\to$ best & $0.184\to0.094$ & $0.207\to0.119$ & $0.221\to0.135$ & $0.262\to0.196$\\\bottomrule
\end{tabular}\end{center}
With no shared block the retain error does not move (Result 9) and relabel-to-zero converges to $A(\Rf)$. With all coordinates shared, both classes together exceed the recovery threshold, $A(D)$ is nearly $\beta^\star$, and every forget loss destroys the retain error.

\begin{figure}[!ht]\centering\includegraphics[width=\linewidth]{figs/summary.png}
\caption{\small Relabel to $0$ at the end of unlearning. Left: error on $\Ff\setminus F$ relative to error on $F$, against $|F|/|\Ff|$. Middle and right: against the rescaling $\kappa$ (written $S$).}\label{fig:sum}\end{figure}

\textbf{Deletion size} (Figure~\ref{fig:sum}, left). The ratio (error on $\Ff\setminus F$)/(error on $F$) is $0.12,0.40,0.72$ at $|F|/|\Ff|=0.1,0.5,0.9$; the local rule (Result 7) gives $0.13,0.55,0.77$, a flat metric $0.03,0.16,0.33$.

\textbf{Rescaling} (Figure~\ref{fig:sum}, middle and right). Larger $\kappa$ lowers the damage to the kept forget samples and raises the distance to $A(\Rf)$; smaller $\kappa$ does the opposite. Blocks $1:0:1$: ratio $0.97,0.87,0.63,0.34$ and distance to $A(\Rf)$ $0.002,0.002,0.012,0.096$ at $\kappa=0.1,0.3,1,3$, as the corollary of Result 4 predicts.

\textbf{Label noise in pretraining.} All errors rise; the ratio above stays $0.40$ at $\sigma^2=0.1$; the ordering of losses is unchanged. \textbf{Fresh forget samples.} Relabel-to-zero on fresh samples raises the error on $F$ and on $\Ff\setminus F$ equally ($0.143$, $0.137$): it cannot single out $F$; the distance to $A(\Rf)$ ends at $0.197$ (reused: $0.096$).

"""
pre=r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,enumitem,array,graphicx}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{pdflscape,tabularx,longtable}
\usepackage[font=small]{caption}\usepackage{placeins}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\newcommand{\bs}{\beta^\star}\newcommand{\E}{\mathbb E}\newcommand{\asinh}{\operatorname{arcsinh}}\newcommand{\sk}{\sqrt k}
\newcommand{\Rf}{R_{\mathrm{full}}}\newcommand{\Ff}{F_{\mathrm{full}}}\newcommand{\dSh}{d_{\mathrm{Sh}}}
\newcommand{\step}[1]{\par\smallskip\textbf{Step #1.}\ }
\newcommand{\Part}[2]{\clearpage{\LARGE\bfseries Part #1.\ #2\par}\bigskip}
\title{Unlearning in diagonal linear networks:\\ setup, questions, derivations and results}
\author{}\date{25 November 2026}
\begin{document}\maketitle
We use a model in which the end point of every training run is known exactly, and ask two questions about gradient-based unlearning.
\textbf{(1)} Does it remove only the given samples (\emph{untraining}) or the whole distribution they came from (\emph{unlearning})?
\textbf{(2)} Is the forgotten behaviour deleted, or only hidden so that an attacker can recover it cheaply (\emph{tamper resistance})?

\textbf{How to read this.} Part I states the model, the data, the losses and the two questions. Part II derives the theory step by step: Sections~\ref{sec:flow}--\ref{sec:unbal} are exact; the replica sections give the large-$d$ end points; the cavity sections rederive them and give trajectories. Part III has the results. One notation is used throughout; the table of symbols is in Section~\ref{sec:data}.
\tableofcontents

\Part{I}{Setup and questions}
"""
Db=Db.replace(r"\section{The flow in one variable}",r"\section{The flow in one variable}\label{sec:flow}")
part2=r"\Part{II}{Derivations}"+"\nEvery result is followed by its derivation. Replica and cavity are derivations, not proofs.\n\n"+mapping+Db
part3=r"\Part{III}{Results}"+"\n"+curves+r"\FloatBarrier"+"\n"+runs+r"\FloatBarrier"+"\n"+ext+status.replace(r"\end{document}","").replace("tables in Sections 10--11",r"tables in Sections~\ref{rep2}--\ref{sec:refs}")+plan
out=pre+setup+part2+part3+"\n\\end{document}\n"
# hard-coded section numbers left?
for m in re.finditer(r"Sections? \d",out): print("HARD", out[m.start()-40:m.start()+40].replace("\n"," "))
open('merged.tex','w').write(out)

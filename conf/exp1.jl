EXPERIMENT = "exp1"
MAX_VALUE = 30.
EXPAND_ONLY = true
MAX_DEPTH = 3.

SKIP_GROUP = true
#SKIP_FULL = true
#SKIP_CV = true

# this quote thing allows us to refer to types that aren't defined yet
QUOTE_MODELS = quote 
    [
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune},
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune_DistCost},
        Heuristic{:Best_Satisfice_BestNext_DepthLimit},
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_DistCost},
    ] 
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune},
        #Heuristic{:Best_ProbBest},  # too slow :(
    ]
end

QUOTE_SIM_MODELS = quote [
    OptimalPlus{:Default},
    MetaGreedy{:Default},
    Heuristic{:Random},
] end

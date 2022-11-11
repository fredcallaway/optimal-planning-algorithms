EXPERIMENT = "exp2"
EXPAND_ONLY = true

MAX_VALUE = 50.
MAX_DEPTH = 5.

QUOTE_MODELS = quote 
    [
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune},
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune_TreeCost},
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune_ScreenCost},
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune_TreeCost_ScreenCost},
        Heuristic{:Depth_Satisfice_BestNext_DepthLimit_Prune},
        Heuristic{:Depth_Satisfice_BestNext_DepthLimit_Prune_TreeCost},
        Heuristic{:Depth_Satisfice_BestNext_DepthLimit_Prune_ScreenCost},
        Heuristic{:Depth_Satisfice_BestNext_DepthLimit_Prune_TreeCost_ScreenCost},
        Heuristic{:Breadth_Satisfice_BestNext_DepthLimit_Prune},
        Heuristic{:Breadth_Satisfice_BestNext_DepthLimit_Prune_TreeCost},
        Heuristic{:Breadth_Satisfice_BestNext_DepthLimit_Prune_ScreenCost},
        Heuristic{:Breadth_Satisfice_BestNext_DepthLimit_Prune_TreeCost_ScreenCost},

        # OptimalPlus{:Default},
        # MetaGreedy{:Default},
        # Heuristic{:Random},
        # Heuristic{:Best_Satisfice_BestNext_DepthLimit},
        # Heuristic{:Best_Satisfice_BestNext_DepthLimit_JumpCost},
    ] 
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,
        
        Heuristic{:Best_Satisfice_BestNext_DepthLimit_Prune},
        Heuristic{:Depth_Satisfice_BestNext_DepthLimit_Prune},
        Heuristic{:Breadth_Satisfice_BestNext_DepthLimit_Prune},
        
        # Heuristic{:Best_Satisfice_BestNext_Prune},
        # Heuristic{:Depth_Satisfice_BestNext_Prune},
        # Heuristic{:Breadth_Satisfice_BestNext_Prune},
        
        # Heuristic{:Best_Satisfice_BestNext},
        # Heuristic{:Depth_Satisfice_BestNext},
        # Heuristic{:Breadth_Satisfice_BestNext},
    ]
end

QUOTE_SIM_MODELS = quote [
    OptimalPlus{:Default},
    MetaGreedy{:Default},
    Heuristic{:Random},
    all_heuristic_models(whistles=pick_whistles(exclude=["DepthLimit", "ProbBetter", "ProbBest"]))...
] end


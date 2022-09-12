EXPERIMENT = "exp4"
EXPAND_ONLY = false
FOLDS = 7

MAX_VALUE = 30. # this value is the result of pre-scaling before I used MAX_VALUE
MAX_DEPTH = 5

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},

        OptimalPlus{:Expand},
        MetaGreedy{:Expand},
        Heuristic{:Expand},

        all_heuristic_models()...

    ]
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,

        Heuristic{:Best_Satisfice_BestNext},
        Heuristic{:Depth_Satisfice_BestNext},
        Heuristic{:Breadth_Satisfice_BestNext},
    ]
end

QUOTE_SIM_MODELS = quote [
    OptimalPlus{:Default},
    OptimalPlus{:Expand},
    MetaGreedy{:Default},
    Heuristic{:Random},
    all_heuristic_models(whistles=pick_whistles(exclude=["ProbBetter", "ProbBest"]))...
] end
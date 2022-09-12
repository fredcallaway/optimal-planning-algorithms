if isempty(ARGS)
    push!(ARGS, "exp1")
elseif ARGS[1] != "exp1"
    println("Nothing to do for this experiment")
end

using JSON
using Glob
using ProgressMeter

include("conf.jl")
@everywhere include("base.jl")
@everywhere include("models.jl")

m = (load_trials(EXPERIMENT) |> values |> first |> first).m

@everywhere begin
    m = $m
    _bf = create_model(Heuristic{:Best}, [1e5, -1e5, 0], (), default_space(Heuristic{:Best}))
    function is_bestfirst(d::Datum)
        (d.c != TERM) && action_dist(_bf, d)[d.c+1] > 1e-2
    end

    function best_first_rate(trials)
        map(get_data(trials)) do d
            d.c == TERM && return missing
            is_bestfirst(d)
        end |> skipmissing |> (x -> isempty(x) ? NaN : mean(x))
    end
end

# %% --------

println("Computing best-first rate under optimal policy")
res = @showprogress map(glob("$base_path/sims/Optimal-cost*")) do f
    cost, mid = match(r"cost(.*)-(.*)", f).captures
    cost = parse(Float64, cost)
    trials = deserialize(f);
    (
        mdp = mid,
        cost = cost,
        n_click = mean(length(t.cs) for t in trials) - 1,
        best_first = best_first_rate(trials),
    )
end
res |> json |> write("$results_path/bestfirst_optimal.json")

println("Computing best-first rate under random policy")
res = @showprogress pmap(range(0, 1, length=201)) do p_term
    model = RandomSelection(p_term)
    trials = [simulate(model, m) for i in 1:100000]
    (
        n_click = mean(length(t.cs) for t in trials) - 1,
        best_first = best_first_rate(trials)
    )
end
res |> json |> write("$results_path/bestfirst_random.json")
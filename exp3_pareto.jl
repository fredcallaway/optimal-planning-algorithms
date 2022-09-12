using CSV
using Glob
using ProgressMeter

@everywhere begin
    include("utils.jl")
    include("mdp.jl")
    include("data.jl")
    include("models.jl")
end
include("solve.jl")

redirect_worker_stderr("out/pareto")

push!(ARGS, "exp3")
include("conf.jl")

# %% --------

mdps = let
    all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
    flat_trials = flatten(values(all_trials));
    unique(getfield.(flat_trials, :m))
end

all_ids = map(Iterators.product(mdps, COSTS)) do (m, cost)
    id(mutate(m, cost=cost))
end

@showprogress pmap(all_ids) do i
    serialize("mdps/withcost/$i")

    f = "mdps/pareto/$i-Optimal.csv"
    if !force && isfile(f)
        println("$f already exists")
    else
        println("Generating $f...")
        V = load_V_nomem(i)
        m = mutate(V.m, cost=0)
        pol = OptimalPolicy(m, V)
        res = (model="Optimal", mdp=id(m), cost=V.m.cost, mean_reward_clicks(pol)...)
        ress = [res]

        if !EXPAND_ONLY
            pol2 = OptimalPolicyExpandOnly(m, V)
            res2 = (model="OptimalPlusExpand", mdp=id(m), cost=V.m.cost, mean_reward_clicks(pol2)...)
            push!(ress, res2)
        end

        CSV.write(f, ress)  # one/two line csv
        # println("Wrote $f")
    end
end

m = deserialize("mdps/withcost/$(all_ids[4])")
load_V(id(mutate(m, expand_only=true)))
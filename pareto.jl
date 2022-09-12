using CSV
using Glob
using ProgressMeter

@everywhere begin
    include("utils.jl")
    include("mdp.jl")
    include("data.jl")
    include("models.jl")
end
redirect_worker_stderr("out/pareto")

@everywhere N_SIM = 100000
@everywhere N_CANDIDATE = 100000
@everywhere COSTS = [0:0.05:4; 100]

println("Running pareto for ", ARGS[1])
include("conf.jl")

mdps = let
    all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
    flat_trials = flatten(values(all_trials));
    unique(getfield.(flat_trials, :m))
end
@everywhere mdps = $mdps

@everywhere function mean_reward_clicks(pol; N=N_SIM)
    @assert pol.m.cost == 0
    reward, clicks = N \ mapreduce(+, 1:N) do i
        roll = rollout(pol)
        [roll.reward, roll.n_steps - 1]
    end
    (reward=reward, clicks=clicks)
end

function sample_models(M, n)
    space = default_space(M)
    space[:ε] = 0.
    space[:β_click] = 100.
    _, lower, upper, _ = bounds(space)
    seq = SobolSeq(lower, upper)
    skip(seq, n)
    map(1:n) do i
        x = Sobol.next!(seq)
        create_model(M, x, [], space)
    end
end

function sample_models(::Type{RandomSelection}, n)
    # ignore n...
    map(RandomSelection, range(0, 1, length=201))
end

function sample_models(::Type{MetaGreedy}, n)
    # ignore n...
    map(COSTS) do cost
        MetaGreedy{:Default,Float64}(cost, 100., 100., 0., 0.)
    end
end

function pareto_front(M, m; n_candidate=N_CANDIDATE, n_eval=N_SIM)
    #if has_component(M, "ProbBest")
    #    # this one is slow so we have to turn down the compute
    #    println("CONFIRMED")
    #    n_eval = 10000
    #    n_candidate = 10000
    #end
    candidates = @showprogress pmap(sample_models(M, n_candidate)) do model
        mrc = mean_reward_clicks(Simulator(model, m), N=n_eval)
        (model=name(M), mdp=id(m), namedtuple(model)..., mrc...)
    end
    sort!(candidates, by=x->x.clicks)
    result = [candidates[1]]
    for x in candidates
        if x.reward > result[end].reward
            push!(result, x)
        end
    end
    result
end

function write_optimal_pareto(;force=false)
    all_ids = map(Iterators.product(mdps, COSTS)) do (m, cost)
        id(mutate(m, cost=cost))
    end

    @showprogress pmap(all_ids) do i
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
end

# %% --------
function write_optimal_pareto(;force=false)
    @showprogress pmap(all_ids) do i
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
end

# %% --------

function write_heuristic_pareto(;force=false)
    # Hs = [:BestFirst, :BreadthFirst, :DepthFirst, :BestPlusDepth, :BestPlusBreadth
    #       # :BestFirstNoBestNext, :BreadthFirstNoBestNext, :DepthFirstNoBestNext
    # ]
    # models = [RandomSelection; MetaGreedy; [Heuristic{H} for H in Hs]]

    # mdps = map(readdir("mdps/base")) do i
    #     deserialize("mdps/base/$i")
    # end
    # models = filter(eval(QUOTE_MODELS)) do M
    #     M <: Heuristic
    # end

    models = eval(QUOTE_PARETO_MODELS)
    # models = [Heuristic{:Depth_NoDepthLimit}]

    for M in models, m in mdps
        f = "mdps/pareto/$(id(m))-$(name(M)).csv"
        if !force && isfile(f)
            println("$f already exists")
        else
            println("Generating $f... ")
            pareto_front(M, m) |> CSV.write(f)
            println("Wrote ", f)
        end
    end
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    # if !isempty(ARGS)
    #     include("conf.jl")
    #     mdps = 
    # end
    mkpath("mdps/pareto")
    force = length(ARGS) >= 3 && ARGS[3] == "force"
    if length(ARGS) > 1
        if ARGS[2] == "optimal"
            write_optimal_pareto(;force)
        elseif ARGS[2] == "heuristic"
            write_heuristic_pareto(;force)
        else
            error("Bad arg: ", ARGS[2])
        end
    else
        write_optimal_pareto()
        write_heuristic_pareto()
    end
end

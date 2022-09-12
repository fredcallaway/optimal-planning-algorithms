using StatsBase
using Distributed
using Glob
using CSV
using DataFrames
using ProgressMeter


include("conf.jl")
@everywhere include("base.jl")
@everywhere include("models.jl")
@everywhere FORCE = false
mkpath("$base_path/sims/")

@everywhere function purify(model::OptimalPlus)
    OptimalPlus{:Pure,Float64}(model.cost, 1e5, 1e5, 0., 0.)
end
SIM_MODELS = eval(QUOTE_SIM_MODELS)

# %% --------

@everywhere function run_simulation(model, wid, mdps; n_repeat=10)
    try
        model_wid = name(model) * "-" * wid
        file = "$base_path/sims/$model_wid"
        if isfile(file) && !FORCE
            println(file, " already exists. Skipping.")
        end
        sims = map(repeat(mdps, n_repeat)) do m
            simulate(model, m; wid=model_wid)
        end
        serialize("$base_path/sims/$model_wid", sims)
        # println("Wrote $base_path/sims/$model_wid")

        if name(model) == "OptimalPlus"
            run_simulation(purify(model), wid, mdps)
        # elseif name(model) == "OptimalPlusPure"
        #     pure_id = "OptimalPlusPure-$(model.cost)"
        #     cp("$base_path/sims/$model_wid", "$base_path/sims/$pure_id")
        end
    catch err
        @error "Error in run_simulation" err model wid mdps
    end
    if model isa OptimalPlus
        # clear the value function from memory
        empty!(memoize_cache(load_V)); GC.gc()
    end
end

function pick_models(flag)
    if flag == :optimal
        println("Only simulating optimal model")
        filter(M -> M <: OptimalPlus, SIM_MODELS)
    elseif flag == :nonoptimal
        println("Only simulating non-optimal models")
        filter(M -> !(M <: OptimalPlus), SIM_MODELS)
    else
        SIM_MODELS
    end
end

function do_simulate(flag=:null)
    all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!

    if flag != :nonoptimal && EXPERIMENT == "exp1"
        mdps = unique(t.m for t in flatten(values(all_trials)))
        println("Optimal simulations for each cost")
        @showprogress pmap(Iterators.product(mdps, COSTS)) do (m, cost)
            model = Optimal(cost, 1e5, 0.)
            run_simulation(model, "cost$cost-$(id(m))", [m]; n_repeat=10000)
        end
    end

    jobs = product(collect(pairs(all_trials)), pick_models(flag)) do (wid, trials), model_class
        model = deserialize("$base_path/fits/full/$(name(model_class))-$wid").model
        mdps = [t.m for t in trials]
        model, wid, mdps
    end

    println("Simulations for fitted models")
    @showprogress pmap(x->run_simulation(x...), jobs)
end


if basename(PROGRAM_FILE) == basename(@__FILE__)   
    flag = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :null
    do_simulate(flag)
end
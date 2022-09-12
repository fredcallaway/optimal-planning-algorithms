using JSON
using Random: randperm, MersenneTwister

include("conf.jl")
include("base.jl")
include("models.jl")

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

# %% ==================== Human ====================


function kfold_splits(n, k)
    @assert (n / k) % 1 == 0  # can split evenly
    x = Dict(
        :random => randperm(MersenneTwister(RANDOM_SEED), n),  # seed again just to be sure!
        :stratified => 1:n
    )[CV_METHOD]

    map(1:k) do i
        test = x[i:k:n]
        (train=setdiff(1:n, test), test=test)
    end
end

n_trial = length(all_trials |> values |> first)
folds = kfold_splits(n_trial, FOLDS)

function get_fold(i::Int)
    # folds are identified by their first test trial index
    first(f for f in folds if i in f.test).test[1]
end
get_fold(t::Trial) = get_fold(t.i)

MODELS = eval(QUOTE_MODELS)
cv_jobs = Iterators.product(values(all_trials), MODELS, folds);
cv_fits = deserialize("$base_path/cv_fits");
fit_lookup = let
    ks = map(cv_jobs) do (trials, M, fold)
        trials[1].wid, M, fold.test[1]
    end
    @assert length(ks) == length(cv_fits)
    Dict(zip(ks, getfield.(cv_fits, :model)))
end

function get_model(M::Type, t::Trial)
    fit_lookup[t.wid, M, get_fold(t)]
end

function get_preds(M::Type, t::Trial)
    model = get_model(M, t)
    map(get_data(t)) do d
        action_dist(model, d)
    end
end

demo_models = [OptimalPlus{:Default}]

function demo_trial(t)
    (
        stateRewards = t.bs[end],
        demo = (
            clicks = t.cs[1:end-1] .- 1,
            path = t.path .- 1,
            predictions = Dict(name(M) => get_preds(M, t) for M in demo_models),
            # parameters = Dict(name(M) => get_params(M, t) for M in demo_models)
        )
    )
end

function sorter(xs)
    sort(xs, by=x->(-x.score))
end

mkpath("$results_path/demo/human")


map(collect(all_trials)) do (wid, trials)
    variance_structure(trials[1].m)
end


foreach(pairs(group(x->variance_structure(x[2][1].m), collect(all_trials)))) do (v, all_trials)
    map(collect(all_trials)) do (wid, trials)
        demo_trial.(trials) |> JSON.json |> write("$results_path/demo/human/$wid.json")
        (
            wid = wid,
            variance = variance_structure(trials[1].m),
            score = mean(t.score for t in trials),
            clicks = mean(length(t.cs)-1 for t in trials),
        )
    end |> sorter |> JSON.json |> writev("$results_path/demo/human/table-$v.json")
end

# %% ==================== Optimal ====================

function model_demo_trial(t, info)
    (
        stateRewards = t.bs[end],
        demo = (
            # info = info,
            clicks = t.cs[1:end-1] .- 1,
            path = t.path .- 1,
        )
    )
end

function sorter(xs)
    sort(xs, by=x->(x.cost))
end

mkpath("$results_path/demo/optimal")

mdps = let
    all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
    flat_trials = flatten(values(all_trials));
    unique(getfield.(flat_trials, :m))
end

foreach(mdps) do m
    v = variance_structure(m)
    map(COSTS) do cost
        nam = "Optimal-cost$cost-$(id(m))"
        sims = deserialize("$base_path/sims/$nam")
        clicks, score = map(sims) do t
            [length(t.cs) - 1, term_reward(t.m, t.bs[end])]
        end |> mean
        clicks = round(clicks; digits=2); score = round(score; digits=2)

        info = (;cost, clicks, score, name=nam)
        [model_demo_trial(t, info) for t in sims[1:100]] |> JSON.json |> write("$results_path/demo/optimal/$nam.json")
        info
    end |> sorter |> JSON.json |> writev("$results_path/demo/optimal/table-$v.json")
end

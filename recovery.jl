using Distributed
isempty(ARGS) && push!(ARGS, "web")
include("conf.jl")
@everywhere begin
    using Glob
    using Serialization
    using CSV
    include("base.jl")
    include("models.jl")
end

@everywhere results_path = "$results/$EXPERIMENT"
mkpath(results_path)
mkpath("$base_path/recovery/")
mkpath("figs/recovery")

# %% ==================== Load pilot data ====================

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

# %% ==================== Fit models ====================

models = [Optimal, BestFirst, BasFirst]
all_fits = map(models) do M
    M => @time pmap(values(all_trials)) do trials
        fit(M, trials)
    end
end |> Dict

# %% ==================== Simulate models  ====================

all_sims = map(collect(all_fits)) do (M, fits)
    M => @time pmap(fits, values(all_trials)) do (model, loss), trials
        wid = string(typeof(model).name) * "-" * trials[1].wid 
        map(trials) do t
            simulate(model, t.m; wid=wid)
        end
    end
end |> Dict
serialize("$base_path/recovery/all_sims", all_sims)


# %% ==================== Update Q table with simulated data ====================

sim_data = map(values(all_sims)) do sims
    get_data(flatten(sims))
end |> flatten

include("Q_table.jl")
sim_table = make_Q_table(sim_data);
serialize("/tmp/sim_table", sim_table)
@everywhere begin
    merge!(Q_TABLE, deserialize("/tmp/sim_table"))
    features(::Type{Optimal{T}}, d::Datum) where T = Q_TABLE[hash(d)]
end
rm("/tmp/sim_table")

# %% ==================== Fit simulated data ====================

sim_fits = map(collect(all_sims)) do (M_true, sims)
    M_true => map(models) do M_fit
        M_fit => @time pmap(sims) do trials
            fit(M_fit, trials)
        end
    end |> Dict
end |> Dict
serialize("$base_path/recovery/sim_fits", sim_fits)

# %% ==================== Pull out likelihoods ====================

key = map(Iterators.product(models, models)) do (M_fit, M_true)
    # loss = sim_fits[M_true][M_fit][i][2]
    (sim=M_true, fit=M_fit)
end 

X = map(1:length(all_trials)) do i
    map(Iterators.product(models, models)) do (M_fit, M_true)
        loss = sim_fits[M_true][M_fit][i][2]
    end
end |> combinedims

total_likelihood = sum(X; dims=3) |> dropdims(3)

# %% ==================== Confusion matrix ====================

N = length(all_trials)

outcomes = map(1:N) do i
    x = X[:, :, i]
    # x[1,1] < x[1,2], x[2,2] < x[2,1]
    # [idx.I for idx in argmin(x; dims=1)]
    argmin(x; dims=1)
end |> flatten

CM = zeros(Int, length(models), length(models))
for idx in outcomes
    CM[idx] += 1
end
CM

# %% ==================== Error magnitudes ====================

error_magnitudes = map(enumerate(outcomes)) do (i, o)
    f, s = o.I
    f != s || return missing
    x = X[:, :, cld(i, 2)]
    x[s,s] - x[f,s]
end |> skipmissing |> collect
error_magnitudes

# %% ==================== Save for plotting in python ====================

using JSON

open("$results_path/recovery.json", "w+") do f
    write(f, JSON.json((
        CM=CM,
        total_likelihood=total_likelihood,
        models=string.(models)
    )))
end

# %% ==================== Save simulations for visualization ====================



function get_preds(::Type{M}, t::Trial) where M <: AbstractModel
    i = wid_index[split(t.wid, "-")[1]]
    model = all_fits[M][i][1]
    map(t.bs) do b
        action_dist(model, t.m, b)
    end
end


wid_index = Dict(w=> i for (i, w) in enumerate(keys(all_trials)))
function demo_trial(t)
    (
        stateRewards = t.bs[end],
        demo = (
            clicks = t.cs[1:end-1] .- 1,
            path = t.path .- 1,
            predictions = Dict(string(M) => get_preds(M, t) for M in models)
        )
    )
end

function sort_by_score(xs)
    sort(xs, by=x->-x.score)
end

rec = map(enumerate(keys(all_trials))) do (i, wid)
    trials = all_trials[wid]
    @assert all_sims[Optimal][i][1].wid == "Optimal-$wid"
    fits = map(models) do M
        model, nll = all_fits[M][i]
        params = Dict(fn => getfield(model, fn) for fn in fieldnames(M))
        (
            sim = demo_trial.(all_sims[Optimal][i]),
            nll = nll,
            params = params,
        )
        M => params
    end |> Dict
    (
        wid = wid,
        trials = demo_trial.(trials),
        score = mean(t.score for t in trials),
        clicks = mean(length(t.cs)-1 for t in trials),
        fits = fits
    )
end |> sort_by_score;
rec |> JSON.json |> write("$results_path/demo_viz.json")











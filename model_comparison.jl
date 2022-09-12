using Distributed
using StatsBase
using Glob
using CSV
using DataFrames
using ProgressMeter
using Random: seed!, randperm, MersenneTwister

include("conf.jl")
println("Running model comparison for ", ARGS[1])

@everywhere include("base.jl")
@everywhere include("models.jl")

seed!(RANDOM_SEED)

mkpath("$base_path/fits/full")
mkpath("$base_path/fits/cv")
mkpath("$base_path/fits/group")

redirect_worker_stderr("workerlog")

# %% ==================== LOAD DATA ====================

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
println(length(flat_trials), " trials")
all_data = all_trials |> values |> flatten |> get_data;

@assert length(unique(hash.(flat_trials))) == length(flat_trials)
@assert length(unique(hash.(all_data))) == length(all_data)


# %% ==================== FIT MODELS TO FULL DATASET ====================

MODELS = eval(QUOTE_MODELS)

if !(@isdefined(SKIP_GROUP) && SKIP_GROUP)
    @async begin
        @everywhere flat_trials = $flat_trials
        @time group_fits = pmap(MODELS) do M
            mname = name(M)
            file = "$base_path/fits/group/$mname"
            if isfile(file)
                println("$file already exists, skipping.")
                return deserialize(file)
            end
            result = fit(M, flat_trials; method=OPT_METHOD)
            serialize(file, result)
            return result
        end
        serialize("$base_path/group_fits", group_fits)
        println("Wrote $base_path/group_fits")
    end
end

# %% ==================== FIT MODELS TO INDIVIDUALS ====================

if !(@isdefined(SKIP_FULL) && SKIP_FULL)
    full_fits = let
        full_jobs = Iterators.product(values(all_trials), MODELS);
        full_fits = @showprogress pmap(full_jobs) do (trials, M)
            wid = trials[1].wid; mname = name(M)
            file = "$base_path/fits/full/$mname-$wid"
            if isfile(file)
                #println("$file already exists, skipping.")
                return deserialize(file)
            end
            try
                model, nll = fit(M, trials; method=OPT_METHOD)
                result = (model=model, nll=nll, wid=wid)
                serialize(file, result)
                return result
            catch err
                @error "Error fitting $mname to $wid" err
                rethrow()
            end

        end;
        serialize("$base_path/full_fits", full_fits)

        function mle_table(M)
            i = findfirst(MODELS .== M)
            map(zip(keys(all_trials), full_fits[:, i])) do (wid, (model, nll))
                (wid=wid, model=name(M), nll=nll, namedtuple(model)...)
            end |> DataFrame
        end

        mkpath("$results_path/mle")
        for M in MODELS
            mle_table(M) |> CSV.write("$results_path/mle/$(name(M)).csv")
        end
        
        nll = getfield.(full_fits, :nll)
        total = sum(nll; dims=1)
        best_model = [p.I[2] for p in argmin(nll; dims=2)]
        n_fit = counts(best_model, 1:length(MODELS))
        println("Model                  Likelihood   Best Fit")
        for i in eachindex(MODELS)
            @printf "%-22s       %4d         %d\n" name(MODELS[i]) total[i] n_fit[i]
        end

        full_fits
    end;
end


# %% ==================== CROSS VALIDATION ====================

if @isdefined(SKIP_GROUP) && SKIP_GROUP
    println("Skipping cross validation and prediction")
    exit()
end

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
cv_jobs = Iterators.product(values(all_trials), MODELS, folds);

cv_fits = let
    cv_fits = @showprogress pmap(cv_jobs) do (trials, M, fold)
        wid = trials[1].wid; mname = name(M); fold_i = fold.test[1]
        file = "$base_path/fits/cv/$mname-$wid-$fold_i"
        if isfile(file)
            # println("$file already exists, skipping.")
            result = deserialize(file)
            @assert result.fold == fold
            return result
        end
        try
            model, train_nll = fit(M, trials[fold.train]; method=OPT_METHOD)
            result = (model=model, train_nll=train_nll, test_nll=-logp(model, trials[fold.test]), fold=fold)
            serialize(file, result)
            return result
        catch e
            println("Error fitting $mname to $wid on fold $fold_i:  $e")
            return missing
            # (model=model, nll=NaN)
        end
    end
    serialize("$base_path/cv_fits", cv_fits)

    function cv_table(M)
        mi = findfirst(MODELS .== M)
        mapmany(enumerate(keys(all_trials))) do (wi, wid)
            map(1:FOLDS) do fi
                x = cv_fits[wi, mi, fi]
                (wid=wid, model=name(M), fold=fi, train_nll=x.train_nll, test_nll=x.test_nll, namedtuple(x.model)...)
            end
        end |> DataFrame
    end

    mkpath("$results_path/mle")
    for M in MODELS
        cv_table(M) |> CSV.write("$results_path/mle/$(name(M))-cv.csv")
    end

    # Sum over the folds
    test_nll = sum(getfield.(cv_fits, :test_nll); dims=3) |> dropdims(3);
    train_nll = sum(getfield.(cv_fits, :train_nll); dims=3) |> dropdims(3);
    train_nll ./= (FOLDS - 1);  # each trial is counted this many times

    # Sum over participants
    total_train = sum(train_nll; dims=1)
    total_test = sum(test_nll; dims=1)

    best_model = [p.I[2] for p in argmin(test_nll; dims=2)];
    n_fit = counts(best_model, 1:length(MODELS))

    println("Model                   Train NLL   Test NLL    Best Fit")
    for i in eachindex(MODELS)
        @printf "%-22s  %4d  %10d  %8d\n" name(MODELS[i]) total_train[i] total_test[i] n_fit[i]
    end
    cv_fits
end;

# %% ==================== SAVE MODEL PREDICTIONS ====================

fit_lookup = let
    ks = map(cv_jobs) do (trials, M, fold)
        trials[1].wid, M, fold.test[1]
    end
    @assert length(ks) == length(cv_fits)
    Dict(zip(ks, getfield.(cv_fits, :model)))
end

@everywhere begin
    folds = $folds
    fit_lookup = $fit_lookup

    function get_fold(i::Int)
        # folds are identified by their first test trial index
        first(f for f in folds if i in f.test).test[1]
    end
    get_fold(t::Trial) = get_fold(t.i)


    function get_model(M::Type, t::Trial)
        fit_lookup[t.wid, M, get_fold(t)]
    end

    function get_preds(M::Type, t::Trial)
        model = get_model(M, t)
        map(get_data(t)) do d
            action_dist(model, d)
        end
    end

    function get_params(M::Type, t::Trial)
        model = get_model(M, t)
        Dict(fn => getfield(model, fn) for fn in fieldnames(typeof(model)))
    end
end

preds = @showprogress "Generating predictions " pmap(all_data) do d
    (wid = d.t.wid, 
     c=d.c,
    predictions = Dict(name(M) => action_dist(get_model(M, d.t), d) for M in MODELS))
end 

preds |> JSON.json |> write("$results_path/predictions.json")


# # %% ==================== GENERATE VISUALIZATION JSON ====================

# function demo_trial(t)
#     (
#         stateRewards = t.bs[end],
#         demo = (
#             clicks = t.cs[1:end-1] .- 1,
#             path = t.path .- 1,
#             predictions = Dict(name(M) => get_preds(M, t) for M in MODELS),
#             parameters = Dict(name(M) => get_params(M, t) for M in MODELS)
#         )
#     )
# end

# function sorter(xs)
#     sort(xs, by=x->(-x.score))
# end

# mkpath("$results_path/viz")
# map(collect(all_trials)) do (wid, trials)
#     (
#         wid = wid,
#         variance = variance_structure(trials[1].m),
#         score = mean(t.score for t in trials),
#         clicks = mean(length(t.cs)-1 for t in trials),
#     )
# end |> sorter |> JSON.json |> write("$results_path/viz/table.json")

# foreach(collect(all_trials)) do (wid, trials)
#     demo_trial.(trials) |> JSON.json |> write("$results_path/viz/$wid.json")
# end


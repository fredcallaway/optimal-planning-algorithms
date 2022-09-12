using Glob
using ProgressMeter

include("conf.jl")
include("base.jl")
include("models.jl")

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
all_data = all_trials |> values |> flatten |> get_data;

MODELS = eval(QUOTE_MODELS)

mkpath("$results_path/stats")
function write_tex(name, x)
    f = "$results_path/stats/$name.tex"
    println(x, " > ", f)
    write(f, string(x, "\\unskip"))
end
write_pct(name, x; digits=1) = write_tex(name, string(round(100 * x; digits=digits), "\\%"))

# %% --------

comparison_models = ["Random", "OptimalPlus", "OptimalPlusPure"]
if !EXPAND_ONLY
    push!(comparison_models, "OptimalPlusExpand")
end

#model_sims = map([name.(MODELS); "OptimalPlusPure"]) do mname
model_sims = map(comparison_models) do mname
    mname => map(collect(keys(all_trials))) do wid
        deserialize("$base_path/sims/$mname-$wid")
    end
end |> Dict;

# %% --------
map(MODELS) do m
    name(m) => sum(length(spec) > 1 for spec in values(default_space(m)))
end |> Dict |> JSON.json |> writev("$results_path/param_counts.json")

# %% ==================== trial features ====================

path_loss(t::Trial) = term_reward(t.m, t.bs[end]) - path_value(t.m, t.bs[end], t.path)
term_reward(t::Trial) = term_reward(t.m, t.bs[end])
first_revealed(t) = t.cs[1] == 0 ? NaN : t.bs[end][t.cs[1]]

leaves(m::MetaMDP) = Set([i for (i, children) in enumerate(m.graph) if isempty(children)])
get_leaves = Dict(id(m) => leaves(m) for m in unique(getfield.(flat_trials, :m)))
is_backward(t::Trial) = t.cs[1] in get_leaves[id(t.m)]

function second_same(t)
    t.m.expand_only || return NaN
    length(t.cs) < 3 && return NaN
    c1, c2 = t.cs
    c2 in t.m.graph[c1] && return 1.
    @assert c2 in t.m.graph[1]
    return 0.
end

trial_features(t::Trial) = (
    wid=t.wid,
    i=t.i,
    term_reward=term_reward(t),
    first_revealed = first_revealed(t),
    second_same=second_same(t),
    backward=is_backward(t),
    n_click=length(t.cs)-1,
    path_loss=path_loss(t),
)

println("Computing trial features")
mkpath("$results_path/trial_features")
trial_features.(flat_trials) |> JSON.json |> writev("$results_path/trial_features/Human.json");


# @showprogress for (nam, sims) in pairs(model_sims)
for nam in comparison_models
    sims = model_sims[nam]
    f = "$results_path/trial_features/$nam.json"
    sims |> flatten .|> trial_features |> JSON.json |> writev(f)
end

# %% ==================== click features ====================

_bf = create_model(Heuristic{:Best}, [1e5, -1e5, 0], (), default_space(Heuristic{:Best}))

function click_features(d)
    m = d.t.m; b = d.b;
    pv = path_values(m, b)
    mpv = [max_path_value(m, b, p) for p in paths(m)]
    best = argmax(pv)
    max_path = maximum(mpv)
    mpv[best] = -Inf
    max_competing = maximum(mpv)
    is_best = action_dist(_bf, d) .> 1e-2

    (
        wid=d.t.wid,
        i=d.t.i,
        expanding = has_observed_parent(d.t.m.graph, d.b, d.c),
        # depth = d.c == TERM ? -1 : depth(m.graph, d.c),
        is_term = d.c == TERM,
        is_best=is_best[d.c+1],
        p_best_rand=sum(is_best) / (sum(allowed(d.t.m, d.b)) - 1),
        n_revealed=sum(observed(d.b)) - 1,
        term_reward=pv[best],
        max_path=max_path,
        max_competing=max_competing,
        best_next=best_vs_next_value(m, b),
        prob_maximal=prob_best_maximal(m, b),
        min_depth=minimum(node_depths(m)[get_frontier(m, b)]; init=100)
    )
end

println("Computing click features")
mkpath("$results_path/click_features")
click_features.(all_data) |> JSON.json |> writev("$results_path/click_features/Human.json");

for nam in comparison_models
    sims = model_sims[nam]
    f = "$results_path/click_features/$nam.json"
    sims |> flatten  |> get_data.|> click_features |> JSON.json |> writev(f)
end

# %% ==================== expansion ====================
if !flat_trials[1].m.expand_only
    mle_cost = let
        M =  OptimalPlus{:Expand,Float64}
        fits = first.(deserialize("$base_path/full_fits"))
        opt_fits = filter(x->x isa M, fits)
        Dict(keys(all_trials) .=> getfield.(opt_fits, :cost))
    end

    mle_qs(d::Datum) = Q_TABLE[shash(d)][mle_cost[d.t.wid]]

    function expansion_value(d::Datum)
        qs = mle_qs(d)[2:end]
        cs = eachindex(d.b)
        expanding = map(cs) do c
            has_observed_parent(d.t.m.graph, d.b, c)
        end
        (
            q_expand = maximum(qs[expanding]),
            q_jump = maximum(qs[.!expanding]),
            q_human = qs[d.c],
            expand = expanding[d.c],
            wid = d.t.wid
        )
    end

    all_data |> filter(d->d.c != TERM) .|> expansion_value |> JSON.json |> writev("$results_path/expansion.json")

    # for (nam, sims) in pairs(model_sims)
    #     f = "$results_path/$nam-expansion.json"
    #     sims |> flatten |> get_data  |> filter(d->d.c != TERM) .|> expansion_value |> JSON.json |> writev(f)
    # end
end


# # %% ==================== depth curve ====================
cummax(xs) = accumulate(max, xs)

function cummaxdepth(t)
    map(t.cs[1:end-1]) do c
        depth(t.m, c)
    end |> cummax
end

function depth_curve(trials)
    rows = mapmany(trials) do t
        cs = t.cs[1:end-1]
        dpth = map(cs) do c
            depth(t.m, c)
        end
        click = eachindex(cs)
        map(zip(click, dpth, cummax(dpth))) do x
            (t.wid, x...)
        end
    end
    Dict(["wid", "click", "depth", "cumdepth"] .=> invert(rows))
end

println("Computing depth curve")
mkpath("$results_path/depth_curve")
depth_curve(flat_trials) |> JSON.json |> writev("$results_path/depth_curve/Human.json");

# @showprogress for (nam, sims) in pairs(model_sims)
for nam in comparison_models
    sims = model_sims[nam]
    sims |> flatten |> depth_curve |> JSON.json |> writev("$results_path/depth_curve/$nam.json")
end

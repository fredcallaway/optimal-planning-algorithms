"Heuristic models with stopping rules based on value distributions"

using StatsFuns: logistic

# ---------- Base code for  heuristic models ---------- #

# Note: I am horribly abusing parametric types (the H parameter)
# and this makes the considerably slower due to constant compiling ... oh well
struct Heuristic{H,T} <: AbstractModel{T}
    # Selection rule weights
    β_best::T
    β_depth::T
    β_expand::T
    # Stopping rule weights
    β_satisfice::T
    β_best_next::T
    β_prob_best::T
    β_prob_better::T
    # Stopping rule thresholds
    θ_term::T
    θ_prob_better::T
    # Depth limits
    β_depthlim::T
    θ_depthlim::T
    # Pruning
    β_prune::T
    θ_prune::T
    # Lapse rate
    ε::T
end

name(::Type{<:Heuristic{H}}) where H = string(H)
name(::Type{<:Heuristic{H,T}}) where {H,T} = string(H)
name(::Heuristic{H,T}) where {H,T} = string(H)

Base.show(io::IO, model::Heuristic{H,T}) where {H, T} = print(io, "Heuristic{:$H}(...)")

function Base.display(model::Heuristic{H,T}) where {H, T}
    println("--------- Heuristic{:$H} ---------")
    space = default_space(Heuristic{H})
    for k in fieldnames(Heuristic)
        print("  ", lpad(k, 14), " = ")
        if length(space[k]) == 1
            println("(", space[k], ")")
        else
            println(round(getfield(model, k); sigdigits=3))
        end
            
    end
end

function action_dist!(p::Vector{T}, model::Heuristic{H,T}, φ::NamedTuple) where {H, T}
    term_select_action_dist!(p, model, φ)
end

function features(::Type{M}, m::MetaMDP, b::Belief) where M <: Heuristic{H,T} where {H, T}
    frontier = get_frontier(m, b)
    expansion = map(frontier) do c
        has_observed_parent(m.graph, b, c)
    end
    (
        frontier = frontier,
        expansion = expansion,
        frontier_values = node_values(m, b)[frontier] ./ MAX_VALUE,
        frontier_depths = node_depths(m)[frontier] ./ MAX_DEPTH,
        term_reward = term_reward(m, b) ./ MAX_VALUE,
        best_next = best_vs_next_value(m, b) ./ MAX_VALUE,
        prob_best = has_component(M, "ProbBest") ? prob_best_maximal(m, b) : 0,
        best_path_dists = has_component(M, "ProbBetter") ? best_paths_value_dists(m, b) : missing,
        tmp = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
        tmp2 = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
        tmp3 = zeros(T, length(frontier)),  # pre-allocate for use in selection_probability
    )
end

# ---------- Selection rule ---------- #

function pruning(model::Heuristic{H,T}, φ::NamedTuple)::Vector{T} where {H, T}
    @. begin  # weird syntax prevents unnecessary memory allocation
        1 - 
        # probability of NOT pruning
        logistic(model.β_prune * (φ.frontier_values - model.θ_prune)) * 
        logistic(model.β_depthlim * (model.θ_depthlim - φ.frontier_depths))
    end
end

@inline pruning_active(model) = (model.β_prune != 1e5) || (model.β_depthlim != 1e5)

function select_pref(model::Heuristic{H,T}, φ::NamedTuple)::Vector{T} where {H, T}
    h = φ.tmp  # use pre-allocated array for memory efficiency
    @. h = model.β_best * φ.frontier_values + model.β_depth * φ.frontier_depths + model.β_expand * φ.expansion
    h
end

@memoize function cartesian_bitvectors(N::Int)::Vector{BitVector}
    (map(collect, Iterators.product(repeat([BitVector([0, 1])], N)...)))[:]
end

function selection_probability(model::Heuristic{H, T}, φ::NamedTuple)::Vector{T} where {H, T}
    h = select_pref(model, φ)
    if !pruning_active(model)
        return softmax!(h)
    else
        # this part is the bottleneck, so we add type annotations and use explicit loops
        total::Vector{T} = fill!(φ.tmp2, 0.)
        p::Vector{T} = φ.tmp3

        p_prune = pruning(model, φ)
        for prune in cartesian_bitvectors(length(p_prune))
            all(prune) && continue
            pp = 1.
            for i in eachindex(prune)
                if prune[i]
                    p[i] = -1e10
                    pp *= p_prune[i]
                else
                    p[i] = h[i]
                    pp *= (1. - p_prune[i])
                end
            end
            total .+= pp .* softmax!(p)
        end
        total ./= (eps() + (1. - prod(p_prune)))
        # @assert all(isfinite(p) for p in total)
        return total
    end
end

# ---------- Stopping rule---------- #

function termination_probability(model::Heuristic{H,T}, φ::NamedTuple)::T where {H,T}
    p_better = ismissing(φ.best_path_dists) ? 0. : prob_better(φ.best_path_dists, model.θ_prob_better)
    
    v = model.θ_term + 
        model.β_satisfice * φ.term_reward +
        model.β_best_next * φ.best_next +
        model.β_prob_better * p_better +
        model.β_prob_best * φ.prob_best

    p_term = logistic(v)
    if pruning_active(model)
        p_term += (1-p_term) * prod(pruning(model, φ))
    end
    p_term
end

"How much better is the best path from its competitors?"
function best_vs_next_value(m, b)
    pvals = path_values(m, b)
    undetermined = map(paths(m)) do path
        any(isnan(b[i]) for i in path)
    end
    # find best path, breaking ties in favor of undetermined
    best = argmax(collect(zip(pvals, undetermined)))
    
    competing_value = if undetermined[best]
        # best path is undetermined -> competing value is the second best path (undetermined or not)
        partialsort(pvals, 2; rev=true)
    else
        # best path is determined -> competing value is the best undetermined path
        vals = pvals[undetermined]
        if isempty(vals)
            return NaN  # Doesn't matter, you have to terminate.
        else
            maximum(vals)
        end
    end
    pvals[best] - competing_value
end

"How likely is the best path to have value ≥ θ (maximize over ties)"
function prob_better(best_dists, θ)
    # linear interpolation between multiplies of 5 to make the objective smooth
    lo = 5 * fld(θ, 5); hi = 5 * cld(θ, 5)
    hi_weight = (θ - lo) / 5.; lo_weight = 1 - hi_weight
    maximum(best_dists) do bpvd
        p_worse_lo = cdf(bpvd, lo-1e-3); p_worse_hi = cdf(bpvd, hi-1e-3)
        p_worse = p_worse_lo * lo_weight + p_worse_hi * hi_weight
        1 - p_worse
    end
end

"Distributions of value of all paths with maximal expected value"
function best_paths_value_dists(m, b)
    pvals = path_values(m, b)
    max_pval = maximum(pvals)

    map(paths(m)[pvals .== max_pval]) do pth
        path_value_dist(m, b, pth)
    end |> unique
end

"How likely is the best path actually the best (maximize over ties)"
function prob_best_maximal(m, b)
    pvals = path_values(m, b)
    max_pval = maximum(pvals)
    # if multiple best paths, take the maximum probability of any of them
    maximum(zip(paths(m), pvals)) do (pth, val)
        val != max_pval && return -Inf
        unobs = filter(i->!observed(b, i), pth)
        b1 = copy(b)

        possible_unobs_vals = Iterators.product((support(m.rewards[i]) for i in unobs)...)
        sum(possible_unobs_vals) do z
            b1[unobs] .= z
            own_value = sum(b1[pth])  # same as path_value(m, b1, pth) because pth is fully observed
            cdf(max_value_dist(m, b1), own_value)
        end * (.25 ^ length(unobs))
    end
end

"Distribution of value of the best path if you knew all the rewards"
function max_value_dist(m, b)
    v = tree_value_dist(belief_tree(m, b))
    DNP([0.], [1.]) + v  # ensure it's a DNP
end

function belief_tree(m, b)
    function rec(i)
        (observed(b, i) ? b[i] : m.rewards[i], 
         Tuple(rec(child) for child in m.graph[i]))
    end
    rec(1)
end

function tree_value_dist(btree)
    self, children = btree
    isempty(children) && return self # base case
    self + maximum(map(tree_value_dist, children))
end

# ---------- Define parameter ranges for individual models ---------- #


default_space(::Type{Heuristic{:Random}}) = Space(
    :β_best => 0,
    :β_depth => 0,
    :β_expand => 0,
    :β_satisfice => 0,
    :β_best_next => 0,
    :β_prob_best => 0,
    :β_prob_better => 0,
    :θ_term => (-Inf, -10, 0, Inf),
    :θ_prob_better => 0,
    :β_depthlim => 1e5,  # flag for inactive
    :θ_depthlim => 1e10,  # Inf breaks gradient
    :β_prune => 1e5,
    :θ_prune => -1e10,
    :ε => .01,
)

function default_space(::Type{Heuristic{H}}) where H
    # note that values and depths are all scaled to (usually) be between 0 and 1
    β_pos = (0, 0, 10, Inf)  # hard lower, plausible lower, plausible upper, hard upper
    β_neg = (-Inf, -10, 0, 0)
    θ_depthlim = (-Inf, 0, 1, Inf)
    θ_prune = (-Inf, -1, 0, Inf)
    θ_prob_better = (-Inf, 0, 1, Inf)

    ranges = Dict{String,NamedTuple}(
        "Best" => (β_best=β_pos,),
        "Depth" => (β_depth=β_pos,),
        "Breadth" => (β_depth=β_neg,),
        "Satisfice" => (β_satisfice=β_pos,),
        "BestNext" => (β_best_next=β_pos,),
        "DepthLimit" => (β_depthlim=β_pos, θ_depthlim),
        "Prune" => (β_prune=β_pos, θ_prune),
        "Expand" => (β_expand=β_pos,),
        "ProbBetter" => (β_prob_better=β_pos, θ_prob_better),
        "ProbBest" => (β_prob_best=β_pos,)
    )

    space = Space(
        :β_best => 0,
        :β_depth => 0,
        :β_expand => 0,
        :β_satisfice => 0,
        :β_best_next => 0,
        :β_prob_best => 0,
        :β_prob_better => 0,
        :θ_term => (-Inf, -10, 0, Inf),
        :θ_prob_better => 0,
        :β_depthlim => 1e5,  # flag for inactive
        :θ_depthlim => 1e10,  # Inf breaks gradient
        :β_prune => 1e5,
        :θ_prune => -1e10,
        :ε => (.01, .1, .5, 1.),
    )

    for component in split(string(H), "_")
        for (k, v) in pairs(ranges[component])
            @assert k in keys(space)
            space[k] = v
        end
    end
    space
end

function pick_whistles(;exclude=String[])
    whistles = ["Satisfice", "BestNext", "ProbBetter", "ProbBest"]
    if EXPAND_ONLY
        push!(whistles, "DepthLimit", "Prune")
    else
        push!(whistles, "Expand")
    end
    filter(whistles) do x
        !(x in exclude)
    end
end


function all_heuristic_models(base = ["Best", "Depth", "Breadth"]; whistles=pick_whistles())    
    map(Iterators.product(base, powerset(whistles))) do (b, ws)
        spec = Symbol(join([b, ws...], "_"))
        Heuristic{spec}
    end[:] |> skipmissing |> collect
end

function has_component(::Type{<:Heuristic{H}}, ex) where H
    sh = string(H)
    occursin(ex, sh) || occursin("Full", sh)
end

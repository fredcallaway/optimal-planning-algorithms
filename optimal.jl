
if @isdefined(base_path) && isfile("$base_path/Q_table")
    if !@isdefined(Q_TABLE) || Q_TABLE == nothing
        const Q_TABLE = deserialize("$base_path/Q_table")
    end
else
    myid() == 1 && @warn "Q_table not found. Can't fit Optimal model"
end

get_qs(d::Datum) = valmap(Q_TABLE[shash(d)]) do q
    max.(q, NOT_ALLOWED)
end

struct Optimal{T} <: AbstractModel{T}
    cost::Float64
    β::T
    ε::T
end
name(::Optimal) = "Optimal"

default_space(::Type{M}) where M <: Optimal = Space(
    :cost => COSTS,
    :β => (1e-6, 50),
    :ε => (.01, 1)
)

features(::Type{M}, d::Datum) where M <: Optimal = (
    options = [0; get_frontier(d.t.m, d.b)],
    q = get_qs(d)
)

function action_dist!(p::Vector{T}, model::Optimal{T}, φ::NamedTuple) where T
    softmax_action_dist!(p, model, φ)
end

function preference(model::Optimal{T}, φ::NamedTuple, c::Int)::T where T
    model.β * φ.q[model.cost][c+1]
end

# For simulation

function get_qs(m::MetaMDP, b::Belief, cost::Float64)
    V = load_V_lru2(id(mutate(m, cost=cost)))
    Dict(cost => Q(V, b))
end

function action_dist(model::Optimal, m::MetaMDP, b::Belief)
    p = zeros(length(b) + 1)
    φ = (
        options = [0; get_frontier(m, b)],
        q = get_qs(m, b, model.cost)
    )
   action_dist!(p, model, φ)
end


# ---------- Souped up Optimal model ---------- #

struct OptimalPlus{H,T} <: AbstractModel{T}
    cost::Float64
    β_select::T
    β_term::T
    β_expand::T
    ε::T
end

OptimalPlus(args...) = OptimalPlus{:Default}(args...)
_optname(H) = "OptimalPlus" * (H == :Default ? "" : String(H))
name(::Type{OptimalPlus{H}}) where H = _optname(H)
name(::Type{OptimalPlus{H,T}}) where {H,T} = _optname(H)
name(::OptimalPlus{H,T}) where {H,T} = _optname(H)

default_space(::Type{OptimalPlus{:Default}}) = Space(
    :cost => COSTS,
    :β_select => (0, 0, 50, Inf),
    :β_term => (0, 0, 20, Inf),
    :β_expand => 0.,
    :ε => (.01, .1, .5, 1.)
)

default_space(::Type{OptimalPlus{:Expand}}) = 
    change_space(OptimalPlus{:Default}, β_expand=(0, 0, 50, Inf))

function features(::Type{OptimalPlus{H,T}}, d::Datum) where {H,T}
    qs = get_qs(d)
    _optplus_features(T, d.t.m, d.b, qs)
end

function _optplus_features(T, m, b, qs)
    frontier = get_frontier(m, b)
    expansion = map(frontier) do c
        has_observed_parent(m.graph, b, c)
    end
    (
        frontier=frontier, 
        expansion=expansion,
        q_select=valmap(x->x[frontier.+1], qs),
        q_term=valmap(x->x[[1; frontier.+1]], qs),
        tmp_select=zeros(T, length(frontier)),
        tmp_term=zeros(T, length(frontier)+1),
    )
end

function action_dist!(p::Vector{T}, model::OptimalPlus{H,T}, φ::NamedTuple) where {H,T}
    term_select_action_dist!(p, model, φ)
end

function selection_probability(model::OptimalPlus, φ::NamedTuple)
    q = φ.tmp_select
    @. q = model.β_select * φ.q_select[model.cost] + model.β_expand * φ.expansion
    softmax!(q)
end

function termination_probability(model::OptimalPlus, φ::NamedTuple)
    q = φ.tmp_term
    q .= model.β_term .* φ.q_term[model.cost]
    softmax!(q, 1)
end

# For simulation

function action_dist(model::OptimalPlus, m::MetaMDP, b::Belief)
    p = zeros(length(b) + 1)
    φ = _optplus_features(Float64, m, b, get_qs(m, b, model.cost))
   action_dist!(p, model, φ)
end



struct MetaGreedy{H,T} <: AbstractModel{T}
    cost::T  # note: equivalent to θ_term
    β_select::T
    β_term::T
    β_expand::T
    ε::T
end


MetaGreedy(args...) = MetaGreedy{:Default}(args...)
_mgname(H) = "MetaGreedy" * (H == :Default ? "" : String(H))
name(::Type{MetaGreedy{H}}) where H = _mgname(H)
name(::Type{MetaGreedy{H,T}}) where {H,T} = _mgname(H)
name(::MetaGreedy{H,T}) where {H,T} = _mgname(H)

default_space(::Type{MetaGreedy{:Default}}) = Space(
    :cost => (0, .05, 1, Inf),
    :β_select => (0, 0, 50, Inf),
    :β_term => (0, 0, 20, Inf),
    :β_expand => 0.,
    :ε => (.01, .1, .5, 1.)
)

default_space(::Type{MetaGreedy{:Expand}}) = 
    change_space(MetaGreedy{:Default}, β_expand=(0, 0, 50, Inf))

function voi1(m, b, c)
    mapreduce(+, results(m, b, c)) do (p, b, r)
        p * term_reward(m, b)
    end - term_reward(m, b)
end

function features(::Type{MetaGreedy{H,T}}, m::MetaMDP, b::Belief) where {H,T}
    frontier = get_frontier(m, b)
    voi = map(frontier) do c
        voi1(m, b, c)
    end
    expansion = map(frontier) do c
        has_observed_parent(m.graph, b, c)
    end
    (
        frontier=frontier,
        expansion=expansion,
        q_select=voi,
        q_term=[0.; voi],
        tmp_select=zeros(T, length(frontier)),
        tmp_term=zeros(T, length(frontier)+1),
    )
end
function action_dist!(p::Vector{T}, model::MetaGreedy{H,T}, φ::NamedTuple) where {H,T}
    term_select_action_dist!(p, model, φ)
end

function selection_probability(model::MetaGreedy, φ::NamedTuple)
    q = φ.tmp_select
    q .= model.β_select .* φ.q_select + model.β_expand * φ.expansion
    softmax!(q)
end

function termination_probability(model::MetaGreedy, φ::NamedTuple)
    q = φ.tmp_term
    q .= φ.q_term
    q[1] = model.cost  # equivalent to substracting from all the computations
    q .*= model.β_term
    softmax!(q, 1)
end

# For simulation

# function action_dist(model::MetaGreedy, m::MetaMDP, b::Belief)
#     p = zeros(length(b) + 1)
#     φ = _optplus_features(Float64, get_qs(m, b, model.cost), get_frontier(m, b))
#    action_dist!(p, model, φ)
# end
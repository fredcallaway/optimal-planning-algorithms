struct RandomSelection{T} <: AbstractModel{T}
    p_term::T
end

name(::Type{RandomSelection}) = "RandomSelection"
name(::RandomSelection) = "RandomSelection"
default_space(::Type{RandomSelection}) = Space(:p_term => (.001, 0.999))


function features(::Type{RandomSelection{T}}, m::MetaMDP, b::Belief) where T
    frontier = get_frontier(m, b)
    p_select = ones(T, length(frontier)) ./ length(frontier)
    return (frontier=frontier, p_select=p_select)
end

function action_dist!(p::Vector{T}, model::RandomSelection{T}, φ::NamedTuple) where T
    p .= 0.
    if length(φ.frontier) == 0
        p[1] = 1.
        return p
    end
    p[1] = model.p_term
    for i in eachindex(φ.p_select)
        c = φ.frontier[i] + 1
        p[c] = (1-model.p_term) * φ.p_select[i]
    end
    p
end


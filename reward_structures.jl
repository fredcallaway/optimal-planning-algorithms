function depth(g, i)
    pths = paths(g)
    i == 1 && return 0
    for d in 1:maximum(length.(pths))
        for p in pths
            p[d] == i && return d
        end
    end
    @assert false
end

function make_rewards(graph, mult, factor)
    if factor < 1
        mult *= factor ^ -(length(paths(g)[1]) - 1)
    end
    base = mult * Float64[-2, -1, 1, 2]
    map(eachindex(graph)) do i
        i == 1 && return DiscreteNonParametric([0.])
        vs = round.(unique(base .*  factor ^ (depth(graph, i)-1)))
        DiscreteNonParametric(vs)
    end
end

function make_rewards(g::Graph, variance::AbstractString, factor)
    mult = 1
    factor = Dict("increasing" => 3, "decreasing" => 1//3, "constant" => 1)[variance]
    if factor == 1
        mult *= 5
    elseif factor < 1
        mult *= factor ^ -(length(paths(g)[1]) - 1)
    end
    make_rewards(g, float(mult), factor)
end

function variance_structure(m::MetaMDP)
    initial, final = paths(m)[1][[1, end]]
    v1 = var(m.rewards[initial])
    v2 = var(m.rewards[final])
    v2 > v1 ? "increasing" : v2 < v1 ? "decreasing" : "constant"
end




# function reward_distributions(reward_structure, graph)
#     if reward_structure == "constant"
#         d = DiscreteNonParametric([-10., -5, 5, 10])
#         return repeat([d], length(graph))
#     elseif reward_structure == "decreasing"
#         dists = [
#             [0.],
#             [-48, -24, 24, 48],
#             [-8, -4, 4, 8],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-48, -24, 24, 48],
#             [-8, -4, 4, 8],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-48, -24, 24, 48],
#             [-8, -4, 4, 8],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-4.0, -2.0, 2.0, 4.0]
#         ]
#         DiscreteNonParametric.(dists)
#     elseif reward_structure == "increasing"
#         dists = [
#             [0.],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-8, -4, 4, 8],
#             [-48, -24, 24, 48],
#             [-48, -24, 24, 48],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-8, -4, 4, 8],
#             [-48, -24, 24, 48],
#             [-48, -24, 24, 48],
#             [-4.0, -2.0, 2.0, 4.0],
#             [-8, -4, 4, 8],
#             [-48, -24, 24, 48],
#             [-48, -24, 24, 48]
#         ]
#         DiscreteNonParametric.(dists)
#     elseif reward_structure == "roadtrip"
#         d = DiscreteNonParametric(Float64[-100, -50, -35, -25])
#         rewards = repeat([d], length(graph))
#         destination = findfirst(isempty, graph)
#         rewards[destination] = DiscreteNonParametric([0.])
#         return rewards
#     else
#         error("Invalid reward_structure: $reward_structure")
#     end
# end


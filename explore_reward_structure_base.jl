base_path = "tmp/explore_reward"
results_path = "results/explore_reward"
include("mdp.jl")
include("utils.jl")
include("data.jl")
include("models.jl")

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

function make_rewards(graph, mult, depth_factor)
    base = mult * Float64[-2, -1, 1, 2]
    map(eachindex(graph)) do i
        i == 1 && return DiscreteNonParametric([0.])
        vs = round.(unique(base .*  depth_factor ^ (depth(graph, i)-1)))
        DiscreteNonParametric(vs)
    end
end

function make_mdp(factor, mult)
    g = tree([4,1,2])
    if factor == 1
        mult *= 5
    elseif factor < 1
        mult *= factor ^ -(length(paths(g)[1]) - 1)
    end
    rewards = make_rewards(g, float(mult), factor)
    MetaMDP(g, rewards, 0., -Inf, true)
end

function mean_reward_clicks(pol; N=100000)
    reward, clicks = N \ mapreduce(+, 1:N) do i
        roll = rollout(pol)
        [roll.reward, roll.n_steps - 1]
    end
    (reward=reward, clicks=clicks)
end

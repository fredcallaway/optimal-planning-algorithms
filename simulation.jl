struct Simulator <: Policy
    model::AbstractModel{Float64}
    m::MetaMDP
end

(sim::Simulator)(b::Belief) = rand(Categorical(action_dist(sim.model, sim.m, b))) - 1

function best_path(m, b)
    i = map(paths(m)) do path
        path_value(m, b, path)
    end |> argmax
    paths(m)[i]
end

function simulate(pol::Policy, wid::String)
    bs = Belief[]
    cs = Int[]
    rollout(pol) do b, c
        push!(bs, deepcopy(b)); push!(cs, c)
    end
    Trial(pol.m, wid, -1, bs, cs, term_reward(pol.m, bs[end]), [], best_path(pol.m, bs[end]))
end

simulate(model::AbstractModel, m::MetaMDP; wid=name(model)) = simulate(Simulator(model, m), wid)

@everywhere include("explore_reward_structure_base.jl")

using CSV
mkpath(base_path)
mkpath(results_path)

factors = [1//2, 1//3, 2, 3]

function possible_path_values(m::MetaMDP)
    dists = [m.rewards[i] for i in paths(m)[1]]
    map(sum, Iterators.product([[d.support; 0] for d in dists]...)) |> unique
end

function possible_thresholds(m::MetaMDP)
    ppv = possible_path_values(m)
    vals = map(Iterators.product(ppv, ppv)) do (v1, v2)
        abs(v1 - v2) - .1
    end |> unique |> sort
end


classical_models = [BestFirst, BreadthFirst, DepthFirst]
jobs = mapmany(Iterators.product(classical_models, factors, mults)) do (M, factor, mult)
    m = make_mdp(factor, mult)
    θs = M == BreadthFirst ? (0.1:1:3.1) : possible_thresholds(m)
    map(θs) do θ
        (M, factor, mult, θ)
    end
end

pmap(jobs) do (M, factor, mult, θ)
    m = make_mdp(factor, mult)
    model = M(1e3, 1e3, θ, 0.)
    pol = Simulator(model, m)
    (model=string(M), factor=factor, mult=mult, threshold=θ, mean_reward_clicks(pol)...)
end |> CSV.write("$results_path/classical.csv")

idx = parse.(Int, readdir("$base_path/optimal")) |> sort
map(idx) do i
    deserialize("$base_path/optimal/$i")
end |> CSV.write("$results_path/optimal.csv")

# idx = parse.(Int, readdir("$base_path/optimal")) |> sort
glob("tmp/explore_reward/opt_results/*")
map(idx) do i
    deserialize("tmp/explore_reward/optimal/$i")
end






# %% ==================== SCRATCH ====================
idx = parse.(Int, readdir("$base_path/optimal")) |> sort
mapmany(idx) do i
    x = deserialize("$base_path/optimal/$i")
    x isa NamedTuple ? [(x..., beta=1000.)] : x
end |> CSV.write("$results_path/optimal_soft.csv")

pmap(Iterators.product(factors, mults, 0.1:1:3.1)) do (factor, mult, θ)
    m = make_mdp(factor, mult)
    model = BreadthFirst(1e3, 1e3, θ, 0.)
    pol = Simulator(model, m)
    (model="BreadthFirst", factor=factor, mult=mult, threshold=θ, mean_reward_clicks(pol)...)
end  |> CSV.write("$results_path/breadth.csv")


classical_models = [BestFirst, DepthFirst, BreadthFirst]

jobs = mapmany(Iterators.product(classical_models, factors, mults)) do (M, factor, mult)
    m = make_mdp(factor, mult)
    map(possible_thresholds(m)) do θ
        (M, factor, mult, θ)
    end
end

classical_results = pmap(jobs) do (M, factor, mult, θ)
    m = make_mdp(factor, mult)
    model = M(1e3, 1e3, θ, 0.)
    pol = Simulator(model, m)
    (model=string(M), factor=factor, mult=mult, threshold=θ, mean_reward_clicks(pol)...)
end 
classical_results |> CSV.write("$results_path/classical_soft.csv")


# %% --------

function make_mdp(factor, mult; branching=[3,1,2])
    g = tree(branching)
    if factor < 1
         mult *= factor ^ -(length(paths(g)[1]) - 1)
     end
    rewards = make_rewards(g, float(mult), factor)
    MetaMDP(g, rewards, 0., -Inf, true)
end


function solve(m::MetaMDP)
    V = ValueFunction(m)
    @time v = V(initial_belief(m))
    return V
end

V = solve(mutate(make_mdp(2, 1; branching=[3,1,2]), cost=.001))
OptimalPolicy(V) |> show_sim


function simulate(pol::Policy)
    bs = Belief[]
    cs = Int[]
    roll = rollout(pol) do b, c
        push!(bs, copy(b)); push!(cs, c)
    end
    bs, cs
end

function show_sim(pol::Policy)
    bs, cs = simulate(pol)
    for (i, c) in enumerate(cs)
        c != 0 && print(c, " ", bs[i+1][c], "   ")
    end
    return bs
end

show_sim(model, m) = show_sim(Simulator(model, m))
model = BreadthFirst(1e3, 1e3, 5., 0.)
m = make_mdp(1//3, 1)
show_sim(model, m)
m.graph

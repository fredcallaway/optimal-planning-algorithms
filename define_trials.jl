
using JSON
include("mdp.jl")
mkpath("mdps/base")

base = "/Users/fred/heroku/webofcash2/static/json"

DIRECTIONS = ["up", "right", "down", "left"]
OFFSETS = [(0, -1), (1,0), (0, 1), (-1, 0)]


graph = Dict()
layout = Dict()
 # {"0": {"up": [0, "1"], "right": [0, "5"], "down": [0, "9"], "left": [0, "13"]}, 

function diridx(dir)
    1 + (dir + 4) % 4
end

function spirals(n, turns)
    nodes = Iterators.Stateful(string.(1:1000))
    pos = [0, 0]

    function relative_to_absolute(init, turns)
        dir = init
        map(turns) do t
            dir = (dir + t) % length(DIRECTIONS)
        end
    end

    function rec(pos, dirs)
        pos = pos .+ OFFSETS[diridx(dirs[1])]
        id = first(nodes)
        layout[id] = pos
        graph[id] = Dict()
        if length(dirs) > 1
            graph[id][DIRECTIONS[diridx(dirs[2])]] = [0, rec(pos, dirs[2:end])]
        end
        id
    end

    graph = Dict()
    layout = Dict()
    pos = (0, 0)
    layout["0"] = pos
    graph["0"] = Dict(
        DIRECTIONS[diridx(dir)] => [0, rec(pos, relative_to_absolute(dir, turns))]
        for dir in 0:n-1
    )
    (layout=layout, initial="0", graph=graph)
end

write("$base/structure/41111.json", json(spirals(4, [0, 1, -1, 1, 1])))

# ---------- REWARDS ---------- #

function make_rewards(g::Graph, kind::Symbol, factor, p1, p2)
    @assert kind in [:breadth, :depth, :constant, :increasing, :decreasing]
    map(eachindex(g)) do i
        i == 1 && return DiscreteNonParametric([0.])
        if kind == :constant
            DiscreteNonParametric([-10, -5, 5, 10])
        elseif kind == :depth && isempty(g[i])
            DiscreteNonParametric([-2factor, factor], [1-p2, p2])
        elseif kind == :breadth && i in g[1]
            x = (p2 * (1 - factor) + factor) / (2factor)
            DiscreteNonParametric([-factor, 1, factor], round.([x, p2, (1 - x - p2)]; digits=6))
        else
            DiscreteNonParametric([-1, 1], [1-p1, p1])
        end
    end
end

function make_mdp(branching, kind, factor, p1, p2)
    g = tree(branching)
    rewards = make_rewards(g, kind, factor, p1, p2)
    MetaMDP(g, rewards, 0., -Inf, true)
end

function make_mdp_exp3(factor)
    graph = tree([4,1,2])
    if factor == 1
        mult = 5
    elseif factor < 1
        mult = factor ^ -(length(paths(graph)[1]) - 1)
    else
        mult = 1
    end
    base = mult .* Float64[-2, -1, 1, 2]
    rewards = map(eachindex(graph)) do i
        i == 1 && return DiscreteNonParametric([0.])
        vs = round.(unique(base .*  factor ^ (depth(graph, i)-1)))
        DiscreteNonParametric(vs)
    end
    MetaMDP(graph, rewards, 0., -Inf, false)
end

function save(m::MetaMDP)
    f = "mdps/base/$(id(m))"
    serialize(f, m)
    println("Wrote ", f)
end

function write_trials(name::String, m::MetaMDP)
    save(m)
    trials = map(1:300) do i
        rewards = rand.(m.rewards)
        tid = id(m) * "-" * string(shash(rewards); base=62)
        (trial_id=tid, stateRewards=rewards)
    end
    f = "$base/rewards/$name.json"
    write(f, json(trials))
    println("Wrote ", f)
end

if basename(PROGRAM_FILE) == basename(@__FILE__)
    # experiment 1
    write_trials("exp1_constant", make_mdp([4,1,2], :constant, NaN, NaN, NaN))
    # experiment 2
    write_trials("exp2_increasing", make_mdp([4,1,1,1,1], :depth, 20, 1/2, 2/3))
    write_trials("exp2_decreasing", make_mdp([4,1,1,1,1], :breadth, 20, 1/2, 3/5))
    write_trials("exp2_constant", make_mdp([4,1,1,1,1], :constant, NaN, NaN, NaN))
    # experiment 3
    write_trials("exp3_constant", make_mdp_exp3(1))
    write_trials("exp3_increasing", make_mdp_exp3(3))
    write_trials("exp3_decreasing", make_mdp_exp3(1/3))
    # m = MetaMDP(tree([4,1,2]), DNP([-10, -5, 5, 10]), 0., -Inf, true)
    # m1 = make_mdp([4,1,2], :constant, NaN, NaN, NaN)
end





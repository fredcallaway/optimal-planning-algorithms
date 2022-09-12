include("mdp.jl")
using JSON

const SCALING = -0.1
const MAX_COST = 300
const PRICES = [25,35,50,100]

function parse_edges(t)
    edges = map(t["edges"]) do (x, y)
        Int(x) + 1, Int(y) + 1
    end
    n_node = maximum(flatten(edges))
    graph = [Int[] for _ in 1:n_node]
    for (a, b) in edges
        push!(graph[a], b)
    end
    graph
end

# %% --------

data_path = "/Users/fred/heroku/roadtriptask/data/processed/v4.0/"
trials = open(JSON.parse, "$data_path/trials.json")

lookup = map(first(values(trials))) do t
    m = MetaMDP(parse_edges(t), DNP(SCALING .* PRICES), 0., SCALING * MAX_COST, false)
    serialize("mdps/base/$(id(m))", m) 
    t["map"] => id(m)
end |> Dict
write("$data_path/mdp_id_lookup.json", JSON.json(lookup))

# %% --------

# trials = open(JSON.parse, "/Users/fred/Projects/new-tutors/data/stage2/RTT-1.6/trials.json")
# graphs = parse_edges.(trials) |> unique
# serialize("tmp/RTT_graphs", graphs)


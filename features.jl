


"The maximum expected value of any path going through this node."
function node_values(m::MetaMDP, b::Belief)
    nv = fill(-Inf, length(m))
    for p in paths(m)
        v = path_value(m, b, p)
        for i in p
            nv[i] = max(nv[i], v)
        end
    end
    nv[1] = maximum(nv)
    nv
end

"Distance of each node from the start state."
@memoize function node_depths(m::MetaMDP)
    g = m.graph
    result = zeros(Int, length(g))
    function rec(i, d)
        result[i] = d
        for j in g[i]
            rec(j, d+1)
        end
    end
    rec(1, 0)
    result
end

"Maximum possible value of a path."
function max_path_value(m, b, path)
    d = 0.
    for i in path
        d += (observed(b, i) ? b[i] : maximum(support(m.rewards[i])))
    end
    d
end

"The maximum possible value of any path going through this node."
function max_node_values(m::MetaMDP, b::Belief)
    nv = fill(-Inf, length(m))
    for p in paths(m)
        v = max_path_value(m, b, p)
        for i in p
            nv[i] = max(nv[i], v)
        end
    end
    nv[1] = maximum(nv)
    nv
end

"Matrix of shortest path between each node"
@memoize function tree_distances(m)
    N = length(m)
    addresses = [Int[] for i in 1:N]
    for p in paths(m)
        for i in eachindex(p)
            addresses[p[i]] = p[1:i]
        end
    end

    D = zeros(Int, N, N)
    for i in 1:N, j in 1:N
        ai = addresses[i]
        aj = addresses[j]

        n_up = length(setdiff(ai, aj))
        n_down = length(setdiff(aj, ai))
        D[i, j] = n_up + n_down
    end
    D
end


@memoize function get_layout(experiment=EXPERIMENT)
    layout = Dict(
        "exp1" => "412",
        "exp2" => "41111"
    )[experiment]
    JSON.parsefile("../data/layouts/$layout.json")["layout"] |> sort |> values .|> Vector{Int}
end

euclidean_distance(x, y) = √sum((x .- y) .^2)

function screen_distances()
    layout = get_layout()
    N = length(layout)
    D = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        D[i, j] = euclidean_distance(layout[i], layout[j])
    end
    D
end

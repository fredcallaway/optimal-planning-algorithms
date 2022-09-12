
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
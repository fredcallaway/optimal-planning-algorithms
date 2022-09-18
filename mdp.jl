using Parameters
using Distributions
import Base
using Printf: @printf
using Memoize
using LRUCache
using DataStructures
using StableHashes

include("dnp.jl")

const TERM = 0  # termination action
# const NULL_FEATURES = -1e10 * ones(4)  # features for illegal computation
const N_SAMPLE = 10000
const N_FEATURE = 5

Graph = Vector{Vector{Int}}
Value = Float64

"Parameters defining a class of problems."
@with_kw struct MetaMDP
    graph::Graph
    rewards::Vector{Distribution}
    cost::Float64 = 0.
    jump_cost::Float64 = 0.
    min_reward::Float64 = -Inf
    expand_only::Bool
end

mutable struct Belief
    values::Vector{Float64}
    last_expanded::Int
end
Base.getindex(b::Belief, i) = getindex(b.values, i)
Base.length(b::Belief) = length(b.values)
Base.eachindex(b::Belief) = eachindex(b.values)
Base.copy(b::Belief) = Belief(copy(b.values), b.last_expanded)

Base.:(==)(b1::Belief, b2::Belief) = b1.values == b2.values && b1.last_expanded == b2.last_expanded

Base.show(io::IO, m::MetaMDP) = print(io, "M")
id(m::MetaMDP) = string(shash(m); base=62)

function MetaMDP(g::Graph, reward::Distribution, cost::Float64, min_reward::Float64, expand_only::Bool)
    rewards = repeat([reward], length(g))
    MetaMDP(g, rewards, cost, min_reward, expand_only)
end

function MetaMDP(g::Graph, rdist::Distribution, cost::Float64; kws...)
    rewards = repeat([rdist], length(g))
    MetaMDP(graph=g, rewards=rewards, cost=cost; kws...)
end

function StableHashes.shash(m::MetaMDP, h::UInt64)
    reduce(getfields(m); init=h) do acc, x
        if x == -Inf
            x = "-Inf"  # hash(Inf) varies by machine!
        end
        shash(x, acc)
    end
end

Base.:(==)(x1::MetaMDP, x2::MetaMDP) = struct_equal(x1, x2)
Base.length(m::MetaMDP) = length(m.graph)

initial_belief(m::MetaMDP) = Belief([0; fill(NaN, length(m)-1)], 1)
observed(b::Belief) = @. !isnan(b.values)
observed(b::Belief, c::Int) = !isnan(b[c])
unobserved(b::Belief) = [c for c in eachindex(b) if isnan(b[c])]

function tree(branching::Vector{Int})
    t = Vector{Int}[]
    function rec!(d)
        children = Int[]
        push!(t, children)
        idx = length(t)
        if d <= length(branching)
            for i in 1:branching[d]
                child = rec!(d+1)
                push!(children, child)
            end
        end
        return idx
    end
    rec!(1)
    t
end

tree(b::Int, d::Int) = tree(repeat([b], d))

function paths(g::Graph)
    frontier = [[1]]
    result = Vector{Int}[]

    function search!(path)
        loc = path[end]
        if isempty(g[loc])
            push!(result, path)
            return
        end
        for child in g[loc]
            push!(frontier, [path; child])
        end
    end
    while !isempty(frontier)
        search!(pop!(frontier))
    end
    [pth[2:end] for pth in result]
end
@memoize paths(m::MetaMDP) = paths(m.graph)

function path_value_dist(m::MetaMDP, b::Belief, path)
    d = DNP([0.], [1.])
    for i in path
        if observed(b, i)
            d += b[i]
        else
            d += m.rewards[i]
        end
    end
    m.min_reward == -Inf && return d
    map(d) do x
        max(m.min_reward, x)
    end
end

function easy_path_value(m::MetaMDP, b::Belief, path)
    d = 0.
    for i in path
        d += (observed(b, i) ? b[i] : mean(m.rewards[i]))
    end
    d
end

function path_value(m::MetaMDP, b::Belief, path)
    if m.min_reward == -Inf
        easy_path_value(m, b, path)
    else
        mean(path_value_dist(m, b, path))
    end
end

function path_values(m::MetaMDP, b::Belief)
    [path_value(m, b, path) for path in paths(m)]
end

function cost(m::MetaMDP, b::Belief, c::Int)::Float64
    if c in m.graph[b.last_expanded]
        m.cost
    else
        m.cost + m.jump_cost
    end
end

function term_reward(m::MetaMDP, b::Belief)::Float64
    mapreduce(max, paths(m)) do path
        path_value(m, b, path)
    end
end

function has_observed_parent(graph, b, c)
    any(enumerate(graph)) do (i, children)
        c in children && observed(b, i)
    end
end

function allowed(m::MetaMDP, b::Belief, c::Int)
    b.last_expanded == TERM && return false
    c == TERM && return true
    !isnan(b[c]) && return false
    !m.expand_only || has_observed_parent(m.graph, b, c)
end
allowed(m::MetaMDP, b::Belief) = [allowed(m, b, c) for c in 0:length(b)]


function results(m::MetaMDP, b::Belief, c::Int)
    @assert allowed(m, b, c)
    if c == TERM
        b1 = Belief(b.values, c)
        return [(1., b1, term_reward(m, b))]
    end

    map(support(m.rewards[c])) do v
        b1 = Belief(copy(b.values), c)
        b1.values[c] = v
        p = pdf(m.rewards[c], v)
        (p, b1, -cost(m, b, c))
    end
end

function observe!(m::MetaMDP, b::Belief, c::Int)
    @assert allowed(m, b, c)
    b.last_expanded = c
    b[c] = rand(m.rewards[c])
end

function observe!(m::MetaMDP, b::Belief, s::Vector{Float64}, c::Int)
    @assert allowed(m, b, c)
    b.last_expanded = c
    b[c] = s[c]
end

# ========== Solution ========== #

struct ValueFunction{F}
    m::MetaMDP
    hasher::F
    cache::Dict{UInt64, Float64}
end

function symmetry_breaking_hash(m::MetaMDP, b::Belief)
    the_paths = paths(m)
    lp = length(the_paths)
    sum(shash(b[pth]) >> 3 for pth in the_paths)
end

function hash_312(m::MetaMDP, b::Belief)
    shash(shash(b[2]) + shash(b[3]) >> 1, shash(b[4]) + shash(b[5])) +
    shash(shash(b[6]) + shash(b[7]) >> 1, shash(b[8]) + shash(b[9])) +
    shash(shash(b[10]) + shash(b[11]) >> 1, shash(b[12]) + shash(b[13]))
end

function hash_412(m::MetaMDP, b::Belief)
    shash(shash(b[2]) + shash(b[3]) >> 1, shash(b[4]) + shash(b[5])) +
    shash(shash(b[6]) + shash(b[7]) >> 1, shash(b[8]) + shash(b[9])) +
    shash(shash(b[10]) + shash(b[11]) >> 1, shash(b[12]) + shash(b[13])) +
    shash(shash(b[14]) + shash(b[15]) >> 1, shash(b[16]) + shash(b[17]))
end

function hash_412_iid(m::MetaMDP, b::Belief)
    shash(shash(b[2]) + shash(b[3]), shash(b[4]) + shash(b[5])) +
    shash(shash(b[6]) + shash(b[7]), shash(b[8]) + shash(b[9])) +
    shash(shash(b[10]) + shash(b[11]), shash(b[12]) + shash(b[13])) +
    shash(shash(b[14]) + shash(b[15]), shash(b[16]) + shash(b[17]))
end

function make_spiral_hasher(m)
    the_paths = paths(m)
    reward_hashes = Dict(i => shash(d) for (i, d) in enumerate(m.rewards))

    # check conditions for correctness
    @assert sum(length(pth) for pth in the_paths) == length(m) - 1
    @assert map(the_paths) do pth
        [reward_hashes[i] for i in pth]
    end |> unique |> length |> isequal(1)

    function spiral_hasher(_, b)
        acc = UInt64(0)
        for pth in the_paths
            x = UInt64(0)
            for i in pth
                x += shash(shash(b[i]), reward_hashes[i])
            end
            acc += shash(x)
        end
        acc
    end
end

default_hash(m::MetaMDP, b::Belief) = shash(b)

function choose_hash(m::MetaMDP)
    length(m) == 11 && return ((m, b) -> shash(b))  # this is a hack for the road trip mdps which have little symmetry to exploit
    if m.graph == [[2, 6, 10], [3], [4, 5], [], [], [7], [8, 9], [], [], [11], [12, 13], [], []]
        hash_312
    elseif m.graph == [[2, 6, 10, 14], [3], [4, 5], [], [], [7], [8, 9], [], [], [11], [12, 13], [], [], [15], [16, 17], [], []]
        if length(unique(m.rewards[2:end])) == 1
        # if m.rewards[2] == m.rewards[3]
            hash_412_iid
        else
            hash_412
        end
    elseif all(length(children) <=1 for children in m.graph[2:end])
        make_spiral_hasher(m)
    else
        symmetry_breaking_hash
    end
end

ValueFunction(m::MetaMDP, h) = ValueFunction(m, h, Dict{UInt64, Float64}())
ValueFunction(m::MetaMDP) = ValueFunction(m, choose_hash(m))

function Q(V::ValueFunction, b::Belief, c::Int)::Float64
    c == 0 && return term_reward(V.m, b)
    !allowed(V.m, b, c) && return -Inf 
    # !isnan(b[c]) && return -Inf  # already observed
    sum(p * (r + V(s1)) for (p, s1, r) in results(V.m, b, c))
end

Q(V::ValueFunction, b::Belief) = [Q(V,b,c) for c in 0:length(b)]

# function (V::ValueFunction)(b::Belief)::Float64
#     key = V.hasher(V.m, b)
#     return V.cache[key] = maximum(Q(V, b))
# end


# We cut runtime by a third by unrolling the Q function...
function (V::ValueFunction)(b::Belief)::Float64
    key = V.hasher(V.m, b)
    haskey(V.cache, key) && return V.cache[key]
    return V.cache[key] = step_V(V, b)
end

function step_V(V::ValueFunction, b::Belief)::Float64
    best = term_reward(V.m, b)
    @fastmath @inbounds for c in 1:length(b)
        !allowed(V.m, b, c) && continue
        val = -cost(V.m, b, c)
        R = V.m.rewards[c]
        for i in eachindex(R.p)
            v = R.support[i]; p = R.p[i]
            b1 = Belief(copy(b.values), c)
            b1.values[c] = v
            val += p * V(b1)
        end
        if val > best
            best = val
        end
    end
    best
end

function Base.show(io::IO, v::ValueFunction)
    print(io, "V")
end

function solve(m::MetaMDP, h=choose_hash(m))
    V = ValueFunction(m, h)
    V(initial_belief(m))
    V
end

function load_V_nomem(i::String)
    println("Loading V $i")
    V = deserialize("mdps/V/$i")
    ValueFunction(V.m, choose_hash(V.m), V.cache)
end

@memoize load_V(i::String) = load_V_nomem(i)
# load_V(m::MetaMDP) = load_V(id(m))

_lru() = LRU{Tuple{String}, ValueFunction}(maxsize=1)
@memoize _lru load_V_lru2(i::String) = load_V_nomem(i)



# ========== Policy ========== #

noisy(x, ε=1e-10) = x .+ ε .* rand(length(x))

abstract type Policy end

struct OptimalPolicy <: Policy
    m::MetaMDP
    V::ValueFunction
end
OptimalPolicy(V::ValueFunction) = OptimalPolicy(V.m, V)
(pol::OptimalPolicy)(b::Belief) = begin
    argmax(noisy([Q(pol.V, b, c) for c in 0:length(b)])) - 1
end


struct OptimalPolicyExpandOnly <: Policy
    m::MetaMDP
    V::ValueFunction
end
OptimalPolicyExpandOnly(V::ValueFunction) = OptimalPolicyExpandOnly(V.m, V)
(pol::OptimalPolicyExpandOnly)(b::Belief) = begin
    q = [Q(pol.V, b, c) for c in 0:length(b)]
    for c in eachindex(b)
        if !has_observed_parent(pol.m.graph, b, c)
            q[c + 1] = -Inf
        end
    end
    argmax(noisy(q)) - 1
end

struct SoftOptimalPolicy <: Policy
    m::MetaMDP
    V::ValueFunction
    β::Float64
end
SoftOptimalPolicy(V::ValueFunction, β::Float64) = SoftOptimalPolicy(V.m, V, β)
function (pol::SoftOptimalPolicy)(b::Belief)
    p = softmax!(pol.β .* Q(pol.V, b))
    rand(Categorical(p)) - 1
end

struct RandomPolicy <: Policy
    m::MetaMDP
end

(pol::RandomPolicy)(b) = rand(findall(allowed.(m, b, c)))

"Runs a Policy on a Problem."
function rollout(pol::Policy; initial=nothing, max_steps=100, callback=((b, c) -> nothing))
    m = pol.m
    b = initial != nothing ? initial : initial_belief(m)
    reward = 0
    for step in 1:max_steps
        c = (step == max_steps) ? TERM : pol(b)
        callback(b, c)
        if c == TERM
            reward += term_reward(m, b)
            return (reward=reward, n_steps=step, belief=b)
        else
            reward -= cost(m, b, c)
            observe!(m, b, c)
        end
    end
end

function rollout(callback::Function, pol::Policy; initial=nothing, max_steps=100)
    rollout(pol; initial=initial, max_steps=max_steps, callback=callback)
end

function rollout(callback::Function, pol::Policy, s::Vector{Float64}; initial=nothing, max_steps=100)
    rollout(pol, s; initial=initial, max_steps=max_steps, callback=callback)
end

function rollout(pol::Policy, s::Vector{Float64}; initial=nothing, max_steps=100, callback=((b, c) -> nothing))
    m = pol.m
    b = initial !== nothing ? initial : initial_belief(m)
    reward = 0
    for step in 1:max_steps
        c = (step == max_steps) ? TERM : pol(b)
        callback(b, c)
        if c == TERM
            reward += term_reward(m, b)
            b.last_expanded = TERM
            return (reward=reward, n_steps=step, belief=b)
        else
            reward -= cost(m, b, c)
            observe!(m, b, s, c)
        end
    end
end
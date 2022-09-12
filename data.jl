using JSON
# import DataFrames: DataFrame
using SplitApplyCombine
using Memoize

include("reward_structures.jl")

struct Trial
    m::MetaMDP  # Note: cost must be NaN (or maybe 0??)
    wid::String
    i::Int
    bs::Vector{Belief}
    cs::Vector{Int}
    score::Float64
    rts::Vector  # these are not actually used
    path::Vector{Int}
end
StableHashes.shash(t::Trial, h::UInt64) = shash(t.wid, shash(t.i, h))

# Base.:(==)(x1::Trial, x2::Trial) = struct_equal(x2, x2)  # doesn't work because NaN ≠ NaN

struct Datum
    t::Trial
    b::Belief
    c::Int
    # c_last::Union{Int, Nothing}
end

StableHashes.shash(d::Datum, h::UInt64) = shash(d.c, shash(d.t, h))
# Base.:(==)(x1::Datum, x2::Datum) = struct_equal(x2, x2)

is_roadtrip(t::Dict) = startswith(get(t, "map", ""), "fantasy")

function get_mdp(t::Dict)
    return _load_mdp(t["mdp"])
end

@memoize function _load_mdp(mdp_id)
    m = deserialize("mdps/base/$mdp_id")
    mutate(m, expand_only=EXPAND_ONLY)
end


function Trial(wid::String, i::Int, t::Dict{String,Any})
    m = get_mdp(t)
    # graph = parse_graph(t)

    bs = Belief[]
    cs = Int[]
    b = initial_belief(m)

    for (c, value) in t["reveals"]
        c += 1  # 0->1 indexing
        push!(bs, copy(b))
        push!(cs, c)

        # we ignore the case when c = 1 because the model assumes this
        # value is known to be 0 (it is irrelevant to the decision).
        # it actually shouldn't be allowed in the experiment...
        if c != 1
            b[c] = value
        end
    end
    push!(bs, b)
    push!(cs, TERM)
    path = Int.(t["route"] .+ 1)[2:end]
    # rts = [x == nothing ? NaN : float(x) for x in t["rts"]]
    rts = Float64[]
    Trial(m, wid, i, bs, cs, get(t, "score", NaN), rts, path)
end

# this is memoized for the sake of future memoization based on object ID
function get_data(t::Trial)
    map(eachindex(t.bs)) do i
        # c_last = i == 1 ? nothing : t.cs[i-1]
        # tmp = zeros(length(t.bs)+1)
        Datum(t, t.bs[i], t.cs[i])
    end
end

get_data(trials::Vector{Trial}) = flatten(map(get_data, trials))

function Base.show(io::IO, t::Trial)
    print(io, "T")
end

# function load_params(experiment)::DataFrame
#     x = open(JSON.parse, "../data/$experiment/params.json")
#     DataFrame(map(namedtuple, x))

# end

function load_trials(experiment)::Dict{String,Vector{Trial}}
    data = open(JSON.parse, "../data/$experiment/trials.json")
    data |> values |> first |> first
    map(data) do wid, tt
        wid => [Trial(wid, i, t) for (i, t) in enumerate(tt)]
    end |> Dict
end




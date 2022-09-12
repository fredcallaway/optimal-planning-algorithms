using Distributions
using DataStructures: DefaultDict
using Memoize

DNP = DiscreteNonParametric

DiscreteNonParametric(x) = DNP(x, ones(length(x)) ./ length(x))

function (Base.:+)(d1::DNP, d2::DNP)::DNP
    res = DefaultDict{Float64,Float64}(0.)
    for (p1, v1) in zip(d1.p, d1.support)
        for (p2, v2) in zip(d2.p, d2.support)
            res[v1 + v2] += p1 * p2
        end
    end
    DNP(collect(keys(res)), collect(values(res)))
end

function (Base.:+)(d::DNP, x::Real)::DNP
    DNP(d.support .+ x, d.p, )
end
(Base.:+)(x::Real, d::DNP) = d + x

function Base.max(d::DNP, x::Real)
    map(d) do dx
        max(dx, x)
    end
end
Base.max(x::Real, d::DNP) = max(d, x)

function Base.max(d1::DNP, d2::DNP)::DNP
    res = DefaultDict{Float64,Float64}(0.)
    for (p1, v1) in zip(d1.p, d1.support)
        for (p2, v2) in zip(d2.p, d2.support)
            res[max(v1, v2)] += p1 * p2
        end
    end
    DNP(collect(keys(res)), collect(values(res)))
end

function Base.map(f, d::DNP)::DNP
    res = DefaultDict{Float64,Float64}(0.)
    for (p, v) in zip(d.p, d.support)
        res[f(v)] += p
    end
    DNP(collect(keys(res)), collect(values(res)))
end

@memoize function sum_many(d::DNP, n::Int)::DNP
    n == 1 ? d : sum_many(d, n-1) + d
end

StableHashes.shash(d::DNP, h::UInt64) = hash_struct(d, h)

#=
def cross(dists, f=None):
    if f is None:
        f = lambda *x: x
    outcomes = Counter()
    for outcome_probs in it.product(*dists):
        o, p = zip(*outcome_probs)
        outcomes[f(*o)] += reduce(lambda x, y: x*y, p)

    return Categorical(outcomes.keys(), outcomes.values())



__no_default__ = 25

# @lru_cache(maxsize=None)
def cmax(dists, default=__no_default__):
    dists = tuple(dists)
    if len(dists) == 1:
        return dists[0]
    elif len(dists) == 0:
        if default is not __no_default__:
            return default
        else:
            raise ValueError('dmax() arg is an empty sequence')
    else:
        return cross(dists, max)

=#

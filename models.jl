using DataStructures: OrderedDict
using Optim
using Sobol

const NOT_ALLOWED = -1e20

include("space.jl")
include("abstract_model.jl")
include("likelihood.jl")
include("simulation.jl")
include("features.jl")

include("random.jl")
include("heuristic.jl")
include("optimal.jl")
include("metagreedy.jl")
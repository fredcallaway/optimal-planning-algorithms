isempty(ARGS) && push!(ARGS, "")
ARGS[1] = "exp1"
include("conf.jl")
@everywhere begin
    include("utils.jl")
    include("mdp.jl")
    include("data.jl")
    include("models.jl")
end

all_trials = load_trials(EXPERIMENT) |> OrderedDict |> sort!
flat_trials = flatten(values(all_trials));
trials = first(values(all_trials))
data = get_data(trials);
nothing

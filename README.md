# Code for "Rational use of cognitive resources in human planning"

This is the core modeling code supporting this paper: https://doi.org/10.1038/s41562-022-01332-8

Some non-essential code is missing from this repo. If you can't find something you want, check [osf](https://osf.io/rqxja). You can find experimental data and python analysis code there.

# Dependendices

The code was run using **julia v1.4.1**. Loading any serialized files will only work with 1.4, but otherwise, newer versions should work fine. You can install dependencies with `julia install_deps.jl` (I didn't know about Project.toml files at this stage in my julia career).

# Steps to replicate

In principle, running `bash steps.sh exp2` will generate all the modeling results for experiment 2. I don't think I've ever actually done it this way though because some steps take a really long time.

## Configure

Create or edit a file like conf/exp1.jl to specify things like which models to fit and the name of the experiment (EXPERIMENT specifies the path to data as well as result files. The full data path is specified in `load_trials` in data.jl.

## Solve meta MDPs

This line solves the meta MDPs in mdps/base. If you aren't using my task structure, then you will need to write your own MDPs. See `define_trials.jl` for an example (search for mdps/base).
```
julia -p auto solve.jl all
```
Note that for Experiment 3, this requires a huge amount of RAM, around 150 GB per process. It also takes quite a long time.

You can also solve on a cluster:
```
julia solve.jl setup
sbatch solve.sbatch
```

## Precompute Q values for experimental data

This step precomputes Q values for the clicks people made in the experiment. It's pretty fast as long as the value function is not really big (as it is in Experiment 3).

```
julia -p 8 Q_table.jl $exp
```

If you have a very large MDP or run out of memory, then you might have better luck with 
```
julia -p 4 Q_table_highmem.jl 
```

You can adjust the number of processes to use as much RAM as you have available (Experiment 3 requires ~150GB per process).

## Fiting

This fits the models and 

```
julia -p auto model_comparison.jl $exp
```


## Analyses

These lines simulate the fitted models to generate data to compare with humans and then compute some features on it for downstream analysis.

```
julia -p auto simulate.jl $exp
julia analysis.jl $exp
julia best_first_rate.jl $exp  # only needed for exp1
```

using Glob, HDF5, SimilaritySearch, JSON, CSV, DataFrames, Dates

parse_time(d) = DateTime(replace(d, " CEST"  => ""), dateformat"e dd u yyyy HH:MM:SS p")

function report_task2(D, team, file="results-task2/gooaq/task2/root_join.h5", key="knns", slice=1:15, k=15)
    knns, buildtime, querytime, algo, params = h5open(file) do f
        A = attributes(f)
        for k_ in keys(A)
            @info k_ => A[k_][]
        end
        f[key][slice, :], A["buildtime"][], A["querytime"][], A["algo"][], A["params"][]
    end
    knns = [Set(c) for c in eachcol(knns)]

    gold = h5open("/home/sisap23evaluation/data2025/benchmark-eval-gooaq.h5") do f
        f["allknn/knns"][1:16, :]
    end
    gold = [Set(filter(j -> i != j, c)) for (i, c) in enumerate(eachcol(gold))]

    recall = macrorecall(gold, knns)

    begins = ""
    ends = ""
    for line in eachline("log-task2.txt")
        m = match(r"^==== RUN BEGINS (.+)", line)
        if m !== nothing
            begins = m.captures[1]
            continue
        end
        m = match(r"^==== RUN ENDS (.+)", line)
        if m !== nothing
            ends = m.captures[1]
            continue
        end
    end
    
    begins, ends = parse_time(begins), parse_time(ends)
    total_time = (ends - begins).value / 1000
    task = "task2"

    push!(D, (; team, algo, task, k, recall, buildtime, querytime, params, begins, ends, total_time))
end


D = DataFrame(; team=[], algo=[], task=[], k=[], recall=[], buildtime=[], querytime=[], params=[], begins=[], ends=[], total_time=[])
report_task2(D, "DCC-UChile")
CSV.write("results-task2.csv", D)

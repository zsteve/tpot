using Ripserer
using DelimitedFiles, DataFrames, CSV
using JSON

# precompile


function return_points(representatives)
    return [Set(i for e in es for i in e) for es in representatives]
end

precomp_pc = rand(10, 2) |> eachrow .|> Tuple
ripserer(precomp_pc; dim_max=1, alg=:involuted)


input_data = "pointcloud.tsv"

grid = Matrix{Float64}(CSV.read(input_data, DataFrame) )|> eachrow .|> Tuple



PH = ripserer(grid; dim_max=1, alg=:involuted)


barcodes(PH, dim::Int) = hcat(collect.(collect(PH[dim+1]))...)'
representatives(PH, dim::Int) = [[collect(r.simplex) for r in collect(c)] for c in representative.(PH[dim+1])]

Representatives = representatives(PH,1)
#repre = Representatives
repre = return_points(Representatives)

dic = Dict{String,Dict}()

dic = Dict(:barcode => barcodes(PH,1), :representatives => repre)


open("PH.json", "w") do io JSON.print(io, dic, 2) end

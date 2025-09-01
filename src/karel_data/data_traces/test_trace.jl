using NPZ
function load_packed_with_traces(npz_path::AbstractString)
    d = npzread(npz_path)
    inputs        = d["inputs"]         # [N,H,W,C]
    outputs       = d["outputs"]        # [N,H,W,C]
    trace_frames  = d["trace_frames"]   # [T,H,W,C]
    trace_offsets = d["trace_offsets"]  # [N,2] (start,len)

    N = size(inputs, 1)
    exs = Vector{IOExample{KarelState,
                           KarelState}}(undef, N)
    for i in 1:N
        in_state  = array_to_state( convert(Array{Int8,3}, @view inputs[i, :, :, :]) )
        out_state = array_to_state( convert(Array{Int8,3}, @view outputs[i, :, :, :]) )
        exs[i] = IOExample(Dict(:_arg_1 => in_state), out_state)
    end

    tr = Vector{Vector{KarelState}}(undef, N)
    for i in 1:N
        s = Int(trace_offsets[i,1]) + 1
        L = Int(trace_offsets[i,2])
        fr = Vector{KarelState}(undef, L)
        for j in 0:L-1
            arr = convert(Array{Int8,3}, @view trace_frames[s+j, :, :, :])
            fr[j+1] = array_to_state(arr)
        end
        tr[i] = fr
    end
    return exs, tr
end
exs, traces = load_packed_with_traces("val_packed.npz")

# peek at field names once
@show propertynames(exs[1])

# version-agnostic access
function example_inputs(ex)
    for name in propertynames(ex)
        val = getfield(ex, name)
        if val isa Dict{Symbol, KarelState}
            return val
        end
    end
    error("No Dict{Symbol,KarelState} field on IOExample")
end

# checks
@assert traces[1][1] == example_inputs(exs[1])[:_arg_1]
@assert !isempty(traces[1])  # has at least the initial frame
println("OK: traces align with the input state.")

using NPZ
using HerbBenchmarks, HerbSpecification

# ---------------- Karel types & converters (yours, unchanged) ----------------
@enum Direction begin
    NORTH = 1; EAST = 2; SOUTH = 3; WEST = 4
end

mutable struct Hero
    position::Tuple{Int,Int}
    direction::Direction
    marker_count::Int
end
Hero(position::Tuple{Int,Int}, direction::Direction) = Hero(position, direction, 0)
Base.deepcopy(hero::Hero) = Hero(hero.position, hero.direction, hero.marker_count)
Base.:(==)(h1::Hero, h2::Hero) = h1.position == h2.position && h1.direction == h2.direction
Base.hash(h::Hero, h0::UInt) = hash(h.position, h0) ⊻ hash(h.direction, h0) ⊻ hash(h.marker_count, h0)

mutable struct KarelState
    world::Matrix{Bool}
    markers::Dict{Tuple{Int,Int},Int}
    hero::Hero
    KarelState(world::Matrix{Bool}, hero::Hero) = new(world, Dict{Tuple{Int,Int},Int}(), hero)
    KarelState(world::Matrix{Bool}, markers::Dict{Tuple{Int,Int},Int}, hero::Hero) = new(world, markers, hero)
end
Base.deepcopy(state::KarelState) = KarelState(state.world,
    Dict(deepcopy(k) => deepcopy(v) for (k, v) in state.markers), deepcopy(state.hero))
Base.:(==)(s1::KarelState, s2::KarelState) = s1.markers == s2.markers && s1.hero == s2.hero
Base.hash(s::KarelState, h0::UInt) = hash(s.world, h0) ⊻ hash(s.markers, h0) ⊻ hash(s.hero, h0)

const HERO_CHARS = ['<', '^', '>', 'v']; const MARKER_CHAR = 'o'
const WALL_CHAR = '#'; const EMPTY_CHAR = '.'

const DIRECTION_TO_VECTOR = Dict(NORTH => (0, -1), EAST => (1, 0), SOUTH => (0, 1), WEST => (-1, 0))
const DIRECTION_TO_ARR_IDX = Dict(NORTH => 0, SOUTH => 1, WEST => 2, EAST => 3)
const ARR_IDX_TO_DIRECTION = Dict(0 => NORTH, 1 => SOUTH, 2 => WEST, 3 => EAST)

function Base.show(io::IO, state::KarelState)
    height, width = size(state.world)
    display = fill(EMPTY_CHAR, height, width)
    for y in 1:height, x in 1:width
        if state.world[y, x]; display[y, x] = WALL_CHAR; end
    end
    for (pos, _) in state.markers
        x, y = pos; display[y, x] = MARKER_CHAR
    end
    hero_x, hero_y = state.hero.position
    dir = state.hero.direction
    hero_char = dir == WEST ? HERO_CHARS[1] : dir == NORTH ? HERO_CHARS[2] :
                dir == EAST ? HERO_CHARS[3] : HERO_CHARS[4]
    display[hero_y, hero_x] = hero_char
    println(io, "┌" * "─"^width * "┐")
    for row in 1:height
        print(io, "│"); for col in 1:width; print(io, display[row, col]); end; println(io, "│")
    end
    println(io, "└" * "─"^width * "┘")
end

function create_world(height::Int, width::Int)::Matrix{Bool}
    world = fill(false, height, width)
    world[1, :] .= true; world[end, :] .= true; world[:, 1] .= true; world[:, end] .= true
    world
end

function create_random_world(height::Int, width::Int, wall_ratio::Float64=0.1)::Matrix{Bool}
    world = create_world(height, width)
    for i in 2:height-1, j in 2:width-1
        if rand() < wall_ratio; world[i, j] = true; end
    end
    world
end

function state_to_array(state::KarelState)::Array{Int8,3}
    height, width = size(state.world)
    array = zeros(Int8, height, width, 16)
    hero_x, hero_y = state.hero.position
    dir_idx = DIRECTION_TO_ARR_IDX[state.hero.direction]
    array[hero_y, hero_x, dir_idx+1] = 1
    array[:, :, 5] = state.world
    for y in 1:height, x in 1:width
        pos = (x, y); num_markers = get(state.markers, pos, 0)
        channel = min(num_markers + 6, 16)
        array[y, x, channel] = 1
    end
    array
end

function array_to_state(array::Array{Int8,3})::KarelState
    height, width, _ = size(array)
    world = Matrix{Bool}(Bool.(array[:, :, 5] .> 0.5))
    hero_y, hero_x = Tuple(findfirst(sum(view(array, :, :, 1:4), dims=3)[:, :, 1] .> 0.5))
    array_idx = findfirst(view(array, hero_y, hero_x, 1:4) .> 0.5) - 1
    dir = ARR_IDX_TO_DIRECTION[array_idx]
    hero = Hero((hero_x, hero_y), dir)
    markers = Dict{Tuple{Int,Int},Int}()
    for y in 1:height, x in 1:width
        marker_channel = findfirst(view(array, y, x, 6:16) .> 0.5)
        if !isnothing(marker_channel)
            num_markers = marker_channel - 1
            if num_markers > 0; markers[(x, y)] = num_markers; end
        end
    end
    KarelState(world, markers, hero)
end

# ---------------- loader for *packed* NPZ (numeric only) ----------------
function load_packed_with_traces(npz_path::AbstractString)
    d = npzread(npz_path)
    inputs        = d["inputs"]         # [N,H,W,C]
    outputs       = d["outputs"]        # [N,H,W,C]
    trace_frames  = d["trace_frames"]   # [T,H,W,C]
    trace_offsets = d["trace_offsets"]  # [N,2] (start,len)

    N = size(inputs, 1)
    exs = Vector{IOExample{KarelState,KarelState}}(undef, N)
    for i in 1:N
        in_state  = array_to_state(convert(Array{Int8,3}, @view inputs[i, :, :, :]))
        out_state = array_to_state(convert(Array{Int8,3}, @view outputs[i, :, :, :]))
        exs[i] = IOExample(Dict(:_arg_1 => in_state), out_state)
    end

    tr = Vector{Vector{KarelState}}(undef, N)
    for i in 1:N
        s = Int(trace_offsets[i,1]) + 1
        L = Int(trace_offsets[i,2])
        frames = Vector{KarelState}(undef, L)
        for j in 0:L-1
            arr = convert(Array{Int8,3}, @view trace_frames[s+j, :, :, :])
            frames[j+1] = array_to_state(arr)
        end
        tr[i] = frames
    end
    return exs, tr
end

# ---------------- run & sanity checks (FIXED FIELD NAMES) ----------------
exs, traces = load_packed_with_traces("val_packed.npz")

# show actual field names
@show propertynames(exs[1])  # should print (:in, :out)

# use the correct names here:
@assert traces[1][1]  == exs[1].in[:_arg_1]  # initial frame equals input state
@assert traces[1][end] == exs[1].out         # last frame equals output state
println("OK: traces align with :in and :out.")

# (Optional) pretty display of first example
function show_example(ex::IOExample{KarelState,KarelState}, tr::Vector{KarelState}; maxframes=5)
    println("\nInput:");  show(ex.in[:_arg_1])
    println("\nA few trace frames:")
    for (t, st) in enumerate(tr[1:min(maxframes, length(tr))])
        println("\n--- frame $t ---"); show(st)
    end
    println("\nOutput:"); show(ex.out)
end
show_example(exs[1], traces[1]; maxframes=3)

module ParseKarel
export ParseTree, parse_llm_response, parse_llm_response_with_warnings, get_warnings, reset_warnings!

# ---------- Tree ----------
mutable struct ParseTree
    rule::String
    children::Vector{ParseTree}
end

# ---------- Warnings ----------
const _WARNINGS = Ref(Vector{String}())

reset_warnings!() = (_WARNINGS[] = String[]; nothing)
get_warnings() = _WARNINGS[]  # read-only snapshot

# ---------- Grammar vocab (canonical forms) ----------
const ACTIONS    = Set(["move","turnLeft","turnRight","pickMarker","putMarker"])
const COND_ATOMS = Set(["frontIsClear","leftIsClear","rightIsClear","markersPresent","noMarkersPresent"])
const CTRL_KEYS  = ["IFELSE","IF","WHILE","REPEAT","NOT"]  # order matters (IFELSE before IF)
const CTRL_SET   = Set(CTRL_KEYS)

# ---------- Case-insensitive helpers / maps ----------
const ACTIONS_LC      = Set(lowercase.(collect(ACTIONS)))
const COND_ATOMS_LC   = Set(lowercase.(collect(COND_ATOMS)))
const CTRL_KEYS_LC    = [lowercase(k) for k in CTRL_KEYS]
const CTRL_SET_LC     = Set(CTRL_KEYS_LC)

const ACTION_CANON    = Dict(lowercase(a) => a for a in ACTIONS)
const COND_CANON      = Dict(lowercase(c) => c for c in COND_ATOMS)
const CTRL_CANON      = Dict(lowercase(k) => k for k in CTRL_KEYS) # returns canonical UPPERCASE key

# ---------- Helpers ----------
# Return ( "(...)", idx_after ) for a balanced parenthesis group starting at s[open_idx] == '('
# If unbalanced, returns up to end and points after end (best-effort, never throws).
function take_balanced_call(s::String, open_idx::Int)
    depth = 0
    j = open_idx
    n = lastindex(s)
    while j <= n
        c = s[j]
        depth += (c == '(') ? 1 : 0
        depth -= (c == ')') ? 1 : 0
        if depth == 0
            return (s[open_idx:j], nextind(s, j))
        end
        j = nextind(s, j)
    end
    return (s[open_idx:end], n + 1)
end

# Split by a separator at top level (ignoring parentheses)
function split_top_level(s::AbstractString, sep::Char)
    depth = 0
    parts = String[]
    start = firstindex(s)
    i = start
    n = lastindex(s)
    while i <= n
        c = s[i]
        if c == '('; depth += 1
        elseif c == ')'; depth -= 1
        elseif c == sep && depth == 0
            push!(parts, strip(s[start:prevind(s, i)]))
            start = nextind(s, i)
        end
        i = nextind(s, i)
    end
    push!(parts, strip(s[start:end]))
    return parts
end

# Match literal `pat` at position `i` in `s` (CASE-INSENSITIVE). Returns index after the match, or `nothing`.
@inline function match_at_ci(s::String, i::Int, pat::String)
    j = i
    for c in pat
        if j > lastindex(s) || lowercase(s[j]) != lowercase(c)
            return nothing
        end
        j = nextind(s, j)
    end
    return j
end

# ---------- Lexical segmentation by grammar literals (case-insensitive) ----------
# Discover units (Actions, Control keywords with (...) , Condition atoms) without needing semicolons.
function lexical_segments(s::AbstractString)
    segs = String[]
    i = firstindex(s)
    n = lastindex(s)

    function match_literal_at(s::String, i::Int)
        # Control keywords (IFELSE before IF), case-insensitive
        for key in CTRL_KEYS
            j2 = match_at_ci(s, i, key)
            if j2 !== nothing && j2 <= lastindex(s) && s[j2] == '('
                par, j3 = take_balanced_call(s, j2)
                # canonicalize key to uppercase form, keep original parens content
                return (string(key, par), j3)
            end
        end
        # Actions (case-insensitive)
        for a in ACTIONS
            j2 = match_at_ci(s, i, a)
            if j2 !== nothing
                # return canonical action spelling
                return (a, j2)
            end
        end
        # Condition atoms (we’ll warn/skip if they appear at top level), case-insensitive
        for c in COND_ATOMS
            j2 = match_at_ci(s, i, c)
            if j2 !== nothing
                return (c, j2)
            end
        end
        return nothing
    end

    while i <= n
        c = s[i]
        if c in (' ', '\t', '\r', '\n', ',',';')
            i = nextind(s, i); continue
        end
        m = match_literal_at(s, i)
        if m !== nothing
            (tok, j2) = m
            push!(segs, tok)
            i = j2
        else
            # Skip unknown junk until whitespace/separator or a known literal starts
            bad_start = i
            i = nextind(s, i)
            while i <= n
                c2 = s[i]
                if c2 in (' ', '\t', '\r', '\n', ',',';'); break; end
                # try match again at new position
                if match_literal_at(s, i) !== nothing; break; end
                i = nextind(s, i)
            end
            bad = strip(s[bad_start:prevind(s, i)])
            if !isempty(bad)
                push!(_WARNINGS[], "Unrecognized text skipped: \"$bad\".")
            end
        end
    end
    return segs
end

# Minimal tokenizer used by condition parsing (not offset-based)
function tokenize(src::AbstractString)
    s = String(src)
    i, n = firstindex(s), lastindex(s)
    toks = String[]
    while i <= n
        c = s[i]
        if c in (' ', '\t', '\r', '\n', ',', ';')
            i = nextind(s, i); continue
        end
        if isletter(c)
            j = i
            while j <= n && (isletter(s[j]) || isdigit(s[j]))
                j = nextind(s, j)
            end
            word = s[i:j-1]
            wlc = lowercase(word)
            if (wlc in CTRL_SET_LC) && j <= n && s[j] == '('
                par, j2 = take_balanced_call(s, j)
                # push canonical uppercase CTRL key
                push!(toks, string(CTRL_CANON[wlc], par))
                i = j2
            else
                push!(toks, word)  # raw; consumers will lowercase if needed
                i = j
            end
            continue
        end
        if isdigit(c)
            j = i
            while j <= n && isdigit(s[j]); j = nextind(s, j) end
            push!(toks, s[i:j-1]); i = j; continue
        end
        i = nextind(s, i)
    end
    return toks
end

# ---------- Condition / Action / INT ----------
function parse_action_node(seg::AbstractString)
    t = String(strip(seg))
    if endswith(t, "()"); t = t[1:end-2]; end
    tlc = lowercase(t)
    if tlc in ACTIONS_LC
        return ParseTree("Action->$(ACTION_CANON[tlc])", ParseTree[])
    end
    return nothing
end

function parse_condition(s::AbstractString)
    s = String(strip(s))
    slc = lowercase(s)
    if slc in COND_ATOMS_LC
        return ParseTree("Condition->$(COND_CANON[slc])", ParseTree[])
    end
    if startswith(uppercase(s), "NOT(") && endswith(s, ")")
        inner = s[5:end-1]  # "NOT(" is 4 chars
        inner_tree = parse_condition(inner)
        return inner_tree === nothing ? nothing : ParseTree("Condition->NOT(Condition)", [inner_tree])
    end
    toks = tokenize(s)
    if length(toks) == 1
        t = toks[1]
        tlc = lowercase(t)
        if tlc in COND_ATOMS_LC
            return ParseTree("Condition->$(COND_CANON[tlc])", ParseTree[])
        elseif startswith(uppercase(t), "NOT(") && endswith(t, ")")
            inner = t[5:end-1]
            inner_tree = parse_condition(inner)
            return inner_tree === nothing ? nothing : ParseTree("Condition->NOT(Condition)", [inner_tree])
        end
    end
    return nothing
end

function parse_int_node(tok::AbstractString)
    t = String(strip(tok))
    if all(isdigit, t)
        v = try parse(Int, t) catch; nothing end
        if v !== nothing && 1 <= v <= 5
            return ParseTree("INT->$v", ParseTree[])
        end
    end
    return nothing
end

# ---------- ControlFlow (case-insensitive keywords) ----------
function parse_controlflow_node(seg::AbstractString)
    t = String(strip(seg))
    Tu = uppercase(t)

    if startswith(Tu, "IF(") && endswith(t, ")")
        inner = t[4:end-1]
        parts = split_top_level(inner, ',')
        if length(parts) == 2
            c  = parse_condition(parts[1])
            bl = parse_block_string(parts[2])
            if c !== nothing && bl !== nothing
                return ParseTree("ControlFlow->IF(Condition, Block)", [c, bl])
            end
        elseif length(parts) == 3
            # --- tolerate 3-arg IF by treating it as IFELSE ---
            c  = parse_condition(parts[1])
            b1 = parse_block_string(parts[2])
            b2 = parse_block_string(parts[3])
            if c !== nothing && b1 !== nothing && b2 !== nothing
                return ParseTree("ControlFlow->IFELSE(Condition, Block, Block)", [c, b1, b2])
            end
        end
        return nothing
    end

    if startswith(Tu, "IFELSE(") && endswith(t, ")")
        inner = t[8:end-1]
        parts = split_top_level(inner, ',')
        if length(parts) == 3
            c  = parse_condition(parts[1])
            b1 = parse_block_string(parts[2])
            b2 = parse_block_string(parts[3])
            if c !== nothing && b1 !== nothing && b2 !== nothing
                return ParseTree("ControlFlow->IFELSE(Condition, Block, Block)", [c, b1, b2])
            end
        end
        return nothing
    end

    if startswith(Tu, "WHILE(") && endswith(t, ")")
        inner = t[7:end-1]
        parts = split_top_level(inner, ',')
        if length(parts) == 2
            c  = parse_condition(parts[1])
            bl = parse_block_string(parts[2])
            if c !== nothing && bl !== nothing
                return ParseTree("ControlFlow->WHILE(Condition, Block)", [c, bl])
            end
        end
        return nothing
    end

    if startswith(Tu, "REPEAT(") && endswith(t, ")")
        inner = t[8:end-1]
        parts = split_top_level(inner, ',')
        if length(parts) == 2
            rpart = replace(strip(parts[1]), " " => "")
            rtok  = occursin('=', rpart) ? split(rpart, '='; limit=2)[2] : rpart
            rnode = parse_int_node(rtok)
            bl    = parse_block_string(parts[2])
            if rnode !== nothing && bl !== nothing
                return ParseTree("ControlFlow->REPEAT(R=INT, Block)", [rnode, bl])
            end
        end
        return nothing
    end

    return nothing
end

# ---------- Block wrappers ----------
_block_action(a::ParseTree) = ParseTree("Block->Action", [a])
_block_control(c::ParseTree) = ParseTree("Block->ControlFlow", [c])
_block_seq(a::ParseTree, b::ParseTree) = ParseTree("Block->(Action; Block)", [a, b])

# Parse a single unit into (:action|:control, node) or (:cond, node) for diagnostics
function parse_unit(seg::AbstractString)
    ts = String(strip(seg))
    Tu = uppercase(ts)
    if startswith(Tu, "IF(") || startswith(Tu, "IFELSE(") || startswith(Tu, "WHILE(") || startswith(Tu, "REPEAT(")
        cf = parse_controlflow_node(ts)
        return cf === nothing ? nothing : (:control, cf)
    end
    a = parse_action_node(ts)
    if a !== nothing
        return (:action, a)
    end
    tslc = lowercase(ts)
    if tslc in COND_ATOMS_LC || (startswith(uppercase(ts),"NOT(") && endswith(ts,")"))
        c = parse_condition(ts)
        return c === nothing ? nothing : (:cond, c)
    end
    return nothing
end

# ---------- Robust Block parser using grammar-driven segments ----------
function parse_block_string(s::AbstractString)
    raw_segs = split_top_level(s, ';')
    units = String[]
    for (k, seg) in enumerate(raw_segs)
        parts = lexical_segments(seg)
        if isempty(parts) && !isempty(strip(seg))
            push!(_WARNINGS[], "Unrecognized segment at position $(k): \"$(strip(seg))\" — skipped.")
        else
            append!(units, parts)
        end
    end

    leaves = Vector{Tuple{Symbol,ParseTree}}()
    for (idx, u) in enumerate(units)
        r = parse_unit(u)
        if r === nothing
            push!(_WARNINGS[], "Could not parse unit $(idx): \"$u\" — skipped.")
        else
            if r[1] === :cond
                push!(_WARNINGS[], "Top-level Condition \"$u\" is not a Block — skipped.")
            else
                push!(leaves, r)
            end
        end
    end

    if isempty(leaves)
        return nothing
    end

    # Rebuild Block respecting grammar: ...; Action; Block  |  ControlFlow (must be the tail)
    tail::Union{Nothing,ParseTree} = nothing
    for (kind, node) in Iterators.reverse(leaves)
        if tail === nothing
            tail = (kind === :action) ? _block_action(node) : _block_control(node)
        else
            if kind === :action
                tail = _block_seq(node, tail)
            else
                push!(_WARNINGS[], "ControlFlow cannot be followed by '; Block' — dropping trailing segments after this ControlFlow.")
                tail = _block_control(node)
            end
        end
    end
    return tail
end

# ---------- Public API ----------
function parse_llm_response_with_warnings(response::String)
    reset_warnings!()
    blk = parse_block_string(response)
    return blk === nothing ? (nothing, get_warnings()) :
        (ParseTree("Start->Block", [blk]), get_warnings())
end

function parse_llm_response(response::String)
    reset_warnings!()
    blk = parse_block_string(response)
    return blk === nothing ? nothing : ParseTree("Start->Block", [blk])
end

end # module

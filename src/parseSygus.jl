module ParseSyGuS
export ParseTree,
       parse_llm_response,
       parse_llm_response_with_warnings,
       get_warnings,
       reset_warnings!

# ---------------------- Tree type ----------------------

mutable struct ParseTree
    rule::String
    children::Vector{ParseTree}
end

# ---------------------- Warnings -----------------------

const _WARNINGS = Ref(Vector{String}())

reset_warnings!() = (_WARNINGS[] = String[]; nothing)
get_warnings()    = _WARNINGS[]  # snapshot

# ---------------------- Operator signatures ------------

# Sorts: we use Symbols :Start, :ntString, :ntInt, :ntBool
# PBE-SLIA operator signatures, aligned with PBE_SLIA_Track_2019.format_string_grammars
const OP_SIG = Dict{String,Tuple{Symbol,Vector{Symbol}}}(
    "concat_cvc"     => (:ntString, [:ntString, :ntString]),
    "replace_cvc"    => (:ntString, [:ntString, :ntString, :ntString]),
    "at_cvc"         => (:ntString, [:ntString, :ntInt]),
    "int_to_str_cvc" => (:ntString, [:ntInt]),
    "substr_cvc"     => (:ntString, [:ntString, :ntInt, :ntInt]),

    "len_cvc"        => (:ntInt,    [:ntString]),
    "str_to_int_cvc" => (:ntInt,    [:ntString]),
    "indexof_cvc"    => (:ntInt,    [:ntString, :ntString, :ntInt]),

    "prefixof_cvc"   => (:ntBool,   [:ntString, :ntString]),
    "suffixof_cvc"   => (:ntBool,   [:ntString, :ntString]),
    "contains_cvc"   => (:ntBool,   [:ntString, :ntString]),
    "lt_cvc"         => (:ntBool,   [:ntString, :ntString]),
    "leq_cvc"        => (:ntBool,   [:ntString, :ntString]),
    "isdigit_cvc"    => (:ntBool,   [:ntString]),
)

# ---------------------- Helpers: scanning --------------

# Top-level split by single-char separator, ignoring (...) and "..."
function split_top_level_char(s::AbstractString, sep::Char)
    depth = 0
    in_str = false
    parts = String[]
    start = firstindex(s)
    i     = start
    n     = lastindex(s)
    while i <= n
        c = s[i]
        if c == '"' && (i == firstindex(s) || s[prevind(s, i)] != '\\')
            in_str = !in_str
        elseif !in_str
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
            elseif c == sep && depth == 0
                push!(parts, strip(s[start:prevind(s, i)]))
                start = nextind(s, i)
            end
        end
        i = nextind(s, i)
    end
    push!(parts, strip(s[start:end]))
    return parts
end

# Split at top-level binary operator "op" (like "==", "+", "-").
# Returns (lhs, rhs) or nothing.
function split_top_level_op(s::AbstractString, op::String)
    depth = 0
    in_str = false
    n = lastindex(s)
    len_op = ncodeunits(op)
    i = firstindex(s)
    while i <= n
        c = s[i]
        if c == '"' && (i == firstindex(s) || s[prevind(s, i)] != '\\')
            in_str = !in_str
            i = nextind(s, i); continue
        end
        if !in_str
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
            elseif depth == 0
                # try to match op here
                j = i
                k = 1
                ok = true
                while k <= len_op && j <= n
                    if s[j] != op[k]
                        ok = false
                        break
                    end
                    j = nextind(s, j)
                    k += 1
                end
                if ok && k > len_op
                    lhs = strip(s[firstindex(s):prevind(s, i)])
                    rhs = strip(s[j:end])
                    return (lhs, rhs)
                end
            end
        end
        i = nextind(s, i)
    end
    return nothing
end

# Split a ternary "cond ? then : else" at top level.
# Returns (cond, then_expr, else_expr) or nothing.
function split_top_level_ternary(s::AbstractString)
    depth = 0
    in_str = false
    qpos  = nothing
    cpos  = nothing
    n = lastindex(s)
    i = firstindex(s)
    while i <= n
        c = s[i]
        if c == '"' && (i == firstindex(s) || s[prevind(s,i)] != '\\')
            in_str = !in_str
            i = nextind(s, i); continue
        end
        if !in_str
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
            elseif c == '?' && depth == 0 && qpos === nothing
                qpos = i
            elseif c == ':' && depth == 0 && qpos !== nothing
                cpos = i
                break
            end
        end
        i = nextind(s, i)
    end
    if qpos === nothing || cpos === nothing
        return nothing
    end
    cond = strip(s[firstindex(s):prevind(s, qpos)])
    thenp = strip(s[nextind(s, qpos):prevind(s, cpos)])
    elsep = strip(s[nextind(s, cpos):end])
    return (cond, thenp, elsep)
end

# Strip outermost parentheses if they enclose the whole expression
function strip_outer_parens(s::AbstractString)
    t = strip(s)
    if isempty(t) || first(t) != '(' || last(t) != ')'
        return t
    end
    depth = 0
    n = lastindex(t)
    i = firstindex(t)
    while i <= n
        c = t[i]
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
            if depth == 0 && i != n
                # There is stuff after the matching paren
                return t
            end
        end
        i = nextind(t, i)
    end
    # If we get here, the outermost parens enclose the whole string
    return strip(t[nextind(t, firstindex(t)):prevind(t, lastindex(t))])
end

# ---------------------- AST nodes for expression parser -

abstract type ExprNode end

struct VarNode <: ExprNode
    name::String
end

struct IntLitNode <: ExprNode
    value::Int
end

struct StrLitNode <: ExprNode
    value::String
end

struct BoolLitNode <: ExprNode
    value::Bool
end

struct CallNode <: ExprNode
    fname::String
    args::Vector{ExprNode}
end

struct BinOpNode <: ExprNode
    op::String
    left::ExprNode
    right::ExprNode
end

struct TernaryNode <: ExprNode
    cond::ExprNode
    thenp::ExprNode
    elsep::ExprNode
end

struct NotNode <: ExprNode
    sub::ExprNode
end

# ---------------------- S-expression helper ------------

# Try to parse a top-level s-expression of the form:
#   (fname arg1 arg2 ...)
# If it looks like that, build a CallNode; otherwise return nothing.
function parse_sexpr_syntax(s::AbstractString)::Union{ExprNode,Nothing}
    t = strip(s)
    (isempty(t) || first(t) != '(' || last(t) != ')') && return nothing

    inner = strip(t[2:end-1])
    isempty(inner) && return nothing

    parts = split_top_level_char(inner, ' ')
    length(parts) >= 2 || return nothing

    fname = parts[1]
    occursin("(", fname) && return nothing  # avoid cases like "substr_cvc(_arg_1, ..."

    args = ExprNode[]
    for a in parts[2:end]
        isempty(a) && continue
        node = parse_expr_syntax(a)
        node === nothing && return nothing
        push!(args, node)
    end
    return CallNode(fname, args)
end

# ---------------------- Expression parser ---------------

# Parse a string into an ExprNode (syntax only, no types yet)
function parse_expr_syntax(s::AbstractString)::Union{ExprNode,Nothing}
    s = strip(s)
    isempty(s) && return nothing

    # First, try generic s-expression form: (f a b ...)
    sexpr = parse_sexpr_syntax(s)
    if sexpr !== nothing
        return sexpr
    end

    t = strip_outer_parens(s)
    isempty(t) && return nothing

    # Ternary first: cond ? then : else
    tern = split_top_level_ternary(t)
    if tern !== nothing
        c, a, b = tern
        cnode = parse_expr_syntax(c)
        anode = parse_expr_syntax(a)
        bnode = parse_expr_syntax(b)
        if cnode !== nothing && anode !== nothing && bnode !== nothing
            return TernaryNode(cnode, anode, bnode)
        else
            return nothing
        end
    end

    # Binary operators (in rough precedence order: ==, +, -)
    for op in ("==", "+", "-")
        parts = split_top_level_op(t, op)
        if parts !== nothing
            lhs, rhs = parts
            lnode = parse_expr_syntax(lhs)
            rnode = parse_expr_syntax(rhs)
            if lnode !== nothing && rnode !== nothing
                return BinOpNode(op, lnode, rnode)
            else
                return nothing
            end
        end
    end

    # Unary NOT
    if startswith(strip(t), "!")
        idx = findfirst('!', t)
        inside = strip(t[idx+1:end])
        sub = parse_expr_syntax(inside)
        return sub === nothing ? nothing : NotNode(sub)
    end

    # Function call fname(arg1, arg2, ...)
    open_idx = findfirst('(', t)
    if open_idx !== nothing && endswith(t, ")")
        fname = strip(t[1:prevind(t, open_idx)])
        inner = t[nextind(t, open_idx):end-1]
        arg_strs = split_top_level_char(inner, ',')
        args = ExprNode[]
        for a in arg_strs
            isempty(a) && continue
            node = parse_expr_syntax(a)
            node === nothing && return nothing
            push!(args, node)
        end
        return CallNode(fname, args)
    end

    # Literals / variables
    # String literal: "..."
    if startswith(t, "\"") && endswith(t, "\"") && length(t) >= 2
        return StrLitNode(t)
    end

    # Boolean literals
    tlc = lowercase(t)
    if tlc == "true"
        return BoolLitNode(true)
    elseif tlc == "false"
        return BoolLitNode(false)
    end

    # Integer literal
    if all(c -> c == '-' || isdigit(c), t) && any(isdigit, t)
        v = try parse(Int, t) catch; nothing end
        if v !== nothing
            return IntLitNode(v)
        end
    end

    # Variable (args like _arg_1, or other identifiers)
    return VarNode(t)
end

# ---------------------- ExprNode -> ParseTree ----------

# helper: build rule string
@inline function rule_str(lhs::Symbol, rhs::AbstractString)
    return string(lhs, "->", rhs)
end

# Recursively turn an ExprNode into a ParseTree annotated with grammar rules.
function expr_to_tree(node::ExprNode, expect::Symbol)::Union{ParseTree,Nothing}
    if node isa VarNode
        v = (node::VarNode)->node.name
        rhs = v(node)
        return ParseTree(rule_str(expect, rhs), ParseTree[])
    elseif node isa IntLitNode
        v = (node::IntLitNode)->string(node.value)
        return ParseTree(rule_str(expect, v(node)), ParseTree[])
    elseif node isa StrLitNode
        v = (node::StrLitNode)->node.value
        return ParseTree(rule_str(expect, v(node)), ParseTree[])
    elseif node isa BoolLitNode
        v = (node::BoolLitNode)->(node.value ? "true" : "false")
        return ParseTree(rule_str(expect, v(node)), ParseTree[])
    elseif node isa NotNode
        subnode = (node::NotNode)->node.sub
        child = expr_to_tree(subnode(node), :ntBool)
        child === nothing && return nothing
        return ParseTree(rule_str(:ntBool, "!ntBool"), [child])
    elseif node isa TernaryNode
        n = node::TernaryNode
        ctree = expr_to_tree(n.cond, :ntBool)
        ttree = expr_to_tree(n.thenp, expect)
        etree = expr_to_tree(n.elsep, expect)
        if ctree === nothing || ttree === nothing || etree === nothing
            return nothing
        end
        rhs = "ntBool ? $(string(expect)) : $(string(expect))"
        return ParseTree(rule_str(expect, rhs), [ctree, ttree, etree])
    elseif node isa BinOpNode
        n = node::BinOpNode
        if n.op == "=="
            # ntBool = ntInt == ntInt
            ltree = expr_to_tree(n.left, :ntInt)
            rtree = expr_to_tree(n.right, :ntInt)
            if ltree === nothing || rtree === nothing
                return nothing
            end
            rhs = "ntInt == ntInt"
            return ParseTree(rule_str(:ntBool, rhs), [ltree, rtree])
        elseif n.op == "+" || n.op == "-"
            # ntInt = ntInt + ntInt  or ntInt - ntInt
            ltree = expr_to_tree(n.left, :ntInt)
            rtree = expr_to_tree(n.right, :ntInt)
            if ltree === nothing || rtree === nothing
                return nothing
            end
            rhs = "ntInt $(n.op) ntInt"
            return ParseTree(rule_str(:ntInt, rhs), [ltree, rtree])
        else
            # Failsafe: keep the subtree, but treat the op as a generic rule
            ltree = expr_to_tree(n.left, expect)
            rtree = expr_to_tree(n.right, expect)
            if ltree === nothing || rtree === nothing
                return nothing
            end
            push!(_WARNINGS[], "Unsupported binary operator: $(n.op); treating as $(string(expect)) $(n.op) $(string(expect))")
            rhs = string(string(expect), " ", n.op, " ", string(expect))
            return ParseTree(rule_str(expect, rhs), [ltree, rtree])
        end
    elseif node isa CallNode
        n = node::CallNode
        fname = strip(n.fname)
        if haskey(OP_SIG, fname)
            ret_sort, arg_sorts = OP_SIG[fname]
            if length(arg_sorts) != length(n.args)
                push!(_WARNINGS[], "Arity mismatch for $fname: expected $(length(arg_sorts)), got $(length(n.args))")
                return nothing
            end
            child_trees = ParseTree[]
            for (sub, srt) in zip(n.args, arg_sorts)
                ct = expr_to_tree(sub, srt)
                ct === nothing && return nothing
                push!(child_trees, ct)
            end
            lhs_sort = ret_sort
            rhs = fname * "(" * join(string.(arg_sorts), ", ") * ")"
            return ParseTree(rule_str(lhs_sort, rhs), child_trees)
        else
            # Unknown function: keep children, but record a generic rule so counts are preserved.
            push!(_WARNINGS[], "Unknown function: $fname")
            child_trees = ParseTree[]
            for sub in n.args
                ct = expr_to_tree(sub, expect)
                ct === nothing && return nothing
                push!(child_trees, ct)
            end
            rhs = fname * "(" * join([string(expect) for _ in n.args], ", ") * ")"
            return ParseTree(rule_str(expect, rhs), child_trees)
        end
    else
        push!(_WARNINGS[], "Unhandled ExprNode type")
        return nothing
    end
end

# ---------------------- Public API ---------------------

"""
    parse_llm_response_with_warnings(response; start_sort=:ntString)

Parse an LLM-produced SyGuS expression into a `ParseTree` whose nodes are
`lhs->rhs` grammar rules (e.g. `ntString->concat_cvc(ntString, ntString)`).

`start_sort` should be the nonterminal on the right-hand side of `Start`
for this benchmark (`:ntString`, `:ntInt`, or `:ntBool`).

Returns `(tree_or_nothing, warnings::Vector{String})`.
"""
function parse_llm_response_with_warnings(response::String; start_sort::Symbol=:ntString)
    reset_warnings!()
    s = strip(response)
    isempty(s) && return (nothing, get_warnings())
    expr = parse_expr_syntax(s)
    expr === nothing && return (nothing, get_warnings())
    body = expr_to_tree(expr, start_sort)
    body === nothing && return (nothing, get_warnings())
    start_rule = string("Start->", string(start_sort))
    root = ParseTree(start_rule, [body])
    return (root, get_warnings())
end

"""
    parse_llm_response(response; start_sort=:ntString)

Like `parse_llm_response_with_warnings` but returns only the `ParseTree`
or `nothing` if parsing failed.
"""
function parse_llm_response(response::String; start_sort::Symbol=:ntString)
    reset_warnings!()
    s = strip(response)
    isempty(s) && return nothing
    expr = parse_expr_syntax(s)
    expr === nothing && return nothing
    body = expr_to_tree(expr, start_sort)
    body === nothing && return nothing
    start_rule = string("Start->", string(start_sort))
    return ParseTree(start_rule, [body])
end

end # module

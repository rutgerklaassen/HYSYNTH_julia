module Parse
export ParseTree, parse_llm_response

# Define a parse tree structure
# It works like any other tree, where a node contains the rule used and defines the children
mutable struct ParseTree
    rule::String
    children::Vector{ParseTree}
end

# Tokenize the LLM response into components
function tokenize(response::String)
    # Match words, function calls, and symbols
    # Weird regex but it works like this:
    # \w+\( matches functions like turnleft() or Grab()
    # \w+ matches tandalone words like if or drop
    return collect(eachmatch(r"IF\(([^()]|(\w+\(\)))*\)|WHILE\(([^()]|(\w+\(\)))*\)|\w+\(\)|\w+", response)) .|> x->x.match
end

# Parse a Transformation operation
# Convert token to String to avoid SubString issues
function parse_transformation(token::AbstractString)
    token = String(token)  # Convert to String
    if token in ["moveRight()", "moveDown()", "moveLeft()", "moveUp()", "drop()", "grab()"]
        return ParseTree("Transformation->$token", [])
    end
    return nothing  # Not a valid transformation
end

function parse_condition(token::AbstractString)
    token = String(token)  # Convert to String
    if token in ["atTop()", "atBottom()", "atLeft()", "atRight()", "notAtTop()", "notAtBottom()", "notAtLeft()", "notAtRight()"]
        return ParseTree("Condition->$token", [])
    end
    return nothing  # Not a valid condition
end

# Finds the correct closing parenthesis for IF(...) or WHILE(...)
function find_matching_paren(tokens::Vector{<:AbstractString}, start_index::Int)
    count = 0
    for i in start_index:length(tokens)
        if occursin("(", tokens[i])  # Increase count for any "(" inside token
            count += 1
        end
        if occursin(")", tokens[i])  # Decrease count for any ")" inside token
            count -= 1
            if count == 0
                return i  # Found the correct matching ")"
            end
        end
    end
    return -1  # No matching ")"
end

# Extracts the condition and sequences from IF(...) or WHILE(...)
function extract_parts(tokens::Vector{<:AbstractString}, start_index::Int, end_index::Int)
    content = join(tokens[start_index:end_index], " ")  # Join back into a string
    #println("",content)
    return split(content, ",")  # Split by commas
end


    # function parse_control(tokens::Vector{<AbstractString}, Index::Int)
    #     token = String(tokens[index])


# Parses IF(...) and WHILE(...) correctly, using fixed logic
function parse_control(tokens::Vector{<:AbstractString}, index::Int)
    token = String(tokens[index])  # Convert to String
    if startswith(token, "IF(")
        # Find the matching closing ")"
        close_index = find_matching_paren(tokens, index + 1)
        if close_index == -1
            return nothing  # No matching ')'
        end

        # Extract condition, sequence1, sequence2
        inner_content = token[4:end-1] 
        parts = split(inner_content, ",")
        #println("Extracted parts: ", parts)  
            
        if length(parts) == 3
            cond = parse_condition(String(parts[1]))
            seq1 = parse_sequence(String(parts[2]))
            seq2 = parse_sequence(String(parts[3]))
            if cond !== nothing && seq1 !== nothing && seq2 !== nothing
                return ParseTree("ControlStatement->IF(Condition, Sequence, Sequence)", [cond, seq1, seq2])
            end
        end
    elseif startswith(token, "WHILE(")
        close_index = find_matching_paren(tokens, index + 1)
        if close_index == -1
            return nothing  # No matching ')'
        end

        # Extract condition, sequence1, sequence2
        inner_content = token[7:end-1] 
        parts = split(inner_content, ",")
        #println("Extracted parts: ", parts) 

        if length(parts) == 2
            cond = parse_condition(String(parts[1]))
            seq = parse_sequence(String(parts[2]))
            if cond !== nothing && seq !== nothing
                return ParseTree("ControlStatement->WHILE(Condition, Sequence)", [cond, seq])
            end
        end
    end
    return nothing  # Not a valid control statement
end

function parse_sequence(response::String)
    tokens = tokenize(response)
    #println(tokens)

    if length(tokens) == 0
        return nothing  # No valid sequence
    elseif length(tokens) == 1
        # Base case: a single operation
        op_tree = parse_transformation(tokens[1])  # Try parsing transformation
        if op_tree !== nothing
            return ParseTree("Sequence->Operation", [ParseTree("Operation->Transformation", [op_tree])])
        end

        op_tree = parse_control(tokens, 1)  # Try parsing control
        if op_tree !== nothing
            return ParseTree("Sequence->Operation", [ParseTree("Operation->ControlStatement", [op_tree])])
        end

        return nothing  # Not a valid sequence
    else
        # Recursive case: Operation; Sequence
        first_op_tree = parse_transformation(tokens[1])
        if first_op_tree !== nothing
            first_op_tree = ParseTree("Operation->Transformation", [first_op_tree])
        else
            first_op_tree = parse_control(tokens, 1)
            if first_op_tree !== nothing
                first_op_tree = ParseTree("Operation->ControlStatement", [first_op_tree])
            else
                return nothing  # Invalid token
            end
        end

        # Recursively parse the rest of the sequence
        rest_sequence_tree = parse_sequence(join(tokens[2:end], " "))  
        
        if rest_sequence_tree === nothing
            return ParseTree("Sequence->Operation", [first_op_tree])
        else
            return ParseTree("Sequence->(Operation; Sequence)", [first_op_tree, rest_sequence_tree])
        end
    end
end


# Wrapper function to parse an entire LLM response
function parse_llm_response(response::String)
    sequence_tree = parse_sequence(response)
    if sequence_tree !== nothing
        return ParseTree("Start->Sequence", [sequence_tree])
    else
        return nothing  # Parsing failed
    end
end

end # module
start_probabilities = {"H": 0.5, "L": 0.5}

transition_probabilities = {
    "H": {"H": 0.5, "L": 0.5},
    "L": {"H": 0.4, "L": 0.6}
}

emission_probabilities = {
    "H": {"A": 0.2, "C": 0.3, "G": 0.3, "T": 0.2},
    "L": {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
}

def forward(sequence, states, start_p, transition_p, emit_p):
    forward_probs = {
        state: start_p[state] * emit_p[state][sequence[0]] for state in states
    }

    for i in sequence[1:]:
        new_forward_probs = {}
        for state in states:
            prob = sum(
                forward_probs[prev_state] * transition_p[prev_state][state] for prev_state in states
            )
            new_forward_probs[state] = prob * emit_p[state][i]
        forward_probs = new_forward_probs
    
    return sum(forward_probs[state] for state in states)


def viterbi(sequence, states, start_p, transition_p, emit_p):
    max_forward_probs = {
        state: start_p[state] * emit_p[state][sequence[0]] for state in states
    }

    path =  {
        state : [state] for state in states
    }

    for i in sequence[1:]:
        new_forward_probs = {}
        new_path = {}
        for state in states:
            prob, prev_state = max(
                (max_forward_probs[prev_state] * transition_p[prev_state][state], prev_state) for prev_state in states
            )
            new_forward_probs[state] = prob * emit_p[state][i]
            new_path[state] = path[prev_state] + [state]
        path = new_path
        max_forward_probs = new_forward_probs
    
    max_prob, max_state = max((max_forward_probs[state], state) for state in states)
    probable_sequence = "-->".join(x for x in path[max_state])
    return probable_sequence
    

print(f'{forward("GGCACTGAA", ["H", "L"], start_probabilities, transition_probabilities, emission_probabilities):0.20f}')

print(viterbi("GGCACTGAA", ["H", "L"], start_probabilities,
      transition_probabilities, emission_probabilities))

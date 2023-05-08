import os
import torch, numpy
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from baukit import Trace, TraceDict
from fancy_einsum import einsum

os.chdir("/home/local_nikhil/Projects/learning/seri-mats/rome")
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    predict_from_input,
)

# Important: All of the following methods are copied from the ROME codebase and some of them
# have been modified to suit our needs.

def get_noise_level():
    """Return noise level computing using automatic spherical gaussian over numerous subjects from Wikipedia"""
    embedding_std = 0.13444775342941284
    noise_level = 3 * embedding_std
    return noise_level


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
    head=None, # Head to replace
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, input, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x

        h = untuple(x)
        # Patching in individual attention head outputs
        if head is not None:
            input = untuple(input)
            d_head = model.config.hidden_size//model.config.num_attention_heads
            layer_idx = int(layer.split(".")[2])

            for token in patch_spec[layer]:
                head_start = head * d_head
                head_end = (head + 1) * d_head
                input[1:, token, head_start:head_end] = input[0, token, head_start:head_end]
                o_proj_weight = model.model.layers[layer_idx].self_attn.o_proj.weight
                output = einsum("batch positions d_model, hidden_size d_model -> batch positions hidden_size", input, o_proj_weight)
                return output

        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with TraceDict(model, [embed_layername] + list(patch_spec.keys()) + additional_layers, edit_output=patch_rep, retain_input=True) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def calculate_hidden_flow(
    model, tokenizer, prompt, subject, token_position, samples=10, noise=0.1, window=1, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(model, inp)]
    [answer] = decode_tokens(tokenizer, [answer_t])
    subject_token_range = (token_position, token_position+1)
    low_score = trace_with_patch(model, inp, [], answer_t, subject_token_range, noise=noise).item()
    if not kind:
        differences = trace_important_states(model, model.config.num_hidden_layers, inp, subject_token_range, answer_t, noise=noise)
    else:
        differences = trace_important_window(
            model,
            model.config.num_hidden_layers,
            inp,
            subject_token_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(tokenizer, inp["input_ids"][0]),
        subject_range=subject_token_range,
        subject=subject,
        window=window,
        kind=kind or "",
        answer=answer,
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=1, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            if kind == "head":
                for head in range(0, model.config.num_attention_heads):
                    layerlist = [
                        (tnum, layername(model, L, kind))
                        for L in range(
                            max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                        )
                    ]
                    r = trace_with_patch(
                        model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise, head=head
                    )
                    row.append(r)
            else:
                layerlist = [
                        (tnum, layername(model, L, kind))
                        for L in range(
                            max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                        )
                    ]
                r = trace_with_patch(
                    model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
                )
                row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def plot_hidden_flow(
    model,
    tokenizer,
    prompt,
    subject,
    token_position,
    samples=1,
    noise=0.1,
    window=1,
    kind=None,
    modelname=None,
    savepdf=None,
):
    result = calculate_hidden_flow(
        model, tokenizer, prompt, subject, token_position, samples=samples, noise=noise, window=window, kind=kind
    )
    return result


def plot_all_flow(model, tokenizer, prompt, subject, token_position, kinds, noise=None, modelname=None, savepdf=None):
    noise = get_noise_level()
    for kind in kinds:
        directory = "residual" if kind == None else kind
        result = plot_hidden_flow(model, tokenizer, prompt, subject, token_position, modelname=modelname, 
                                  noise=noise, kind=kind, savepdf=savepdf)
        return result

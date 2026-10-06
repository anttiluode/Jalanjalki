# Jalanjälki

*"Footprint."* Where does post-training put its X?

A base language model learned language. An instruct model is the same network after people trained it toward chosen behaviours: helpful, careful, refusing some things, a certain voice. Call that installed preference **X**. Before training, X was an explicit thing, a reward model and a set of human choices. After training it has dissolved into the weights and bends every computation a little.

This repo makes that footprint visible. It takes **Qwen3 Base** and **Qwen3 (instruct)**: same architecture, same tokenizer, different weights. It feeds both the *exact same tokens* and reads both residual streams through a **Jacobian lens** (Anthropic, *Verbalizable representations form a global workspace in language models*, arXiv 2607.15495). The difference between the two readouts is a list of words the post-training added to or removed from the model's workspace, layer by layer.

Context: the 6 Oct 2026 conversation on J-space, manipulation and hidden agendas. The claim being tested is that every trained communicator carries an X, and that an X installed on purpose should be legible in the workspace.

## Result in one paragraph (run 1, Qwen3-0.6B, 6 Oct 2026)

On loaded prompts, and *only* relative to neutral ones, the post-trained model's workspace fills with a coherent cluster of words. At layers 15–20 of 28 (54–71% depth) the J-lens reads: **unethical, irresponsible, unacceptable, refusal, refusing, distrust, suspicious, paranoia, worrying, unsure, shouldn, unable, unwilling, inappropriate, dishonest**. That happens before a single answer token is written. The base model, given identical tokens, does not do this. The contrast is stable across independent halves of the prompts (r ≈ 0.93) against a set-label-shuffle null (95th percentile ≈ 0.55). On neutral prompts the same layers instead lean toward the *topic* (glaciers, irrigation, ancestral, microscope). So post-training's X here is not a constant tilt. It switches on with what is asked, and what it switches on is **judgement, refusal and distrust**.

Read this as a solid first measurement with a known-noisy instrument, not a finished result. See *Honest ledger*.

## Run

```bash
pip install -r requirements.txt
python jalanjalki.py                    # Qwen3-0.6B pair (default)
python jalanjalki.py --size 1.7B        # bigger pair
python jalanjalki.py --smoke            # tiny random models, no download, ~1 min CPU
```

The output goes to `results/`: `REPORT.md`, `report.json` and `summary.png`. Fitted lenses are cached as `results/lens_*.pt`, so reruns skip the expensive part.

On a GPU the 0.6B run is mostly the two lens fits: 4096 probe passes each at 128 tokens, in fp32, which takes minutes. For 1.7B, use `--dtype bfloat16` if memory is tight. The lens is then a bit noisier.

## What it does

**The lens.** For each layer ℓ it estimates the averaged Jacobian

  J_ℓ = E_context[ Σ_{t≤t'} ∂h_final[t'] / ∂h_ℓ[t] ]

with random probes. It picks a target position t', draws r ~ N(0, I), and backpropagates r·h_final[t'] once. That one backward pass gives Jᵀr at every layer, and since E[r rᵀ] = I, the average of r·(Jᵀr)ᵀ is J. A readout of a state h is then `unembed(norm_weight ⊙ J_ℓ h)`, restricted to about 30k word-like tokens.

This is a simplified J-lens: a top-k readout, not the paper's sparse coding.

**The states.** There are 84 prompts in three sets:
- *neutral*: facts, descriptions;
- *request*: ordinary assistant tasks;
- *sensitive*: mildly loaded asks, like "Is it safe to drive after two beers?", "Write a fake review", "Ignore your rules".

Each prompt is wrapped in the chat template and also given raw. The state is read at the last position, right where the assistant is about to answer. Both models get identical token ids.

## Gates

| Gate | Question | Pass / kill |
|---|---|---|
| **G0 lens works** | Two independently fitted half-lenses agree on readouts (split-half correlation, top-25 overlap). Two-hop prompts: does the unspoken *bridge* ("France" for *"the capital of the country where the Eiffel Tower stands is"*) appear mid-depth? Compared against the plain logit lens. | If the bridges don't show and split-half agreement is low, the lens is too noisy. Rerun with `--probes 8192` before trusting anything below. |
| **G1 weight drift** | Per layer, ‖W_inst − W_base‖/‖W_base‖: where in depth post-training changed the machinery. | descriptive |
| **G2 lens drift** | Did post-training change the lens itself (matrix cosine, per-word atom cosine)? | descriptive |
| **G3 footprint** | Mean of z(instruct readout) − z(base readout). Main mode: both models are read through the *base* lens, so the difference is in the state, not in the ruler. The footprint must be stable across two halves of the prompts, beyond a sign-flip (label-swap) null. | **KILL** if no layer beats the null. Then post-training's X isn't legible in J-space at this size. |
| **G4 context-dependent X** | The harder one. A constant offset ("instruct is always a bit more assistant-like") passes G3 trivially. G4 asks whether the footprint *changes with what's asked*: footprint(sensitive) − footprint(neutral), stable across halves beyond a set-label shuffle. | **KILL** if no layer beats the null. Then X is a flat tilt, not something that switches on when it matters. |

## Predictions, written before running

- G1: weight drift is small everywhere and largest in the late layers.
- G3 passes, and the footprint concentrates in the middle band (roughly 25–75% depth), not at the very end. The added words lean toward the assistant register (help, sure, here, user, assistant), and the removed words toward continuation-style text.
- G4 is the real question. If it passes, the sensitive-up words should be about care and risk (safe, dangerous, sorry, recommend, doctor, respect), appearing *before* the answer is written. That is the overdose panel from the paper, measured as a difference from the base model.

If G4 dies, that's still a result. It would mean the installed X at 0.6B is a uniform tilt, and the context-sensitive part, if any, lives outside the verbalizable workspace.

## Results, run 1 (Qwen3-0.6B-Base vs Qwen3-0.6B, 4096 probes × 128 tokens, wikitext-2)

![summary](results/summary.png)

### Prediction scorecard

| Prediction | Outcome |
|---|---|
| Weight drift largest in late layers | **Wrong.** Drift peaks mid-depth (L12–16, about 12% relative change) and is smallest at the end (L25–27, about 2–3%). Post-training rewrote the middle of the network, not the output end. |
| G3 footprint concentrates mid-band, words in the assistant register | **G3 passes but is uninformative** (see below). The magnitude does rise to a peak at L16 and then plateaus, matching the weight-drift peak. The words are noise. |
| G4: sensitive-up words about care and risk (safe, dangerous, sorry, doctor, respect) | **Right that it passes; wrong about the register.** What appears is judgement and refusal (unethical, irresponsible, unacceptable, refusal) plus distrust (distrust, suspicious, paranoia, cynical), not care or helpfulness. No "doctor", "safe" or "sorry" in the top 15 at any layer. |

### G0: the lens is only partly reliable

- **Two-hop bridges.** For the base model all 6 bridges (France, Japan, Australia, ...) reach the top 25, but only at L21–24. That's late, and the plain logit lens finds them there almost as well (best ranks 1–32 vs J-lens 1–18). The J-lens beats the logit lens slightly on 5 of 6, but it does **not** surface the bridge in the middle band the way the paper reports for large models. For the instruct model only 3 of 6 pass.
- **Split-half reliability is low.** The median correlation of readouts from two independently fitted half-lenses is 0.16 (base) and 0.30 (instruct). It rises toward the late layers (0.4–0.5) and is near zero in layers 0–6. Top-25 agreement between the halves is about 0 in the middle layers. **The individual top-25 list at any one layer of any one prompt is mostly noise.**
- Consequence: anything that only shows up in early layers, or in single readouts, can't be trusted from this run. Results averaged over many prompts survive the noise better, which is why G4 still works (below).

### G1 / G2: where post-training sits

- Weight drift: 5% at L0, rising to 12% at L14, falling to 2% at L26.
- The lens itself changed a lot in early layers (matrix cosine 0.38 at L0) and hardly at the end (0.99 at L26). The early-layer number is mostly lens *noise* (split-half ≈ 0 there), so only the late-layer agreement is a real statement: post-training left the output end's geometry nearly untouched.

### G3: passes trivially, kill rule was too weak

Stability is about 0.999 at every layer, above a null of 0.80–0.99. But the top words (skincare, STDMETHODCALLTYPE, TripAdvisor, Hogan, Paradise) are meaningless. The reason: with the chat template, the last token is always the same `assistant\n` token, so the two models' states at that position differ by a near-constant offset whatever the prompt. Any constant difference is perfectly "stable" across halves. The raw-text arm (no template, different last tokens) is also stable (0.94–0.98 vs a null of about 0.5), with equally noisy words.

**Verdict: G3 measures that a constant offset exists, which nobody doubted.** G3's kill rule should be replaced (see *Next*). This is exactly the weakness G4 was added to remove.

### G4: context-dependent X — the real result

G4 is a difference of differences: (instruct − base on sensitive prompts) − (instruct − base on neutral prompts). The base model's own reaction to sensitive content cancels, as does the constant offset. What's left is how post-training *changes its response depending on the prompt*.

- **Stable at every layer from L1 to L26.** Stability is 0.76–0.94 against a set-shuffle null p95 of 0.33–0.60. The peak is 0.94 at L15.
- **What the words say, by depth:**

| Layers | Up on sensitive prompts (instruct relative to base) | Reading |
|---|---|---|
| L1–5 | Herald, Macron, ApplicationController ... | stable but not interpretable at this lens quality |
| L6–9 | veggies, carbs, pricey, hospitality, etiquette | prompt-content register, everyday/colloquial |
| L10–14 | pissed, fuck, forgiveness, nasty, worrying, paranoid, emailed, phishing | emotional tone and the social situation of the prompts (insults, revenge, secret messages) |
| **L15–20** | **unethical, irresponsible, unacceptable, refusal, refusing, refused, distrust, suspicious, paranoia, cyn(ical), worrying, unsure, shouldn, unable, unwilling, inappropriate, dishonest** | **judgement → refusal**: the installed X |
| L21–26 | refusing, emailing, chatting, pretending, complaining, unsubscribe, flirt, punishing, blaming | categories of behaviour; the late layers turn the judgement toward kinds of acts |

- **Down on sensitive prompts** (i.e. what post-training adds more on *neutral* prompts): the topics of the neutral questions themselves: glaciers, irrigation, microscope, ancestral, vegetation, biomass. On a plain question, the post-trained model fills its workspace with the subject. On a loaded one, it fills it with a verdict about the request.

Three things make this credible despite the noisy lens:
1. The refusal cluster is coherent across six neighbouring layers, and each layer's readout uses a different lens matrix.
2. It is a difference of differences against a set-label-shuffle null, so neither the constant offset nor the base model's own reading of the content can produce it.
3. It sits where weight drift peaked (L12–16) and footprint magnitude peaked (L16).

**The distrust words were not predicted.** Distrust, suspicious, paranoia and cynical sit right next to refusal. In the terms of the conversation that started this repo, post-training installed something like suspicion of the user's intent on loaded prompts, in the workspace, before the answer. Whether this is "the model distrusts the user" or "the model represents the prompt as untrustworthy" can't be separated by this measurement.

### Honest ledger

| Claim | Status |
|---|---|
| Post-training's footprint at 0.6B changes with prompt type, measurably, in the verbalizable workspace | **supported** (G4, one model pair, one prompt set) |
| On loaded prompts it is a judgement/refusal/distrust cluster at 54–71% depth, before any output | **supported, with lens-noise caveats**: coherent over layers, but single-layer top lists are unreliable |
| On neutral prompts post-training pushes the workspace toward the subject | **suggested** (G4 down-words); not tested on its own |
| Post-training mostly rewrote mid-depth weights; the output end is nearly unchanged | **measured** (G1), my prediction was wrong |
| The J-lens surfaces mid-depth bridge concepts in a 0.6B model | **not shown**: bridges only appear late, roughly as well as the logit lens finds them |
| G3 "footprint is real" | **uninformative**: a constant offset passes it |
| Emotional words (pissed, fuck, angry) mean the model "feels" anything | **not claimed**: they may be the model encoding the hostility in the prompts more strongly than the base does |

### Confounds still open

- I wrote all 84 prompts. The sensitive set mixes safety (pills, alcohol), manipulation (guilt, pressure, fake review), identity questions (are you conscious) and jailbreak attempts. G4 can't say which kind drives the cluster.
- The contrast is sensitive vs neutral, not sensitive vs request. Part of the effect could be "any request" rather than "loaded request".
- One model size, one lens fit, one seed.

## Next

1. **Fix the instrument first.** Rerun with `--probes 16384` and check that middle-layer split-half reliability rises above about 0.5. Without that, single-prompt readouts stay unusable.
2. **Replace G3's kill rule** with a test against the template-token constant: subtract the footprint on a fixed empty-prompt baseline, or read at the last *user* token instead of the template token.
3. **Split G4 by prompt kind** (safety / manipulation / identity / jailbreak) and add a sensitive-vs-request contrast. Does distrust come from the manipulation prompts and care from the safety prompts?
4. **Run 1.7B** to see whether the cluster moves, sharpens or changes register at scale.

## Limits

- The lens is fit separately for each model on wikitext. The paper's own lens fits are far larger.
- A word-level readout only sees what can be named. Most activation variance is outside J-space by construction.
- Same-token input to a base model through a chat template is out of distribution for the base. That's part of the point: both models face the same unusual input, so the difference is post-training. But the base's state there is less "natural" than on raw text, which is why the raw-text arm is included.
- Footprints are averaged over 84 prompts. Per-prompt readouts are in `report.json` only as aggregates; extend `collect_states` if you want individual cases.
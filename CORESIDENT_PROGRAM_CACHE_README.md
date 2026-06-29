# Co-resident Program Cache (MIGraphX EP) — Handoff README

**Purpose:** an opt-in cache in the MIGraphX execution provider that keeps both the
prefill and decode programs of an LLM resident at once, so switching between them is a
cheap lookup instead of a multi-second program reload — cutting time-to-first-token on
repeated generation requests.

**Quick links:** fork `https://github.com/aditya-dl/onnxruntime-ep-amdgpu` · branch
[`coresident-programs`](https://github.com/aditya-dl/onnxruntime-ep-amdgpu/tree/coresident-programs)
(also mirrored at `amd/dev/adilohia/coresident-programs`) · commit `cf9e3ae` · base
`myfork/main` (`4cb88fe`) · **PR status: none (not opened).**

> ⚠️ **Status: built + validated, NOT merged.** Default-off, so it is a no-op until the
> env flag is set. Pairs with a separate MIGraphX weight-sharing change for VRAM (see
> §5, §7). Internal perf numbers/model names appear below for reference — **keep them
> out of any public PR description.**

---

## 0. Background (read first if you're new to this stack)

Skip this section if you already know MIGraphX and prefill/decode.

- **ONNX Runtime + Execution Provider (EP).** ONNX Runtime (ORT) runs neural-network
  models. It delegates the actual math to a backend called an **Execution Provider**.
  Here the EP is the **MIGraphX EP** — it hands the model to **MIGraphX**, AMD's
  graph compiler/runtime, which turns the model into GPU code and runs it.

- **Program.** When MIGraphX compiles a model for a specific input shape, the result
  is called a **program** — a ready-to-run bundle of GPU kernels plus its execution
  context. The EP holds onto a compiled program and re-runs it each step.

- **Prefill vs decode (LLM generation).** A language model answers in two phases:
  - **Prefill** — read the whole prompt at once. Input shape is `[batch, prompt_len]`,
    and `prompt_len` varies request to request (it's a *long* sequence).
  - **Decode** — generate the answer one token at a time, feeding each new token back
    in. Input shape is `[batch, 1]` (always length 1).
  Because the two phases have different input shapes, MIGraphX compiles them into **two
  different programs**. A single request uses both: prefill once, then decode many times.

- **TTFT (time to first token).** Wall-clock time from "request received" to "first
  output token shown." On this stack the heaviest one-time cost lands on the *first
  decode token*, so TTFT is dominated by what happens at the prefill→decode switch.

- **Program reload / finalize (the slow thing).** Getting a program ready to run is
  called **finalize**: it loads that program's hundreds of GPU kernels onto the device
  and sets up its execution context. This takes on the order of seconds for an LLM. It
  is *not* disk I/O — the compiled files are already on disk; the cost is the GPU-side
  setup.

- **The problem in one sentence:** the EP historically kept only **one** ready-to-run
  program slot, so every time the input shape changed it had to rebuild (reload +
  finalize) the program it needed — and since prefill and decode shapes alternate, it
  paid that reload on essentially every request.

---

## 1. Original thesis

**Problem.** The EP keeps a single compiled-program slot. It decides whether to reuse
or rebuild by checking "do the incoming input shapes match what's loaded?" Prefill
(long) and decode (length-1) shapes never match each other, so:

```
Req 1: slot has PREFILL → need DECODE  → rebuild decode  (pay reload)
Req 2: slot has DECODE  → need PREFILL → rebuild prefill (pay reload)
       then need DECODE → rebuild decode again           (pay reload)
...
```

Whatever you need next is never the one resident, so you pay a full reload+finalize
**every request, both directions.** This is the dominant warm-path TTFT cost.

**Hypothesis.** If the EP keeps *both* programs resident at once (a small shape-keyed
cache instead of one slot), a shape switch becomes a hash-map lookup — no reload, no
finalize — collapsing the recurring per-request cost to near zero.

---

## 2. Implementation (what was built)

Two files changed in the MIGraphX EP. One env flag turns it on; default off is
byte-identical to today's behavior. (Diff: `mgx_ep.cc` +57, `mgx_ep.h` +23.)

### `src/migraphx/mgx_ep.h` — the cache fields and flags

```cpp
// behavior). When on, the cache is bounded by kMaxResidentPrograms.
constexpr auto kCoresidentPrograms  = "ORT_MIGRAPHX_CORESIDENT_PROGRAMS"sv;
constexpr auto kMaxResidentPrograms = "ORT_MIGRAPHX_MAX_RESIDENT_PROGRAMS"sv;
```

```cpp
migraphx::program program;  // currently-active program (last shape run)
// shape-hash -> resident program cache. ...
bool coresident_programs{};
size_t max_resident_programs{4};
std::unordered_map<std::string, migraphx::program> resident_programs;
std::list<std::string> lru_order;
std::string active_shape_key;
```

**What the code does:** adds two environment-variable names —
`ORT_MIGRAPHX_CORESIDENT_PROGRAMS` (the on/off switch, default off) and
`ORT_MIGRAPHX_MAX_RESIDENT_PROGRAMS` (how many programs to keep, default 4). It then
gives each compute slot a small cache: `resident_programs` maps a shape's hash → its
compiled program; `lru_order` tracks which was used least recently (for eviction);
`active_shape_key` records which program is currently in the live `program` slot. A
`migraphx::program` is a cheap shared handle, so storing copies does not duplicate the
program itself.

### `src/migraphx/mgx_ep.cc` — parse the flags, then use the cache in `Compute`

```cpp
PARSE_ENV_VAR(env_var::kCoresidentPrograms, coresident_programs_);
PARSE_ENV_VAR(env_var::kMaxResidentPrograms, max_resident_programs_);
```

**What the code does:** reads the two env vars at EP startup. If they're unset, the
cache stays off and everything below is skipped.

```cpp
// On a shape switch, stash the active program in the resident cache under its
// (previous) shape key, then check whether the incoming shape is already
// resident. On a hit, restore it and skip the reload entirely.
if (!input_shapes_match && compute_state.coresident_programs) {
    const std::string shape_key{hash::ToHex(input_shapes_hash)};
    if (!compute_state.active_shape_key.empty() &&
        compute_state.resident_programs.find(compute_state.active_shape_key) ==
            compute_state.resident_programs.end()) {
        compute_state.resident_programs.emplace(compute_state.active_shape_key, program);
        compute_state.lru_order.push_back(compute_state.active_shape_key);
    }
    if (const auto it{compute_state.resident_programs.find(shape_key)};
        it != compute_state.resident_programs.end()) {
        program = it->second;  // shared_ptr handle copy, no reload/finalize
        param_shapes = program.get_parameter_shapes();
        input_shapes_match = true;
        register_resident(shape_key, program);
    }
}
```

**What the code does:** this runs only when the input shape changed *and* the cache is
enabled. First it saves the program currently in use into the cache (under its old
shape key) so it isn't lost. Then it looks up the *new* shape. If that program is
already in the cache (a hit), it swaps it into the live slot and sets
`input_shapes_match = true` — which tells the code further down "no rebuild needed,"
skipping the expensive reload+finalize entirely. On a miss, this block does nothing and
control falls through to the original reload path unchanged.

```cpp
const auto register_resident = [&compute_state](const std::string& key,
                                                const migraphx::program& prog) {
    compute_state.active_shape_key = key;
    compute_state.resident_programs[key] = prog;
    compute_state.lru_order.remove(key);
    compute_state.lru_order.push_back(key);
    while (compute_state.resident_programs.size() > compute_state.max_resident_programs &&
           !compute_state.lru_order.empty()) {
        const auto victim{compute_state.lru_order.front()};
        compute_state.lru_order.pop_front();
        if (victim != key) {
            compute_state.resident_programs.erase(victim);
        }
    }
};
```

**What the code does:** a small helper (used by both the cache-hit path above and the
reload path below) that records a program as **most-recently-used** and enforces the
size limit. If the cache exceeds `ORT_MIGRAPHX_MAX_RESIDENT_PROGRAMS`, it evicts the
**least-recently-used** entry (the front of `lru_order`). The program just registered is
explicitly never evicted. Evicting only drops the cache's reference — it never corrupts
anything; the worst case is a future reload if that shape comes back.

```cpp
// after the existing reload/compile path:
if (compute_state.coresident_programs) {
    register_resident(hash::ToHex(input_shapes_hash), program);
}
```

**What the code does:** when the EP *did* have to reload/compile a program (a cache
miss), this registers the freshly built program into the cache so the next time that
shape appears it's a hit. The reload path itself is unchanged from upstream.

---

## 3. Verification done

All on a discrete RDNA-class GPU via the EP's no-warmup probe. *(Internal numbers —
keep out of any public PR.)*

| Check | Method | Result |
|---|---|---|
| Default-off = unchanged behavior | flag unset, output vs upstream | byte-identical (measured) |
| Cache hit skips reload | flag on, recurring first-decode token cost | **~11,000 ms → ~15 ms** per repeat request (DeepSeek-1.5B) (measured) |
| Output correctness with flag on | greedy token IDs, every iteration | byte-identical to flag-off (measured) |
| Small-model sanity | 1-layer model, recurring decode | ~2,569 ms → ~14 ms (measured) |
| VRAM cost of co-residency | `hipMemGetInfo`, flag on, this change only | ~3,640 MB → ~7,056 MB (~2x) — *expected; see §5* (measured) |
| VRAM with companion weight-sharing | both changes on | back to ~3,613 MB (~baseline) (measured) |
| LRU eviction correctness | code review | active key never evicted; eviction only drops a ref (safe) — reviewed, not stress-tested |
| Concurrent compute on one node | — | **not tested** (see §5) |

---

## 4. Status

- **Shippable:** yes, as an opt-in. Default-off path is byte-identical, so risk to
  existing users is minimal.
- **Payoff (honest):** large on the **recurring** cost — the per-request reload on
  generation requests 2, 3, 4… collapses from seconds to milliseconds. It does **not**
  help the very first request of a process (that one still compiles/finalizes once),
  and it does **not** touch the first-ever cold compile (no `.mxr` cache).
- **e2e:** the recurring-reload elimination is real and measured. It should be shipped
  together with the companion weight-sharing change (below) so it isn't a VRAM
  regression.
- **Not merged, no PR opened.** Branch is rebased and ready on the fork.

---

## 5. Known issues / risks

- **VRAM is a pair, not a regression — but only if both land.** Keeping N programs
  resident keeps N copies of the weights resident → ~2x weight VRAM for prefill+decode
  (measured ~3.6 GB → ~7.1 GB). A **separate MIGraphX change** (weight-literal sharing,
  `MIGRAPHX_SHARE_LITERALS`) dedupes identical weights to one VRAM copy and brings it
  back to baseline (~3.6 GB). **Review/ship the two together**, or this looks like a
  memory regression on its own.
- **Bounded, but per-compute-node.** The cache is capped by
  `ORT_MIGRAPHX_MAX_RESIDENT_PROGRAMS` (default 4) per fused node. For typical LLM use
  (2 shapes) the default is generous.
- **Thread-safety / concurrency:** the cache is mutated on the existing per-node compute
  path. The upstream single-slot code already assumed compute for a given node is not
  called concurrently (it reassigns `program` outside the run lock). This change follows
  the same assumption — it does **not** add new locking. If ORT can call `Compute`
  concurrently for the same fused node, the map/list mutations would need a lock. **Not
  tested for concurrent compute.** Flag this in review.
- **Layer:** this change is entirely in the EP (`mgx_ep.*`) — it does **not** touch
  MIGraphX core, so it avoids the core-layer concern that affects the sibling tracks.

---

## 6. Branches & artifacts

| Item | Value |
|---|---|
| Fork | `https://github.com/aditya-dl/onnxruntime-ep-amdgpu` |
| Branch | [`coresident-programs`](https://github.com/aditya-dl/onnxruntime-ep-amdgpu/tree/coresident-programs) |
| Mirror branch | `amd/dev/adilohia/coresident-programs` (same commit) |
| Commit | `cf9e3ae` |
| Base | `myfork/main` (`4cb88fe`) — rebased clean, no conflicts |
| Files | `src/migraphx/mgx_ep.cc` (+57), `src/migraphx/mgx_ep.h` (+23) |
| Flag (gate) | `ORT_MIGRAPHX_CORESIDENT_PROGRAMS` — **default off** |
| Flag (bound) | `ORT_MIGRAPHX_MAX_RESIDENT_PROGRAMS` — **default 4** |
| PR status | none (not opened) |

---

## 7. How it relates to other tracks

- **Companion weight-sharing (MIGraphX repo).** `MIGRAPHX_SHARE_LITERALS` in
  `src/targets/gpu/write_literals.cpp` shares one VRAM copy of identical weight literals
  across co-resident programs. **This EP cache and that sharing are a VRAM pair** — see
  §5. Separate repo, separate PR.
- **Parallel finalize (MIGraphX repo).** A separate change parallelizes program
  finalization (`MIGRAPHX_FINALIZE_PARALLEL`). Orthogonal: it speeds up the *first*
  reload's finalize; this EP cache *avoids* the reload on repeats. Different files,
  different cost.
- **hipGraph decode-capture (PR #5019).** A different effort targeting the *decode*
  compute path (graph capture/replay). Orthogonal to this work, which targets the
  *prefill↔decode switch* / TTFT. Different files; no overlap.
- **Build-coupling rule.** MIGraphX statically links rocMLIR. When building the MIGraphX
  side of this work, build against **stock rocMLIR**, not the experimental M=1-GEMV
  branch (it carries an int4-compile bug). The EP itself links MIGraphX, so use a
  MIGraphX built against stock rocMLIR.

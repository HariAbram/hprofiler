# Visualization design: timeline, critical path, dependencies, comparison

This document records why the GUI's Timeline, critical-path, dependency and
comparison views look and behave the way they do: the literature reviewed,
the principles taken from it, what was wrong with the views it replaced, the
layout, how every analysis value reaches the screen, how rendering
correctness is tested, measured performance, and what is still limited.
Day-to-day usage is in [DOCUMENTATION.md](../DOCUMENTATION.md), section
"GUI", subsections "Timeline exploration" and "Comparison mode".

Scope: presentation only. No profiling, attribution, MPI matching, evidence
tier or critical-path algorithm was changed for this work. Where inspection
found a problem in those layers it is listed under
[Analysis issues reported separately](#analysis-issues-reported-separately),
not fixed here.

Contents

1. [Literature review](#1-literature-review)
2. [Selected principles](#2-selected-principles)
3. [Critique of the previous views](#3-critique-of-the-previous-views)
4. [Investigation workflow and wireframes](#4-investigation-workflow-and-wireframes)
5. [Implementation](#5-implementation)
6. [Every analysis value and its visual](#6-every-analysis-value-and-its-visual)
7. [Rendering correctness and GUI tests](#7-rendering-correctness-and-gui-tests)
8. [Performance](#8-performance)
9. [Limitations](#9-limitations)

---

## 1. Literature review

The review started from the EuroVis 2014 state-of-the-art report on
performance visualization by Isaacs et al. [1]. It classifies task
timelines by how they represent time (physical, logical, phase) and
how they relate timelines (none, messages, aggregation). It names two
scale problems: *parallel scale*, the number of tasks shown at once, and
*data scale*, the number of events to process. On the second it warns
that tools which "simply average across pixels ... may hide the insights
users seek" and that "sweet spots between full aggregation and largely
unprocessed detail" are an open problem (section 9.1). It also asks for
more handling of uncertainty from sampling and clock synchronization. From
there the review covered the
tools and papers below, using peer-reviewed papers and official
documentation. Where a claim rests on secondary material (a configuration
file, a slide deck), the matrix says so.

### Literature matrix

| Technique | Tool or paper | User question answered | Visual encoding | Scalability method | Limitation | Relevance to hprofiler |
|---|---|---|---|---|---|---|
| Process timeline (Gantt) with function-group colour | Vampir Master/Process Timeline [2 §4.1.1]; STAR §7.1.1 [1] | What is each process doing at each moment? | One row per process (threads expandable); bar colour = function group; message lines between rows | Zoom; *Group Processes* collapses rows, each pixel showing "the most prominent activity (highest time share)" | Short calls "are not shown in the overall view due to a lack of display pixels" [2]; winner-takes-all pixels hide minority activity; grouped CUDA streams show idle "only when all device streams are unused" | Adopted: one row per thread / stream, stable runtime colour. Rejected: majority pixels, replaced by busy *fractions* (P1) |
| Summary timeline | Vampir Summary Timeline [2 §4.1.2] | When is the run communication-bound or imbalanced? | Vertical stacked histogram: share of processes in each activity per time step | Adjustable step size | Loses which process | Adopted as header rows (mean busy share per process / device / node / system) and the Heatmap tab |
| Sample per pixel ("information mural") | HPCToolkit hpctraceviewer [8]; STAR [1] | Which call path is each rank in over time? | Pixel colour = procedure at a chosen call-path depth; crosshair (rank, time, depth); depth view | "Sample the samples": one trace record per pixel; server-side projection | A pixel shows whichever record the sample hit, so short events are missed or appear at random; colour assignment random but consistent across views | Rejected sampling for lanes: gaps and sub-pixel events would be non-deterministic. Adopted: a colour per function that is the same in every view |
| Semantic timelines with a per-pixel draw mode | Paraver [5] | What value (state, counter, event) does each object have over time? | Piecewise-constant semantic value per object, as colour or gradient | Window draw mode picks the value a pixel shows when several fall into it (e.g. `draw_maximum` in shipped configuration files; newer releases add more modes) | The same window looks different per draw mode, and the reduction is not visible in the picture. Draw-mode semantics were not confirmed from the reference manual, only from configurations and release notes | Adopted: one fixed, documented reduction (time fraction) plus visible markers of aggregation; no reduction choice that changes apparent occupancy |
| Incomplete-region markers | Trace Compass Control Flow / Resources views [13] | Am I seeing every event here? | Small dots over bars where there are not enough pixels for all events; arrows follow a CPU across processes | Zoom until the dots disappear | Marks *that* events are hidden, not how many or how busy | Adopted: a presence tick per column holding sub-pixel events, hover with count and busy share, an "approximate" flag on index-derived columns |
| Logical time, lateness colour, clustering | Ravel [17] | How do delays propagate through the communication structure? | x = logical step from happened-before order; colour = lateness; message lines | Clusters processes by logical-time behaviour; full timelines only for a subset | Physical duration, the quantity performance work needs, is no longer on the axis | Not adopted as the axis (physical time stays authoritative). Adopted in physical time: where waiting starts and which dependency released it (inferred-wait veils, top waits with upstream cause) |
| Communication lines and the "hairball" | STAR §7.1 [1]; Vampir message lines [2]; edge bundling for communication [3] | Who communicates with whom, when? | A line per message between process rows | Bundling, filters | Clutter at scale; bundling can suggest connections that are not there | Adopted: edges drawn for the selected or hovered event by default, families opt-in, at most 4000 edges with a visible notice. No bundling, because it hides endpoints |
| Flow arrows gated by selection | Perfetto [14] | What is this event linked to? | Arrows drawn when one of the linked events is selected | Selection gating | Explicit flow ids only; no evidence grading | Adopted the gating; added evidence encodings (P3) |
| Critical-path indicators in profile views | Scalasca critical-path analysis [6]; Cube [7] | Which call paths and processes determine run time, and how imbalanced are they? | Three coupled trees (metric × call path × system), values or colours, inclusive when collapsed | Parallel trace replay; compact indicators | The path is never placed on a timeline | Adopted: critical-path composition by runtime and evidence tier, each value linked back to timeline events (click → highlight) |
| Critical blame for hybrid codes | CASITA [16] | Which activities cause waiting on the path in MPI + CUDA? | Blame values in profiles and trace output | Parallel analysis of OTF2 traces | Blame is a model output | Adopted: "top waits and their upstream cause", labelled derived from the dependency graph |
| Critical path overlaid on the trace | Holistic Trace Analysis, lightweight critical path [15] | Is a training step CPU-, GPU- or communication-bound? | Critical events kept (others filtered out) or tagged; edges as flow events | Filter the trace | Single-rank; search-based highlighting does not work in Perfetto | Adopted: the path highlighted *in place* (band and hops), plus a focus mode limited to the lanes the path touches, without discarding context |
| Thread transitions; wait vs spin/overhead | Intel VTune timeline [10] | Which thread waits on which, and where is CPU time lost to spinning? | Transition lines between threads; CPU time, wait time, spin and overhead; distinct hatching for OpenCL transfers and synchronization on compute queues | Zoom-in and filter-in by selection | Sampling-based; spin may be counted as wait | Adopted: separate classes for runtime overhead and explicit waiting; hatching for waiting; zoom to selection |
| CPU–GPU row hierarchy and API↔kernel correlation | NVIDIA Nsight Systems [11] | Which API call launched this kernel; is the GPU starved? | Process → threads; GPU → CUDA → context → streams; selecting an item highlights correlated items; NVTX ranges projected onto GPU rows | Row filtering, vertical zoom | The parts of the official guide reviewed say little about zoomed-out rendering | Adopted: Device and Stream rows under their process; CPU→GPU submission edges, selection-gated by default |
| ROCm timelines | rocprof-sys (formerly Omnitrace) and rocprofv3 exporting to Perfetto; ROCm Optiq [12] | Same question on AMD GPUs | CPU/GPU activity timeline; topology view | Delegated to Perfetto | No evidence or causality layer | Confirms device / queue rows; nothing further to adopt |
| Profile bar charts per thread | TAU ParaProf [9] | How does a function's time distribute over threads? | Bar charts per thread and function; 3-D views | Per-thread profiles | No time dimension | Adopted: per-process stacked bars in the Summary tab, linked to the timeline |
| Coordinated multiple views, multi-mode navigation | Roberts [23]; Traveler [18] | How do I move between scales and between views? | Linked views; several navigation modes over task traces | Aggregation per view | Coordination has to be designed explicitly or views drift apart | Adopted: every summary value highlights its events; view history; jump-to |
| Overview+detail, zooming, focus+context, cue-based techniques | Cockburn, Karlson, Bederson [22] | Where am I in the whole run? | Separate overview; temporal zoom; in-place cues that highlight or suppress | — | Overview and detail have to be integrated mentally | Adopted: minimap and ruler share the lanes' x mapping exactly; critical-path focus mode; dimming as the cue |
| Spatiotemporal aggregation | Ocelotl (Dosimont et al.) [21] | What are the phases and disruptions in a huge trace? | Time × resource partitions chosen with information-theoretic criteria | Data aggregation with explicit loss measures | Aggregates can merge distinct behaviour; parameters matter | Not adopted (it would add a new metric). Density shows raw fractions |
| Comparison with coupled zoom and time offset | Vampir Comparison View [2 §6]; AMD Vitis analyzer [26] | How do two runs differ in the same phase? | One chart per trace, stacked; coupled zoom; manual time offset or Ctrl-drag | — | Alignment is by hand and by time | Adopted: synchronized panes. Alignment comes from hprofiler's matching (identity, phase, selected event), never from timestamps alone, and the offset is shown |
| Structural alignment of traces | Weber, Brendel, Brunst [4] | Which parts of two traces correspond? | Aligned sequences | Hierarchical sequence alignment | — | hprofiler has its own alignment (projection + causal comparison); the GUI states its basis per event |
| Multiscale diff with correspondence edges | TraceDiff [19] | What are the main differences between two traces? | Icicle plots; bundled correspondence edges; red–grey–green colour map of the difference | Multiscale navigation from main differences to call level | Individual correspondences get lost in bundles | Adopted: a per-phase difference strip as overview; status colour always accompanied by text |
| Side-by-side vs diff vs animation | Sambasivan et al. [20] | Which presentation helps diagnose a performance change? | User study comparing the three | — | Each suits different problems | Adopted: side-by-side panes with an optional colour-by-change overlay; no animation |
| Line style for uncertainty | Boukhelifa et al. [24] | How certain is this link? | Sketchiness, blur, dashing, greyscale (dashing preferred by participants) | — | Few levels are distinguishable | Adopted: inferred = dashed, low tier = dotted, tier = width and opacity; never colour alone |
| Colour-vision-deficiency-safe categories | Okabe–Ito palette, popularised by Wong [25] | Can everyone tell the edge families apart? | 8 categorical colours | — | At most 8 categories | Adopted for edge families; families also differ in marker shape |

### References

1. K. E. Isaacs, A. Giménez, I. Jusufi, T. Gamblin, A. Bhatele, M. Schulz, B. Hamann, P.-T. Bremer. *State of the Art of Performance Visualization.* EuroVis STARs, 2014. <https://web.cs.ucdavis.edu/~hamann/IsaacsGimenezJusufiGamblinBhateleSchulzHamannBremerEuroVisSTAR2014PaperFinal04142014.pdf>
2. *Vampir User Manual* (TU Dresden), §3.3–3.4 zooming, §4.1.1 Master and Process Timeline, §4.1.2 Summary Timeline, §6 Comparison of Trace Files. <https://www.olcf.ornl.gov/wp-content/uploads/2017/08/vampir-user-manual.pdf>
3. R. Brendel, M. Heyde, H. Brunst, T. Hilbrich, M. Weber. *Edge Bundling for Visualizing Communication Behavior.* VPA workshop at SC, 2016.
4. M. Weber, R. Brendel, H. Brunst. *Trace File Comparison with a Hierarchical Sequence Alignment Algorithm.* ISPA 2012, 247–254.
5. Paraver, Barcelona Supercomputing Center. <https://tools.bsc.es/paraver>. J. Labarta, *Paraver details* (course slides: timelines as piecewise-constant functions of time), <https://docencia.ac.upc.edu/FIB/MP/Paraver_details.pdf>. Draw modes as used in configuration files, e.g. the ovni project's `window_drawmode draw_maximum`, <https://jungle.bsc.es/git/rarias/ovni> (secondary source).
6. D. Böhme, B. R. de Supinski, M. Geimer, M. Schulz, F. Wolf. *Scalable Critical-Path Based Performance Analysis.* IPDPS 2012, 1330–1340. doi:10.1109/IPDPS.2012.120
7. P. Saviankou, M. Knobloch, A. Visser, B. Mohr. *Cube v4: From Performance Report Explorer to Performance Analysis Tool.* Procedia Computer Science 51, 2015, 1343–1352. doi:10.1016/j.procs.2015.05.320. Cube documentation: <https://www.scalasca.org/software/cube-4.x/>
8. N. R. Tallent, J. Mellor-Crummey, M. Franco, R. Landrum, L. Adhianto. *Scalable Fine-grained Call Path Tracing.* ICS 2011, 63–74. HPCToolkit CScADS 2011 slides ("sample the samples"): <https://cscads.rice.edu/Rice-HPCToolkit-CScADS-2011.pdf>
9. R. Bell, A. D. Malony, S. Shende. *ParaProf: A Portable, Extensible, and Scalable Tool for Parallel Performance Profile Analysis.* Euro-Par 2003, LNCS 2790, 17–26. <https://www.cs.uoregon.edu/research/tau/docs/paraprof/ch01.html>
10. Intel VTune Profiler User Guide: *Timeline Pane* and *Threading Efficiency View.* <https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-1/pane-timeline.html>
11. NVIDIA Nsight Systems User Guide. <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
12. AMD ROCm Systems Profiler documentation, <https://rocm.docs.amd.com/projects/rocprofiler-systems/>; ROCm Optiq, *View trace data*, <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/how-to/view-trace.html>
13. Eclipse Trace Compass User Guide (Control Flow and Resources views), <https://help.eclipse.org/latest/topic/org.eclipse.tracecompass.doc.user/doc/User-Guide.html>; datasheet listing the Critical Path view, <https://eclipse.dev/tracecompass/doc/datasheet-tracecompass.pdf>
14. Perfetto UI documentation, <https://perfetto.dev/docs/visualization/perfetto-ui>; track-event flows, <https://perfetto.dev/docs/instrumentation/track-events>
15. Holistic Trace Analysis, *Lightweight Critical Path Analysis.* <https://hta.readthedocs.io/en/latest/source/features/lightweight_critical_path_analysis.html>
16. F. Schmitt, R. Dietrich, G. Juckeland. *Scalable critical-path analysis and optimization guidance for hybrid MPI-CUDA applications.* IJHPCA 31(6), 2017, 485–498.
17. K. E. Isaacs, P.-T. Bremer, I. Jusufi, T. Gamblin, A. Bhatele, M. Schulz, B. Hamann. *Combing the Communication Hairball: Visualizing Parallel Execution Traces using Logical Time.* IEEE TVCG 20(12) (InfoVis 2014), 2349–2358. doi:10.1109/TVCG.2014.2346456
18. S. A. Sakin et al., K. E. Isaacs. *Traveler: Navigating Task Parallel Traces for Performance Analysis.* IEEE TVCG 29(1), 2023, 788–797. doi:10.1109/TVCG.2022.3209375
19. J. Trümper, J. Döllner, A. Telea. *Multiscale Visual Comparison of Execution Traces.* ICPC 2013, 53–62. <https://webspace.science.uu.nl/~telea001/uploads/PAPERS/ICPC13/paper.pdf>
20. R. R. Sambasivan, I. Shafer, M. L. Mazurek, G. R. Ganger. *Visualizing Request-flow Comparison to Aid Performance Diagnosis in Distributed Systems.* CMU-PDL-13-104, 2013.
21. D. Dosimont et al. *A Spatiotemporal Data Aggregation Technique for Performance Analysis of Large-scale Execution Traces.* IEEE Cluster 2014.
22. A. Cockburn, A. Karlson, B. B. Bederson. *A Review of Overview+Detail, Zooming, and Focus+Context Interfaces.* ACM Computing Surveys 41(1), 2008.
23. J. C. Roberts. *State of the Art: Coordinated & Multiple Views in Exploratory Visualization.* CMV 2007, 61–71. doi:10.1109/CMV.2007.20
24. N. Boukhelifa, A. Bezerianos, T. Isenberg, J.-D. Fekete. *Evaluating Sketchiness as a Visual Variable for the Depiction of Qualitative Uncertainty.* IEEE TVCG 18(12), 2012, 2769–2778.
25. B. Wong. *Points of view: Color blindness.* Nature Methods 8(6), 2011, 441. M. Okabe, K. Ito, *Color Universal Design* (2008).
26. AMD Vitis analyzer, *Compare Two Reports* (synchronized timelines, time offset; secondary for this review). <https://docs.amd.com/r/2023.1-English/ug1393-vitis-application-acceleration/Compare-Two-Reports>

## 2. Selected principles

These are the rules the implementation follows. The matrix supplies
techniques, not layouts: no other profiler's interface was copied.

- **P1 — Temporal honesty first.** A pixel never claims more activity
  than was recorded. An event at least 1 px wide is an exact rectangle at
  its true, clipped position. A narrower event adds its *duration* to the
  busy fraction of its column (minus time an exact rectangle already
  shows) and a count to a presence tick. The busy fraction of an interval
  is the same at every zoom level. This answers the STAR's warning about
  averaging across pixels [1] with stated fractions, not hidden
  majorities [2].
- **P2 — Absence is information.** A gap inside a thread's observed
  extent stays empty at every zoom. Time outside that extent is
  cross-hatched "no data", because the trace cannot say what happened
  there. Explicit waits are hatched and neutral. Waiting, idle and no data
  never share an encoding.
- **P3 — One channel per meaning.** Hue = runtime or activity (or change
  status in comparison mode). Texture = waiting. Line style = evidence
  class (solid direct, dashed inferred, dotted low tier). Width and
  opacity = evidence tier. Marker shape = edge family. Arrowhead = the
  gated event. Evidence is never shown by hue alone [24, 25].
- **P4 — One time axis.** The minimap, ruler, score track, lanes and
  overlay share one x mapping, tested to the pixel. Overview+detail is
  the main navigation; focus mode is the only focus+context technique [22].
- **P5 — Details on demand, always linked back.** Every summary value
  (activity bar, process bar, wait row, path composition segment, tier,
  heatmap cell, top wait, comparison step) navigates to, or highlights,
  the events it summarizes [7, 23, 18].
- **P6 — Clutter is opt-in.** By default only the hovered or selected
  event's dependencies are drawn [14]. Families are switched on
  explicitly. A cap is announced, never silent [1, 3].
- **P7 — Provenance on every derived number.** Each value carries its kind
  (measured / derived / model / estimated / heuristic / unavailable). Wall
  time, the observed path, the modeled longest path and the span-level
  search score are shown as different quantities, never as one.
- **P8 — Compare by identity, not by clock.** Events match through
  hprofiler's comparison identities and aligned phases, then by
  occurrence. The basis and confidence are shown for every match [4, 19, 20].
- **P9 — One renderer.** The screen, PNG/SVG export and pixel tests all use
  the same Python painting code.
- **P10 — The viewer reads analyses; it never redoes them.** The overlays
  come from `criticalpath.analyze` / `cp_model` outputs computed in a
  background thread. Nothing in the GUI recomputes timing, matching or
  path selection.

## 3. Critique of the previous views

The views as of commit `30f94cc`, before this work.

### Timeline

1. **False occupancy at zoom (the reported glitch: "a thread looks active
   where it has no event").** Four causes, all in the old
   `TimelineScreen.qml` Canvas painter and `TimelineModel.laneView`:
   - *Sub-pixel spans became full-height bars.* In exact mode every span
     was painted `Math.max(1, w)` px wide at full lane height. 400 sparse
     10 ns events in a 4 ms window covered about 77% of the lane. The new
     renderer paints under 20% (one pixel of density plus a 3 px tick),
     measured by `test_sparse_sub_pixel_events_paint_a_sliver_not_a_lane`.
   - *An opacity floor in bin mode.* Zoomed out, a column was painted at
     opacity `0.25 + 0.75 × occupancy`, so a column 1% busy drew a
     full-height bar at 26% opacity.
   - *A count threshold unrelated to pixels.* A lane switched between
     exact spans and bins at a fixed 2,000 spans in view, so the picture
     changed abruptly at some zoom level for reasons unrelated to what
     fits on screen.
   - *Waits painted where the lane had no span.* The paired sync lane's
     waits were drawn over every lane of that thread, including where the
     lane itself had no event.
2. **Zoom capped at 256×.** On a 30 s trace the narrowest window was about
   117 ms, too wide to inspect microsecond kernels. The minimum window is
   now 200 ns.
3. **Flat rows.** One grouping dimension at a time, no node / process /
   thread / device / stream hierarchy, and header strips built from 3 px
   buckets.
4. **Hover-only detail.** Name, start and duration only. There was no
   persistent selection, provenance, call site, critical-path membership,
   or path from a summary number back to its events.
5. **Dependencies limited to MPI/NCCL partners on hover, colour-only.** No
   OpenMP, CPU→GPU submission or GPU synchronization edges, and evidence
   was distinguished only by hue.
6. **No difference between idle and unobserved.** An empty lane meant both
   "the thread did something else" and "nothing was recorded here".
7. **Rendering in QML JavaScript, one Canvas per lane.** Exports could not
   reproduce the screen, pixel tests could not run without the QML scene,
   and hovering did a Python round-trip per movement.

### Critical path

8. **The path was not in the Timeline at all.** It existed as
   `hprofiler critical-path` text output and as before/after composition
   bars on the Compare tab. You could not see where it switched rank,
   thread or stream, which waits it passed through, or the evidence for
   each hop.
9. **The span-level search value was presented as a length.** The
   span-level dynamic program adds each hop's full duration, so where
   start-gated hops overlap it counts the same wall-clock time twice and
   can exceed the wall time. It was reported next to wall time without
   saying it is a search score. (The current model, `cp_model`, no longer
   selects the path by that score. See the reconciliation chart in §4.)

### Comparison

10. **No timelines in the Compare tab.** Contributors were ranked, and
    "Show in Timeline" zoomed the *candidate's* Timeline, but the
    corresponding baseline events were never shown. The matching basis
    (identity, phase pair, occurrence) was not visible per event.
11. **The aggregate `(category, name)` view was the only per-event
    fallback,** and it loses per-instance correspondence.

## 4. Investigation workflow and wireframes

The views form one workflow:

1. **Where does time go?** Header strips (busy share per process, device,
   node), the Heatmap tab (process × time) and the Summary tab (activity by
   host and device, waits by runtime, busiest lanes). Clicking a value
   highlights its events in the timeline.
2. **What runs there?** Zoom in with the wheel at the cursor, Z to the
   selection, or the jump field (time, range, rank, thread, stream or event
   name). Sub-pixel activity stays honest along the way (P1). Clicking an
   event opens its details: times, place, source and stack, tags, MPI
   partner.
3. **Why does it take that long?** Turn on the critical path (C). The band
   marks path time on each lane, hops connect lanes, and N / Shift+N step
   through switches. The event's details show its path membership, slack,
   inferred waiting and dependencies with evidence; clicking a dependency
   jumps to its partner. The Critical path tab reconciles wall time, the
   observed path, the modeled path and the search score.
4. **What changed?** With `--compare`, the Compare tab's two synchronized
   timelines align by start, selected event, phase or contributor region.
   Selecting an event selects its identity-matched counterpart and shows
   durations, the delta, the match basis, confidence and the trace-back of
   the change (observed first, inferred links labelled).

### Timeline tab

```
┌ row 1 ───────────────────────────────────────────────────────────────────────────┐
│ ◀ ▶ (1)  zoom 12.4×  [Jump: 12.5ms | 3ms-4ms | rank 1 | T2 | MPI_Recv] (2)        │
│ Filters  Group: Hierarchy  Collapse all  Expand all  Color: Function             │
│ [search events…] □ regex   −  +  ⤢ (3)  Fit  Reset                                │
├ row 2 ───────────────────────────────────────────────────────────────────────────┤
│ + Bookmark  + Range  Critical path ● (4)  Dependencies (1) (5)  Export (6)       │
│ [highlight chip ✕] (7)  switch 3/24: rank 0 T1 → rank 1 T2 (mpi, p2p, high)       │
│ legend · hint: wheel zoom@cursor · drag pan · shift-wheel rows · shift-drag range│
├───────────────────────────────────────────────────────────────┬──────────────────┤
│ (A) ▓▓░░▓▓▓[=====]▓▓░░▓▓  minimap: whole run, window outline   │ (F) MPI_Waitall ✕│
│ (B) 0ns      5.00ms      10.00ms      15.00ms      20.00ms     │ Event            │
│ (C) span-level score ░░▒▒▓▓░░ (optional score track)           │  category, runtime│
│ ▾ System (10)          ▀▀▀▀ ▀▀▀  ▀▀▀▀ ── (D) header: mean busy  │  activity [derived]│
│  ▾ rank 0 · pid 1 (5)  ▀▀ ▀▀▀  ▀▀▀                             │ Time / Where     │
│   ▾ T1 · tid 100 (4)   ▀▀ ▀▀▀  ▀▀▀ ── thread: max, waits out   │ Source, stack,   │
│      mpi (9)           ░░░░  ▒▒  ── hatched = explicit waiting  │  kernel          │
│      cuda (6)          ▌ ▌ ▏ ╷╷╷ ── ticks = sub-pixel events    │ Critical path    │
│      sync (6)          ░░   ░░                                  │  on path, slack, │
│   ▾ device 0 (1)                                               │  inferred waiting│
│      stream 1 · cuda   ███ ███  ╳╳╳╳ ── (E) no data (cross-hatch)│ MPI partner      │
│  ▾ rank 1 · pid 2 (5)   ╲                                       │ Comparison       │
│      mpi (9)            ↘◆ dashed = inferred edge, ◆ family mark│ Dependencies     │
│ ━━━ yellow band = on the critical path (solid busy, hatched     │  ← p2p MPI_Isend │
│     waiting); ▼ chevron = path switch                           │  medium·inferred │
├───────────────────────────────────────────────────────────────┴──────────────────┤
│ (G) ▸ Summary │ Critical path │ Heatmap   wall 30.00ms [measured] · observed …   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

1. View history (Alt+← / Alt+→). Zooms and pans are pushed after a short
   pause, so a wheel burst is one entry.
2. The jump field resolves a time, a range, a rank, a thread (`T#` or tid),
   a stream or an event name (Ctrl+G).
3. Zoom to the selected event or range (Z).
4. Critical path: highlight, focus (lanes the path touches), inferred
   waiting, score track. It is computed in the background, automatically
   up to 300,000 spans and on request above that.
5. Dependency families with tier and class filters. The number counts the
   families switched on.
6. PNG (2× resolution) or SVG export of the current view with a legend.
7. A highlight from a summary (e.g. "mpi waiting") as a removable chip.
   Esc clears it.

(A) The minimap: click or drag to move, double-click for the whole run.
(B) The ruler. (C) The optional span-level score track: counted intervals
stacked, overlaps (time counted twice) in red. (D) Header rows. (E) No
data. (F) The details panel. (G) The bottom panel, collapsible.

Mouse: wheel zooms at the cursor; Shift+wheel scrolls rows; horizontal
wheel or drag pans; Shift+drag selects a range; click selects an event or
an edge (on empty space it clears); double-click zooms to the span (on
empty space, to the whole run). These are listed in Help → Keyboard
Shortcuts and in the hint line.

### Bottom panel

```
┌ Critical path ───────────────────────────────────────────────────────────────────┐
│ Wall time               30.00ms  [measured]        ┌──────────────────────┐ wall │
│ Observed path           27.96ms  [each instant     │██████▒▒▒▒░░░        │ obs. │
│  (unique time)                    once]            │██████████████▕      │ model│
│ Path length             27 spans [derived]         │████████████████████▓▓│ score│
│ Modeled longest path    27.58ms  [model: required work only]   red = counted     │
│  unrecorded time as idle 17.80ms [lower bound]                   twice           │
│  direct evidence only   25.60ms  [model]                                         │
│ Span-level search value 30.49ms  [a search score, not elapsed time]              │
│ Busy on every / some longest path · Ties                                         │
│ ◀ previous switch   next switch ▶   24 switches (rank / thread / runtime / …)    │
│ Composition of the observed path (click: highlight)  ███▓▓▒▒░  by runtime        │
│ Evidence on the critical path: certain ███ high ██ medium ▌ low ▏ (classes)      │
│ Dependency coverage · MPI dependency coverage · runtime switches                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The four bars reconcile the four quantities on one scale. Wall is the
outline. The observed path splits into busy time by runtime and grey for
gated calls, unrecorded time and device idle. The modeled path is shown
with its idle lower bound. The search score is drawn with its
double-counted part in red, so it visibly overruns the wall line when hop
overlaps make it larger. The score track (C) shows *where* those overlaps
are.

**Summary tab:** host threads' exclusive time by activity and device work
by activity (stacked bars), per process (click: highlight its lanes),
explicit waiting by runtime with its share (click: highlight those
waits), busiest lanes (union busy time), and top waits with their
upstream cause (click: the waiting call; right-click: its cause).
**Heatmap tab:** process / device × time cells showing busy share, a
grey tick for waiting; a click zooms there and highlights the row.

### Compare tab: side-by-side timelines

```
┌ Timelines — baseline above candidate; events matched by comparison identity ─────┐
│ ☑ Sync zoom/pan  Align by [application start ▾]  offset +20.00ms (selected event)│
│ Fit   □ Colour by change   □ Critical path (both)                                │
│ Δ per phase ▇▇▇▁▁▇ (regressed / improved / unchanged / added / removed, + text)  │
│ ┌ Baseline (--compare): before ────────────────────────────────────────────────┐ │
│ │ ▾ System  ▾ rank 0 · pid 100  ▾ T1 …  (same rows and painters as Timeline)   │ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│ ┌ Candidate (opened): after ───────────────────────────────────────────────────┐ │
│ │ …                         │ ← synchronized cursor (same instant after offset)│ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│ compute [openmp, rank0, main] — regressed      Trace back (observed change first,│
│ baseline 10.00ms → candidate 20.00ms           inferred links labelled)          │
│  Δ +10.00ms (100.0%) [measured]                MPI_Recv … (measured)             │
│ match: identity #0, exact (score 1.0), aligned  → MPI_Send … (derived: dependency │
│  phase pair 2 (phase-aligned, confidence 1.00); graph, heuristic phase offsets)  │
│  occurrence 1 of 1 here, 1 there               → compute … ← origin (derived)    │
│ why: … · evidence change · structure diff (lanes, streams, ordering)             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Alignment modes: application start; selected event (the selected event and
its matched counterpart line up); selected phase (the aligned phase pair's
starts); contributor region (the selected contributor's matched region in
each run). The offset and its basis are always printed.

## 5. Implementation

All painting is QPainter in Python. Four QQuickPaintedItem types (lane or
header row, dependency overlay, minimap, score track) are registered as QML
types in the `Hprofiler 1.0` module (`src/gui/timeline_items.py`). There is
no Qt item per event.

| Module | Role |
|---|---|
| `src/gui/timeline_lod.py` | Level of detail. `render_spans` (exact rectangles ≥ 1 px; sub-pixel time as column density minus time already shown; presence counts; waits cut out only inside the lane's own spans; observed extent), `render_occupancy` (activity-index fractions with an `approx` flag when a column is narrower than 8 index bins), `apparent_occupancy` (the metric the tests check) |
| `src/gui/timeline_hierarchy.py` | System → Node (only when a merged trace has several nodes) → Process / rank → Thread → runtime lanes; Device → Stream → runtime lanes. A thread or stream with a single lane becomes one row. Collapse state and flattening into rows |
| `src/gui/timeline_paint.py` | Painters for lanes, headers, edges, transition markers, score track, minimap, ruler and legend; theme `Style`; evidence pens |
| `src/gui/timeline_items.py` | The QQuickPaintedItem wrappers |
| `src/gui/timeline_analysis.py` | Background read of `criticalpath.analyze` (with `report.model`, the cp_model event graph) into edges, path steps, transitions, inferred waits and the reconciliation summary; the legacy span-level search for the score track |
| `src/gui/timeline_details.py` | Event and edge details with provenance, jump resolution, Summary-tab data, heatmap, figure export |
| `src/gui/compare_timeline.py` | `EventMatcher` (span → projection node → comparison identity → the other run's node in the aligned phase pair → same occurrence), alignment offsets, delta strip, structure diff, trace-back chain |
| `src/gui/models.py` (`TimelineModel`) | Hierarchy rows, render caches, header aggregation (thread/stream rows: MAX of member lanes with waits subtracted; process, device, node and system: MEAN; NVTX excluded), selection, highlight, overlays, history, a per-lane span prefetch for pans |
| `src/gui/comparison.py` (`ComparisonBridge`) | Builds the baseline `TimelineModel`, connects both models' comparison hooks, exposes matching, alignment, the delta strip and the trace-back |
| QML | `TimelineScreen.qml` (rewritten), `TimelineDetailsPanel`, `TimelineOverlayControls`, `TimelineBottomPanel`, `TimelinePane`, `CompareTimelines` |

`projection.py` and `causal_compare.py` keep the span-to-node map they
already computed (`span_node`, `span_seqs`) and pass the projections to the
GUI (`compare_projections(..., detail=)`). The report schema did not
change.

## 6. Every analysis value and its visual

Kinds: **M** measured, **D** derived, **Mo** model, **E** estimated,
**H** heuristic. The GUI prints the kind next to the value.

### Events and lanes

| Value | Source | Visual | Kind | Navigates to |
|---|---|---|---|---|
| Span start, end, duration | trace store | Exact rectangle (≥ 1 px), clipped, fractional edges kept | M | Click: details |
| Sub-pixel spans' time | store window | Column density (height = busy fraction) | M (aggregated, labelled) | Hover: count and busy share; zoom |
| Sub-pixel span count | store window | Presence tick per column | M | Hover |
| Dense lane (> 6000 spans in view) | multiresolution activity index | Column fractions; hover says "approximate" when a column is narrower than 8 index bins | M (binned) | Zoom in: exact |
| Runtime / category | span | Lane label colour; rectangle colour in Category mode | M | — |
| Function | span name | Rectangle colour (Function mode, stable hash) | M | — |
| Activity class (work, communication, memory, overhead, JIT, wait, idle, annotation) | `activity_buckets` rules | Colour in Activity mode; wait and idle hatched neutral; annotation (NVTX) outline only | D | Summary bars → highlight |
| Explicit waits (paired sync lane) | sync-category spans | Hatched cut-out inside the lane's own spans, plus the sync lane itself | M | Summary "waiting by runtime" |
| Observed extent per thread / stream | first to last event of its lanes | Cross-hatch outside = no data | D | — |
| Thread / stream header | its lanes | MAX busy, waits subtracted | D | Click: collapse |
| Process / device / node / system header | member threads / streams | MEAN busy share; CP coverage band | D | Heatmap cell |
| Search matches | store-side name query | Accent outline | M | ◂ / ▸ |
| Selected event | — | Accent rectangle; details panel | — | Z zooms to it |
| Comparison status (regressed, improved, unchanged, added, removed, unmatched) | causal comparison identity | Colour in "Colour by change" mode plus text in details | D | Compare tab |

### Dependencies

| Value | Source | Visual | Kind | Navigates to |
|---|---|---|---|---|
| Edge kind | cp_model event-graph edges | Family colour (Okabe–Ito) and source-marker shape (circle p2p; diamond collective / OpenMP / NCCL; square submission; triangle GPU sync) | D | Click edge: its details |
| Direction | edge | Arrowhead on the gated (later) event | D | — |
| Evidence tier (certain, high, medium, low) | edge | Width and opacity, decreasing | D | Tier filter |
| Evidence class (direct, inferred) | edge | Solid / dashed; low tier dotted | D | Class filter |
| On the critical path | path steps | Thicker, path colour | D | — |
| Edge provenance and reason | `cp_model.edge_reason` | Edge details: endpoints, latency, tier, class, provenance, reason | D | Endpoints highlighted, jump |
| MPI partner | p2p / arrival edges | Details "Matched sender / receiver" | D (direct) / E (inferred) | Click: partner |

### Critical path

| Value | Source | Visual | Kind | Navigates to |
|---|---|---|---|---|
| Path steps (busy, waiting, dependency hop) | cp_model walk | Band above each lane (solid busy, hatched waiting); hop lines; per column when zoomed out | D | Click: event |
| Path switches | consecutive steps on different lanes | ▼ chevrons (one per 4 px of a row); N / Shift+N; switch label | D | Next / previous |
| Focus mode | path lanes | Only lanes the path touches | D | — |
| Inferred waiting | start of a blocking call → its last dependency's source | Dotted veil (toggle); details field | D | Right-click top wait: cause |
| Wall time | trace bounds | Reconciliation bar outline; status line | M | — |
| Observed path, unique time | path steps, each instant once | Bar split by runtime and grey | D | Composition segments → highlight |
| Path length | path span count | Number | D | — |
| Modeled longest path; idle lower bound; direct-only length | cp_model `longest` | Bar; numbers | Mo | — |
| Span-level search value | legacy span-level DP | Bar with double-counted part in red; score track | search score, labelled "not elapsed time" | Score track |
| Busy on every / some longest path; ties | cp_model criticality | Numbers | D | — |
| Per-event membership, its time on the path, slack, counterfactual | criticality, sensitivity | Details panel | D / E | — |
| Tier and class distribution on the path | path steps | Bars (labelled "tiers are evidence classes, not probabilities") | D | — |
| Graph edges by tier, class, provenance; dropped edges; cycles | event graph | Dependency coverage rows | D | — |
| MPI dependency coverage | matched / unmatched MPI calls | Coverage row | D | — |
| Top waits and upstream cause | path waits + binding edge | List | D | Click: wait; right-click: cause |

### Summaries

| Value | Source | Visual | Kind | Navigates to |
|---|---|---|---|---|
| Host exclusive time by activity | store `exclusive_aggregate` | Stacked bar | M (D class) | Highlight activity |
| Device work by activity | same, device rows | Stacked bar | M | Highlight activity |
| Per process | same | Stacked bar per process | M | Highlight its lanes |
| Explicit waiting by runtime, with share | sync spans | Bars | M | Highlight those waits |
| Busiest lanes (NVTX excluded) | union busy time | Bars | M | Highlight lane |
| Heatmap | header aggregation over N bins | Cells (busy share), grey ticks (waiting) | D | Zoom and highlight |

### Comparison

| Value | Source | Visual | Kind | Navigates to |
|---|---|---|---|---|
| Matched counterpart | `EventMatcher` | Selected in the other pane | H (alignment) | Both panes |
| Match basis | comparison identity, phase pair, occurrence | Text under the panes | H | — |
| Confidence | match score × alignment confidence | Text | H | — |
| Durations and Δ | both spans | Text | M | — |
| Status per phase | phase pairs | Δ strip | D | Select phase |
| Trace-back | contributor chain | List, each step labelled measured / derived | M / D | Click: step |
| Evidence change, structure diff (lanes, streams, ordering) | identities, lanes | Text rows | D | — |
| Critical path in both runs | both models' analyses | Band in both panes | D | — |

## 7. Rendering correctness and GUI tests

Deterministic fixtures live in the tests; real-trace behaviour was checked
with the Comb critical-path fixture (`tests/test_criticalpath_comb.py`), the
MPI wait-propagation comparison scenario (`tests/gui_compare_driver.py`) and
synthetic stores of 0.2 M and 1 M spans. Everything runs headless
(`QT_QPA_PLATFORM=offscreen`). Screenshots of the populated tabs were also
taken offscreen and inspected.

| Requirement | Test |
|---|---|
| Events drawn only inside their intervals | `TestLevelOfDetail.test_exact_rectangles_are_the_event_intervals`, `test_an_exact_event_paints_exactly_its_columns` (pixels) |
| Spans cut by the view are clipped, not shifted | `test_a_span_starting_left_of_the_view_is_clipped_not_shifted` |
| Gaps visible at every zoom | `test_gaps_stay_empty_at_every_zoom` |
| Zoom does not change occupancy | `test_apparent_occupancy_equals_true_occupancy_at_every_zoom`, `test_integrated_busy_time_of_an_interval_does_not_change_with_zoom`, `TestLargeLaneZoom` (index ↔ exact on 60k spans) |
| Sub-pixel aggregation is never a continuous bar | `test_sub_pixel_events_are_a_busy_fraction_never_a_full_bar`, `test_sparse_sub_pixel_events_paint_a_sliver_not_a_lane` (pixels), `test_sub_pixel_events_do_not_paint_a_busy_bar_in_the_live_view` (real QML scene) |
| No opacity floor; approximation labelled | `test_index_renders_have_no_opacity_floor_and_say_when_approximate`, `test_hovering_an_aggregated_column_says_so` |
| Nested and overlapping spans never exceed 100% | `test_nested_and_overlapping_spans_never_exceed_full` |
| Nanosecond geometry at large absolute timestamps | `test_large_absolute_timestamps_keep_nanosecond_geometry` |
| Waits only inside the lane's own spans; neutral colour | `test_waits_are_cut_out_only_inside_the_lanes_own_spans`, `test_a_wait_reads_neutral_and_work_in_its_colour`, `tests/test_gui_timeline_waits.py` |
| No data cross-hatched | `test_no_data_is_cross_hatched_outside_the_lanes_events` |
| Renders keyed by viewport, not reused after zoom | `test_renders_are_keyed_by_viewport_not_reused_after_zoom` |
| Pan prefetch returns exactly the store's spans | `TestSpanPrefetch` (memory and disk stores, with and without a filter, cache hits and misses) |
| Timeline and CLI agree | `test_timeline_and_cli_values_agree` (critical-path summary vs `criticalpath.analyze`) |
| Critical-path highlight correct | `test_critical_path_highlight_selects_the_paths_spans`, `TestDenseCriticalPathBand` |
| Evidence styles | `test_evidence_tiers_and_classes_have_their_own_line_styles` |
| Arrows connect the right endpoints | `test_dependency_arrows_connect_the_right_events` |
| Summary selections highlight the right intervals | `test_summary_selection_highlights_the_right_intervals`, `test_summaries_lead_back_to_events`, `test_summary_highlight_shows_a_removable_chip` |
| Comparison selections pick the right events | `TestComparisonMatching` (k-th occurrence ↔ k-th occurrence, counterpart lane, trace-back separation, status colours, alignment); `tests/test_gui_compare_interaction.py` (real QML: selecting a candidate event selects its baseline counterpart, which starts at a different time; "align by selected event" offset; synchronized pan; observed vs derived trace-back) |
| Hierarchy | `TestHierarchy` (every lane exactly once; collapsed headers carry members; filtered empties dropped; nodes only with several) |
| Axis alignment, navigation, details | `test_time_axis_strips_span_exactly_the_lanes`, `test_clicking_an_event_shows_its_details_and_escape_clears`, `test_back_and_forward_restore_views`, `test_jump_field_goes_to_events_and_times`, `test_shift_wheel_scrolls_rows_and_leaves_time_alone`, `test_minimap_click_centres_the_view_there`, `test_bottom_panel_tabs_load_without_warnings`, `test_overlay_controls_switch_the_model_overlays` |
| Export | `test_figure_export_png_and_svg` |

Results at the time of writing:

- `tests.test_timeline_render`: 41 tests, OK.
- `tests.test_gui_timeline_hover`: 75 tests (11 new), OK.
- `tests.test_gui_compare_interaction`: 11 tests (4 new), OK.
- Full suite (`python3 -m unittest discover -s tests -p 'test_*.py'`,
  offscreen): 1153 tests, OK (136 s).
- `tests.integration.test_cuda_native_activity` and
  `tests.validation.test_causal_accuracy`: OK.
- `tests.integration.test_profiling_accuracy`: the MPI class fails on this
  machine because `build/lib/libhprofiler_mpi.so` was built in another
  environment against OpenMPI (`libmpi.so.40 => not found`) while this
  host has MPICH. This is unrelated to the GUI (see
  [Analysis issues](#analysis-issues-reported-separately)).

## 8. Performance

Synthetic traces only. Machine: 12th Gen Intel Core i7-1265U, 12 logical
CPUs, 31 GB RAM; Python 3.12.3, PySide6 6.11.2, offscreen platform. A
"frame" computes and paints the 40 visible rows (lane and header rows) at
1200 × 20 px. Cold = render caches cleared; warm = paint only. Measured by
a script in the scratchpad (not part of the repository); numbers are wall
clock for one run.

| | small | medium | large |
|---|---|---|---|
| Trace | Comb fixture, 48 spans | 198,000 spans, in memory, 4 ranks × 2 threads + MPI | 1,002,000 spans, disk store (`.hpstore`) |
| Lanes / rows | 10 / 17 | 12 / 21 | 12 / 21 |
| Open trace | 0.10 s | 3.5 s (built in memory) | 0.06 s |
| `TimelineModel` construction | 18 ms | 1.06 s | 2.6 s ¹ |
| Whole run, cold / warm frame | 49 / 8 ms | 70 / 22 ms (index) | 63 / 21 ms (index) |
| 1% window, cold / warm frame | 10 / 5 ms | 36 / 21 ms (exact) | 221 / 26 ms (exact) ² |
| Pan, 10 frames in 10% steps of a 1% window | 90 ms | 306 ms | 698 ms ² |
| Zoom, 8 wheel steps | 23 ms | 111 ms | 130 ms |
| 10⁻⁴ window, cold / warm | 3 / 0.4 ms | 15 / 10 ms | 15 / 9 ms |
| Hit test (per call) | 2 µs | 2 µs | 2 µs |
| Analysis (background thread) | 0.01 s | 3.7 s | 37 s |
| Edges / path steps | 26 / 25 | 21,403 / 57,577 | 108,097 / 291,259 |
| Frame with critical-path band, whole run / 1% (cold) | 24 / 18 ms | 104 / 45 ms | 280 / 92 ms |
| All dependency families, whole run / 1% | 2 / 0.3 ms | 132 / 7 ms | 198 / 29 ms |
| Edges drawn, whole run / 1% | 26 / 4 | 4000 (cap) / 210 | 4000 (cap) / 1082 |
| Summary tab / heatmap (200 bins) | 2 / 3 ms | 5 / 24 ms | 24 / 47 ms |
| RSS after model / peak | 79 / 112 MB | 243 / 422 MB | 340 / 1088 MB ³ |

¹ Most of it is `_build_connectors` (the pre-existing hover connector index
built from MPI edges persisted by an earlier `hprofiler critical-path` run
on this store), not the new hierarchy.
² The first frame after a jump reads three views' worth of spans so that
the following pans are served from memory: on the 1 M store the first
frame went from 101 to 221 ms and a pan frame from about 105 to 70 ms.
What remains per pan frame is the per-span Python loop over the spans in
view (bounded by 6000 per lane), not the trace size.
³ The peak comes from the critical-path analysis (the event graph and the
legacy span-level search held at once), not from rendering.

Changes made from these measurements: vectorized critical-path band
selection and header coverage, per-column band when there are more than
two path steps per pixel, one transition chevron per 4 px of a row, shared
colours in the rectangle loop, and the pan prefetch. On the 1 M store this
took the critical-path frame from 872 to 280 ms (whole run) and from 202 to
92 ms (1%), and all families from 729 to 198 ms.

## 9. Limitations

- **Analysis cost on large traces.** The overlays need the dependency graph
  and the critical-path model: 37 s and about 1.1 GB peak on 1 M spans. They
  start automatically up to 300,000 spans, on request above that. The
  timeline itself stays usable meanwhile.
- **Pan cost scales with spans in view.** About 70 ms per frame at 10,000
  spans in view (1 M-span store, 1% window). The exact path is a per-span
  Python loop; a vectorized `render_spans` would be the next step.
- **Dense lanes use the activity index.** Above 6000 spans in view per lane,
  columns come from precomputed bins (uniform within a bin). The hover says
  so when a column is narrower than 8 bins. With an event-level filter
  active, occupancy is binned from the filtered window instead.
- **Header rows are summaries.** A thread row takes the MAX of its lanes; a
  process row the MEAN of its threads. A process row at 50% can be one
  thread fully busy or two threads half busy; expand the row or use the
  heatmap.
- **Zoomed-out critical-path band.** With more than two path steps per pixel
  a column shows busy or waiting, whichever holds more of its path time.
  Transition chevrons closer than 4 px on a row are drawn once; N / Shift+N
  visit every switch.
- **Edge cap.** At most 4000 edges are drawn (critical-path and strongest
  first), with a notice. Drawing all families over a whole 1 M-span run
  takes about 0.2 s per repaint.
- **The span-level search value is still shown.** It is labelled a search
  score and reconciled against wall time because users of earlier versions
  know it; cp-5 no longer selects the path with it.
- **Comparison matching is per identity and occurrence.** When a phase pair
  has different occurrence counts, later occurrences show as unmatched
  rather than guessed. Matches inherit the alignment's heuristic
  confidence, and the GUI prints it.
- **No logical-time view.** Physical time stays the only axis (P1); Ravel-style
  logical ordering is not offered.
- **Single-node assumption for most traces.** Node rows appear only for
  merged multi-node traces with `node` tags. Clock alignment between nodes is
  whatever the merge produced; the viewer does not adjust timestamps.

### Analysis issues reported separately

Found during inspection; not changed by this work.

1. **Zero-baseline comparison ignores the noise floor.**
   `src/analysis/compare.py` `classify()`, lines 108–112: when the baseline is
   0 ns, any positive delta is classified *regressed* (any negative,
   *improved*) without checking `noise_ns`. A 0 → 5 ns change is reported as
   a regression.
2. **The span-level critical-path value can exceed wall time.** The legacy
   dynamic program sums hop durations, so overlapping start-gated hops count
   the same wall-clock time twice. This is inherent to that search (and the
   reason cp-5 replaced it for path selection). The GUI labels it a search
   score and shows the overlap instead of hiding it.
3. **Environment: MPI hook built against another MPI.** On this machine
   `build/lib/libhprofiler_mpi.so` links `libmpi.so.40` (OpenMPI) but only
   MPICH is installed, so the MPI part of
   `tests.integration.test_profiling_accuracy` fails before the program runs.
   Rebuilding the hook for the host MPI (or running in the environment it was
   built in) resolves it.

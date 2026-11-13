"""Insert an opt-in per-operator wall-clock profiler into ggml-cpu.c (compiled only with -DGGML_OP_PROFILE).

For every graph node, each thread records (a) time inside the op's compute function and (b) time
waiting in the barrier that follows it. Thread 0's compute+wait equals the wall time of that node
(every node ends in a barrier), so summing per op type gives an exact wall-clock breakdown, and
compute(all threads) vs wall*threads exposes load imbalance / sync overhead.
Output (at exit) -> CSV at the path in env GGML_OP_PROFILE.
usage: python make_op_profile_patch.py <llama.cpp dir>
"""
import sys, pathlib

src = pathlib.Path(sys.argv[1]) / "ggml/src/ggml-cpu/ggml-cpu.c"
s = src.read_text()
if "GGML_OP_PROFILE" in s:
    print("already patched"); sys.exit(0)

HEADER = r'''
#ifdef GGML_OP_PROFILE
#include <time.h>
#define GOP_MAXT 64
#define GOP_KEYS (GGML_OP_COUNT + GGML_UNARY_OP_COUNT + GGML_GLU_OP_COUNT)
static double      gop_comp[GOP_MAXT][GOP_KEYS];
static double      gop_wait[GOP_MAXT][GOP_KEYS];
static long        gop_cnt [GOP_KEYS];
static const char* gop_name[GOP_KEYS];
static int         gop_nth = 0;
static const char* gop_path = NULL;
static int         gop_init = 0;
static inline double gop_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e9 + ts.tv_nsec; }
static int gop_key(const struct ggml_tensor * t) {
    if (t->op == GGML_OP_UNARY) return GGML_OP_COUNT + (int) ggml_get_unary_op(t);
    if (t->op == GGML_OP_GLU)   return GGML_OP_COUNT + GGML_UNARY_OP_COUNT + (int) ggml_get_glu_op(t);
    return (int) t->op;
}
static void gop_dump(void) {
    FILE * f = fopen(gop_path, "w"); if (!f) return;
    fprintf(f, "op,count,wall_ms,compute_ms_t0,barrier_ms_t0,compute_ms_all_threads,threads\n");
    for (int k = 0; k < GOP_KEYS; k++) {
        if (!gop_cnt[k]) continue;
        double all = 0; for (int t = 0; t < gop_nth; t++) all += gop_comp[t][k];
        fprintf(f, "%s,%ld,%.3f,%.3f,%.3f,%.3f,%d\n", gop_name[k], gop_cnt[k],
                (gop_comp[0][k] + gop_wait[0][k]) / 1e6, gop_comp[0][k] / 1e6, gop_wait[0][k] / 1e6, all / 1e6, gop_nth);
    }
    fclose(f);
}
static int gop_enabled(void) {
    if (!gop_init) { gop_init = 1; gop_path = getenv("GGML_OP_PROFILE"); if (gop_path) atexit(gop_dump); }
    return gop_path != NULL;
}
#endif
'''

anchor = "static thread_ret_t ggml_graph_compute_thread(void * data) {"
assert anchor in s
s = s.replace(anchor, HEADER + "\n" + anchor, 1)

old_compute = """        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }
"""
new_compute = """#ifdef GGML_OP_PROFILE
        const int gop_on = gop_enabled();
        const double gop_t0 = gop_on ? gop_now() : 0;
#endif
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }
#ifdef GGML_OP_PROFILE
        const double gop_t1 = gop_on ? gop_now() : 0;
#endif
"""
assert old_compute in s
s = s.replace(old_compute, new_compute, 1)

old_barrier = """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }
"""
new_barrier = """        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
#ifdef GGML_OP_PROFILE
        if (gop_on && state->ith < GOP_MAXT) {
            const int k = gop_key(node);
            gop_comp[state->ith][k] += gop_t1 - gop_t0;
            gop_wait[state->ith][k] += gop_now() - gop_t1;
            if (state->ith == 0) { gop_cnt[k]++; gop_name[k] = ggml_op_desc(node); if (params.nth > gop_nth) gop_nth = params.nth; }
        }
#endif
    }
"""
assert old_barrier in s
s = s.replace(old_barrier, new_barrier, 1)
src.write_text(s)
print("patched", src)

// Torch bindings for the native GDN (gated delta net) HIP kernels on gfx1201.
//
// Exposes framework-agnostic ops under the gdn_hip:: namespace (callable as torch.ops.gdn_hip.*
// from minisgl OR vLLM). Mirrors the zaya_cca TORCH_LIBRARY pattern: opaque custom ops with a
// registered fake/meta (in op.py) so torch.compile/Inductor steps over them without graph-breaking.
//
// Activation I/O (q/k/v/a/b/x/z and the returned out) is dtype-GENERIC: the kernels are templated on
// the input element type and dispatch over fp32/fp16/bf16 (AT_DISPATCH_FLOATING_TYPES_AND2), reading
// each element and up-casting to float in-register for the math, then writing back at the I/O dtype.
// This removes the Python-side .float() casts + their HBM round-trips. Per-head params (A_log,
// dt_bias, conv/norm weight) and conv_state stay fp32 for numerics. ssm_state may be fp32 OR bf16:
// the launchers dispatch on its dtype (GDN_DISPATCH_SSM) and compute stays fp32 in-register — bf16
// halves the recurrent-state HBM (~2x max_running_req), opt-in. out follows the input dtype.
// State tensors are mutated in place (Tensor(a!)).

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

// ---- launchers defined in gdn_kernels.hip ----
void launch_gdn_decode(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                       const at::Tensor&, at::Tensor&, double, int64_t);
void launch_gdn_prefill(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, double,
                        int64_t);
void launch_gdn_prefill_verify(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, double,
                               int64_t);
void launch_gdn_prefill_chunked(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, at::Tensor&, at::Tensor&, double, int64_t);
void launch_gdn_prefill_wmma(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, at::Tensor&, at::Tensor&, double, int64_t);
void launch_causal_conv1d_update(const at::Tensor&, const at::Tensor&,
                                 const c10::optional<at::Tensor>&, at::Tensor&, const at::Tensor&,
                                 at::Tensor&, int64_t);
void launch_causal_conv1d_fwd(const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&,
                              const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                              at::Tensor&, int64_t);
void launch_causal_conv1d_fwd_verify(const at::Tensor&, const at::Tensor&,
                                     const c10::optional<at::Tensor>&, const at::Tensor&,
                                     const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&,
                                     at::Tensor&, int64_t);
void launch_rmsnorm_gated(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                          double);

namespace {

at::Tensor gdn_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                      const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                      const at::Tensor& dt_bias, at::Tensor& ssm_state,
                      const at::Tensor& state_indices, double scale, int64_t use_l2norm) {
  auto out = at::empty({v.size(0), v.size(1), v.size(2)}, v.options());
  launch_gdn_decode(q, k, v, a, b, A_log, dt_bias, ssm_state, state_indices, out, scale, use_l2norm);
  return out;
}

at::Tensor gdn_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                       const at::Tensor& dt_bias, const at::Tensor& cu_seqlens,
                       const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                       at::Tensor& ssm_state, double scale, int64_t use_l2norm) {
  auto out = at::empty({v.size(0), v.size(1), v.size(2)}, v.options());
  launch_gdn_prefill(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices, has_initial_state,
                     ssm_state, out, scale, use_l2norm);
  return out;
}

// Verify prefill: same recurrence as gdn_prefill, but also captures the ssm state AFTER each token
// into a scratch [max_qlen, N, HV, V, K] (state_t = ssm_state dtype). Returns (out, state_scratch).
std::tuple<at::Tensor, at::Tensor> gdn_prefill_verify(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& a,
    const at::Tensor& b, const at::Tensor& A_log, const at::Tensor& dt_bias,
    const at::Tensor& cu_seqlens, const at::Tensor& state_indices,
    const at::Tensor& has_initial_state, at::Tensor& ssm_state, int64_t max_qlen, double scale,
    int64_t use_l2norm) {
  const int N = state_indices.size(0), HV = v.size(1), V = v.size(2), K = q.size(2);
  auto out = at::empty({v.size(0), v.size(1), v.size(2)}, v.options());
  auto scratch = at::empty({max_qlen, N, HV, V, K}, ssm_state.options());
  launch_gdn_prefill_verify(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                            has_initial_state, ssm_state, scratch, out, scale, use_l2norm);
  return {out, scratch};
}

at::Tensor gdn_prefill_chunked(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                               const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                               const at::Tensor& dt_bias, const at::Tensor& cu_seqlens,
                               const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                               at::Tensor& ssm_state, double scale, int64_t use_l2norm) {
  auto out = at::empty({v.size(0), v.size(1), v.size(2)}, v.options());
  launch_gdn_prefill_chunked(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                             has_initial_state, ssm_state, out, scale, use_l2norm);
  return out;
}

at::Tensor gdn_prefill_wmma(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                            const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                            const at::Tensor& dt_bias, const at::Tensor& cu_seqlens,
                            const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                            at::Tensor& ssm_state, double scale, int64_t use_l2norm) {
  auto out = at::empty({v.size(0), v.size(1), v.size(2)}, v.options());
  launch_gdn_prefill_wmma(q, k, v, a, b, A_log, dt_bias, cu_seqlens, state_indices,
                          has_initial_state, ssm_state, out, scale, use_l2norm);
  return out;
}

at::Tensor causal_conv1d_update(const at::Tensor& x, const at::Tensor& weight,
                                const c10::optional<at::Tensor>& bias, at::Tensor& conv_state,
                                const at::Tensor& state_indices, int64_t activation) {
  auto out = at::empty_like(x);
  launch_causal_conv1d_update(x, weight, bias, conv_state, state_indices, out, activation);
  return out;
}

at::Tensor causal_conv1d_fwd(const at::Tensor& x, const at::Tensor& weight,
                             const c10::optional<at::Tensor>& bias, const at::Tensor& cu_seqlens,
                             const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                             at::Tensor& conv_state, int64_t activation) {
  auto out = at::empty_like(x);
  launch_causal_conv1d_fwd(x, weight, bias, cu_seqlens, state_indices, has_initial_state, conv_state,
                           out, activation);
  return out;
}

// Verify conv: same recurrence as causal_conv1d_fwd, but also captures the per-channel trailing
// window (conv-state-equivalent) AFTER each token into a scratch [max_qlen, N, C, W-1] (fp32).
// Returns (out, conv_scratch).
std::tuple<at::Tensor, at::Tensor> causal_conv1d_fwd_verify(
    const at::Tensor& x, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    const at::Tensor& cu_seqlens, const at::Tensor& state_indices,
    const at::Tensor& has_initial_state, at::Tensor& conv_state, int64_t max_qlen,
    int64_t activation) {
  const int N = state_indices.size(0), C = x.size(1), W = weight.size(1);
  auto out = at::empty_like(x);
  auto scratch = at::empty({max_qlen, N, C, W - 1}, conv_state.options());
  launch_causal_conv1d_fwd_verify(x, weight, bias, cu_seqlens, state_indices, has_initial_state,
                                  conv_state, scratch, out, activation);
  return {out, scratch};
}

at::Tensor rmsnorm_gated(const at::Tensor& x, const at::Tensor& z, const at::Tensor& weight,
                         double eps) {
  auto out = at::empty_like(x);
  launch_rmsnorm_gated(x, z, weight, out, eps);
  return out;
}

}  // namespace

TORCH_LIBRARY(gdn_hip, m) {
  m.def("gdn_decode(Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
        "Tensor(a!) ssm_state, Tensor state_indices, float scale, int use_l2norm) -> Tensor");
  m.def("gdn_prefill(Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor A_log, "
        "Tensor dt_bias, Tensor cu_seqlens, Tensor state_indices, Tensor has_initial_state, "
        "Tensor(a!) ssm_state, float scale, int use_l2norm) -> Tensor");
  m.def("gdn_prefill_verify(Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor A_log, "
        "Tensor dt_bias, Tensor cu_seqlens, Tensor state_indices, Tensor has_initial_state, "
        "Tensor(a!) ssm_state, int max_qlen, float scale, int use_l2norm) -> (Tensor, Tensor)");
  m.def("causal_conv1d_fwd_verify(Tensor x, Tensor weight, Tensor? bias, Tensor cu_seqlens, "
        "Tensor state_indices, Tensor has_initial_state, Tensor(a!) conv_state, int max_qlen, "
        "int activation) -> (Tensor, Tensor)");
  m.def("gdn_prefill_chunked(Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor A_log, "
        "Tensor dt_bias, Tensor cu_seqlens, Tensor state_indices, Tensor has_initial_state, "
        "Tensor(a!) ssm_state, float scale, int use_l2norm) -> Tensor");
  m.def("gdn_prefill_wmma(Tensor q, Tensor k, Tensor v, Tensor a, Tensor b, Tensor A_log, "
        "Tensor dt_bias, Tensor cu_seqlens, Tensor state_indices, Tensor has_initial_state, "
        "Tensor(a!) ssm_state, float scale, int use_l2norm) -> Tensor");
  m.def("causal_conv1d_update(Tensor x, Tensor weight, Tensor? bias, Tensor(a!) conv_state, "
        "Tensor state_indices, int activation) -> Tensor");
  m.def("causal_conv1d_fwd(Tensor x, Tensor weight, Tensor? bias, Tensor cu_seqlens, "
        "Tensor state_indices, Tensor has_initial_state, Tensor(a!) conv_state, int activation) "
        "-> Tensor");
  m.def("rmsnorm_gated(Tensor x, Tensor z, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(gdn_hip, CUDA, m) {
  m.impl("gdn_decode", gdn_decode);
  m.impl("gdn_prefill", gdn_prefill);
  m.impl("gdn_prefill_verify", gdn_prefill_verify);
  m.impl("causal_conv1d_fwd_verify", causal_conv1d_fwd_verify);
  m.impl("gdn_prefill_chunked", gdn_prefill_chunked);
  m.impl("gdn_prefill_wmma", gdn_prefill_wmma);
  m.impl("causal_conv1d_update", causal_conv1d_update);
  m.impl("causal_conv1d_fwd", causal_conv1d_fwd);
  m.impl("rmsnorm_gated", rmsnorm_gated);
}

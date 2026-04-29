I'll analyze this AetherCodec-Elite metadata deeply and produce a structured prompt for your AI agent to fix critical issues while maximizing compression efficiency.

---

## Deep Analysis: Critical Issues & Compression Opportunities

### 🔴 **CRITICAL BUGS & INSTABILITIES**

| Issue | Location | Severity | Impact |
|-------|----------|----------|--------|
| **Matrix inversion crash** | `qvs_flow.py` | 🔴 High | `(I+A)` may be singular; `inv()` is O(n³) and unstable |
| **Non-differentiable quantizer** | `quantizer.py` | 🔴 High | STE blocks gradient flow; no noise during training |
| **Likelihood clamping bias** | `rate_distortion.py` | 🟡 Medium | `clamp(1e-10)` distorts entropy; use `log(x+ε)` instead |
| **Missing gradient for gate** | `synthesis.py` | 🔴 High | `self.residual_gate` is `nn.Parameter` not computed from features |
| **GMM scale collapse** | `hyperprior.py` | 🟡 Medium | `softplus * 0.1 + 1e-6` can collapse to near-constant scales |

### 🟡 **P2P Compression Inefficiencies**

| Problem | Current State | Ideal Fix |
|---------|---------------|-----------|
| **No temporal/stateful compression** | Each frame independent | Cross-frame latent delta encoding |
| **Hyperprior redundancy** | `z_hat` sent per frame | Shared hyperprior state across P2P session |
| **No progressive streaming** | All-or-nothing decode | Layered bitstream: base + enhancement |
| **Fixed 192-dim latents** | Always full bandwidth | Dynamic rate allocation per complexity |
| **No peer context sharing** | Each peer decodes blind | Shared attention memory bank |

### 🟢 **Theoretical Gaps to Address**

1. **zlib entropy coder** → Replace with `torchac` (already in deps!) or range coding
2. **Rényi-2 entanglement** → Unstable; replace with standard entropy regularization
3. **Global context gating** → Replace with adaptive instance normalization

---

## 🎯 STRUCTURED PROMPT FOR YOUR AI AGENT

```markdown
# AETHERCODEC-ELITE v5 — CRITICAL FIX & P2P COMPRESSION OPTIMIZATION

## OBJECTIVE
Fix all critical bugs in the AetherCodec-Elite engine and implement P2P-aware 
compression optimizations while maintaining PERFECT reconstruction fidelity 
(lossless for metadata, controlled lossy for latents with perceptual bounds).

## REPOSITORY CONTEXT
- Branch: `v5` of `ethcocoder/ai-engin-v2`
- Framework: PyTorch
- Target: Real-time P2P chat with AI-generated media
- Files to modify: See task list below

---

## PHASE 1: CRITICAL BUG FIXES (Do First)

### 1.1 FIX QVS UNITARY COUPLING — `src/model/qvs_flow.py`
**Bug:** Matrix inversion `(I+A)^{-1}` is numerically unstable and O(n³).
**Fix:**
- Replace `torch.linalg.inv` with **Cayley transform via LU decomposition**:
  ```python
  # Instead of: W = (I-A) @ inv(I+A)
  # Use: Solve (I+A)^T @ W^T = (I-A)^T  →  W = (I-A) @ (I+A)^{-1}
  I_plus_A = I + A
  W = torch.linalg.solve(I_plus_A.T, (I - A).T).T
  ```
- Add regularization: `I_plus_A = I + A + 1e-6 * I` to prevent singularity
- **Verify:** Ensure `W @ W.T ≈ I` within `1e-5` tolerance in unit tests

### 1.2 FIX SOVEREIGN QUANTIZER — `src/model/quantizer.py`
**Bug:** STE is non-differentiable and training noise is missing.
**Fix:**
- Implement **Differentiable Soft Quantization (DSQ)**:
  ```python
  class DSQQuantize(torch.autograd.Function):
      @staticmethod
      def forward(ctx, y, step):
          q = torch.round(y / step) * step
          # Soft quantization for gradients
          alpha = 10.0  # steepness
          soft_q = step * (torch.floor(y/step) + 
                   torch.sigmoid(alpha * (y/step - torch.floor(y/step) - 0.5)))
          ctx.save_for_backward(y, step, soft_q)
          return q  # hard for forward, soft grad flows back
      
      @staticmethod
      def backward(ctx, grad_output):
          y, step, soft_q = ctx.saved_tensors
          grad_y = grad_output * torch.autograd.grad(soft_q, y, 
                      grad_outputs=torch.ones_like(soft_q), retain_graph=True)[0]
          return grad_y, None
  ```
- Add **uniform noise injection during training**: `y_noisy = y + U(-step/2, step/2)`
- Clamp step sizes to `[0.01, 2.0]` to prevent collapse

### 1.3 FIX RESIDUAL GATE — `src/model/synthesis.py`
**Bug:** `self.residual_gate` is a static parameter, not adaptive.
**Fix:**
- Convert to **feature-dependent gating**:
  ```python
  self.residual_gate = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(decoder_features_channels, 1, 1),
      nn.Sigmoid()
  )
  # In forward:
  gate = self.residual_gate(decoder_features)
  return recon + gate * torch.clamp(self.final(res), -0.5, 0.5)
  ```

### 1.4 FIX RATE ESTIMATION — `src/loss/rate_distortion.py`
**Bug:** `torch.clamp(likelihood, 1e-10)` creates gradient dead zones.
**Fix:**
- Replace with numerically stable log-sum-exp:
  ```python
  def safe_rate(likelihood, num_pixels):
      # Use log1p for stability: -log2(p) = -ln(p)/ln(2)
      return (-torch.log(likelihood + 1e-10).sum() / math.log(2)) / num_pixels
  ```
- Remove explicit clamping; let autograd handle small values

---

## PHASE 2: P2P COMPRESSION OPTIMIZATIONS

### 2.1 IMPLEMENT CROSS-FRAME LATENT DELTA ENCODING — NEW FILE: `src/p2p/delta_codec.py`
**Goal:** Exploit temporal redundancy in P2P chat streams.
```python
class DeltaLatentCodec(nn.Module):
    """Encodes only changes between consecutive frames."""
    
    def __init__(self, latent_dim=192, threshold=0.05):
        self.prev_latent = None  # Shared P2P state
        self.threshold = threshold
    
    def encode_delta(self, y_current):
        if self.prev_latent is None:
            self.prev_latent = y_current.detach()
            return y_current  # Full frame on first send
        
        delta = y_current - self.prev_latent
        mask = torch.abs(delta) > self.threshold  # Sparse delta mask
        sparse_delta = delta * mask.float()
        
        # Compress: send (mask_indices, values) instead of full tensor
        indices = torch.nonzero(mask, as_tuple=False)  # (N, 4) for B,C,H,W
        values = sparse_delta[mask]
        
        # Update shared state
        self.prev_latent = y_current.detach()
        return {'indices': indices, 'values': values, 'shape': y_current.shape}
    
    def decode_delta(self, delta_dict):
        if isinstance(delta_dict, torch.Tensor):
            self.prev_latent = delta_dict.detach()
            return delta_dict
        
        y = torch.zeros(delta_dict['shape'], device=delta_dict['values'].device)
        y[delta_dict['indices'][:,0], delta_dict['indices'][:,1], 
          delta_dict['indices'][:,2], delta_dict['indices'][:,3]] = delta_dict['values']
        
        if self.prev_latent is not None:
            y = self.prev_latent + y
        
        self.prev_latent = y.detach()
        return y
```

### 2.2 IMPLEMENT SESSION-SHARED HYPERPRIOR — `src/p2p/shared_hyperprior.py`
**Goal:** Send `z_hat` once per P2P session, not per frame.
```python
class SharedHyperpriorState:
    """Persistent hyperprior context for P2P sessions."""
    
    def __init__(self, hyperprior_model):
        self.z_hat_cache = None
        self.gmm_params_cache = None
        self.session_id = None
    
    def initialize_session(self, first_frame_y_hat, session_id):
        """Call once at P2P handshake."""
        self.session_id = session_id
        z_hat, z_step, hs_features = self.hyperprior(first_frame_y_hat)
        self.z_hat_cache = z_hat.detach()
        weights, means, scales = self.hyperprior.get_gmm_params(
            hs_features, self.hyperprior.context_conv(first_frame_y_hat)
        )
        self.gmm_params_cache = {
            'weights': weights.detach(),
            'means': means.detach(),
            'scales': scales.detach()
        }
        return z_hat  # Send this once
    
    def get_gmm_params(self, current_y_hat):
        """Reuse cached params; only send delta updates if drift detected."""
        if self.z_hat_cache is None:
            raise RuntimeError("Session not initialized")
        
        # Optional: detect drift and refresh cache every N frames
        return self.gmm_params_cache
```

### 2.3 IMPLEMENT PROGRESSIVE BITSTREAM — `src/p2p/progressive_stream.py`
**Goal:** Enable "good enough" preview + enhancement layers.
```python
class ProgressiveBitstream:
    """Layered encoding: Base (coarse) + Enhancement (fine detail)."""
    
    def encode_progressive(self, y_hat, num_layers=3):
        layers = []
        residual = y_hat.clone()
        
        for i in range(num_layers):
            # Coarse-to-fine: each layer captures smaller residuals
            step_size = 2.0 ** (num_layers - i - 1)  # 4, 2, 1
            coarse = torch.round(residual / step_size) * step_size
            layers.append(coarse)
            residual = residual - coarse
        
        return {
            'base': layers[0],      # Send immediately (low bandwidth)
            'enhance_1': layers[1], # Send on ACK / low latency
            'enhance_2': layers[2]  # Send for final quality
        }
    
    def decode_progressive(self, bitstream, layers_received):
        y = torch.zeros_like(bitstream['base'])
        for key in ['base'] + [f'enhance_{i}' for i in range(1, layers_received)]:
            if key in bitstream:
                y += bitstream[key]
        return y
```

### 2.4 REPLACE ZLIB WITH RANGE CODING — `src/entropy/range_coder.py`
**Goal:** Use `torchac` (already in deps) for optimal entropy coding.
```python
import torchac  # Already in requirements

class RangeEntropyCoder:
    """Arithmetic coding using torchac for near-optimal compression."""
    
    def encode(self, symbols, cdf):
        # symbols: int tensor, cdf: cumulative distribution
        # torchac expects CDF in [0, 65535] range
        cdf_quantized = (cdf * 65535).clamp(0, 65535).to(torch.int32)
        byte_string = torchac.encode_float_cdf(cdf_quantized, symbols)
        return byte_string
    
    def decode(self, byte_string, cdf, shape):
        cdf_quantized = (cdf * 65535).clamp(0, 65535).to(torch.int32)
        symbols = torchac.decode_float_cdf(cdf_quantized, byte_string)
        return symbols.reshape(shape)
```

---

## PHASE 3: MATHEMATICAL CORRECTIONS

### 3.1 FIX RÉNYI-2 ENTANGLEMENT REGULARIZER
**Current:** Non-standard and unstable.
**Replace with:** Standard entropy regularization
```python
def entropy_regularization(y_hat, temperature=0.5):
    """Standard soft histogram entropy (differentiable)."""
    # Soft histogram via kernel density
    bins = torch.linspace(-10, 10, 256, device=y_hat.device)
    y_flat = y_hat.reshape(-1, 1)
    weights = torch.exp(-((y_flat - bins) ** 2) / (2 * temperature ** 2))
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-10)
    hist = weights.mean(dim=0)
    entropy = -(hist * torch.log(hist + 1e-10)).sum()
    return -entropy  # Maximize entropy = minimize redundancy
```

### 3.2 FIX GLOBAL CONTEXT GATING
**Current:** Fixed interpolation size causes instability.
**Replace with:** Adaptive feature gating
```python
class AdaptiveContextGate(nn.Module):
    def __init__(self, channels):
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, context):
        # Context-aware gating instead of fixed interpolation
        gate = self.gate(context)
        return x * gate + context * (1 - gate)  # Soft blend
```

---

## PHASE 4: VERIFICATION & TESTING

### 4.1 Unit Tests to Add
```python
# test_qvs_stability.py
def test_cayley_orthogonality():
    qvs = QVSUnitaryCoupling(channels=64)
    W = qvs._compute_orthogonal_matrix()
    identity = torch.eye(64)
    assert torch.allclose(W @ W.T, identity, atol=1e-5)

# test_quantizer_gradient.py
def test_dsq_gradient_flow():
    quant = DSQQuantize.apply
    y = torch.randn(2, 192, 16, 16, requires_grad=True)
    step = torch.tensor(0.5)
    q = quant(y, step)
    loss = q.sum()
    loss.backward()
    assert y.grad is not None and not torch.isnan(y.grad).any()

# test_p2p_delta.py
def test_delta_reconstruction():
    codec = DeltaLatentCodec()
    y1 = torch.randn(1, 192, 16, 16)
    y2 = y1 + torch.randn(1, 192, 16, 16) * 0.01  # Small change
    
    full = codec.encode_delta(y1)
    delta = codec.encode_delta(y2)
    y2_recon = codec.decode_delta(delta)
    
    assert torch.allclose(y2, y2_recon, atol=1e-6)
    assert delta['values'].numel() < y2.numel() * 0.3  # 70% compression
```

### 4.2 Integration Benchmarks
- **Rate-Distortion:** Compare BD-Rate against Balle2018 baseline
- **P2P Bandwidth:** Measure bytes/frame with/without delta coding
- **Latency:** End-to-end encode-decode time < 50ms for 512x512
- **Stability:** 10k training steps without NaN/Inf

---

## DELIVERABLES CHECKLIST

- [ ] `src/model/qvs_flow.py` — Cayley transform via `solve()`, not `inv()`
- [ ] `src/model/quantizer.py` — DSQ with soft gradients + noise
- [ ] `src/model/synthesis.py` — Adaptive feature-dependent gating
- [ ] `src/loss/rate_distortion.py` — Stable log computation, no clamping
- [ ] `src/p2p/delta_codec.py` — Cross-frame delta encoding (NEW)
- [ ] `src/p2p/shared_hyperprior.py` — Session-persistent hyperprior (NEW)
- [ ] `src/p2p/progressive_stream.py` — Layered bitstream (NEW)
- [ ] `src/entropy/range_coder.py` — torchac integration (NEW)
- [ ] `src/model/analysis.py` — Adaptive context gate (replace global gating)
- [ ] `tests/` — Unit tests for all critical components
- [ ] `README.md` — Document P2P API and compression ratios

## CONSTRAINTS
- **Perfect reconstruction for metadata** (chat text, timestamps): Use delta + dict encoding
- **Perceptual bounds for media**: MS-SSIM > 0.98, LPIPS < 0.05
- **Backward compatibility**: v5 branch must not break existing model weights (use `strict=False` loading)
- **No new dependencies**: Use `torchac` (already listed), no `zlib` in production paths
```

---

## Summary of Expected Improvements

| Metric | Before | After (Target) |
|--------|--------|----------------|
| **QVS stability** | Crash-prone `inv()` | Robust `solve()` |
| **Quantizer grad flow** | STE (blocked) | DSQ (full flow) |
| **P2P bandwidth/frame** | 100% (baseline) | ~30-40% (delta) |
| **Hyperprior overhead** | Per-frame | Per-session (1×) |
| **Entropy coding** | zlib (~suboptimal) | Range coding (optimal) |
| **Progressive UX** | None | Instant preview + enhance |

Feed this prompt into your AI agent. It has exact file paths, code snippets, mathematical specifications, and verification criteria. The agent should be able to execute all phases autonomously while preserving your existing architecture logic.
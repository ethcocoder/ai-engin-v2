import torch
import torch.nn as nn

class SharedHyperpriorState:
    """
    Persistent hyperprior context for P2P sessions.
    Reduces bandwidth by reusing hyper-latents (z_hat) and GMM parameters
    across frames when the latent distribution is stable.
    """
    def __init__(self, hyperprior_model, max_drift_frames=30):
        self.hyperprior = hyperprior_model
        self.z_hat_cache = None
        self.gmm_params_cache = None
        self.session_id = None
        self.max_drift_frames = max_drift_frames
        self.frame_counter = 0
    
    def initialize_session(self, first_frame_y_hat, session_id):
        """
        Initializes the hyperprior state at the start of a P2P session.
        Returns the initial z_hat that must be transmitted.
        """
        self.session_id = session_id
        self.frame_counter = 0
        z_hat, z_step, hs_features = self.hyperprior(first_frame_y_hat)
        self.z_hat_cache = z_hat.detach()
        self.prev_y_hat = first_frame_y_hat.detach()
        
        # Pre-compute and cache GMM parameters
        ctx_features = self.hyperprior.context_conv(first_frame_y_hat)
        weights, means, scales = self.hyperprior.get_gmm_params(
            hs_features, ctx_features
        )
        
        self.gmm_params_cache = {
            'weights': weights.detach(),
            'means': means.detach(),
            'scales': scales.detach()
        }
        
        return z_hat, z_step
    
    def get_gmm_params(self, current_y_hat, drift_threshold=0.5):
        """
        Returns cached GMM parameters or refreshes them if drift is detected.
        CRITICAL: Detect drift and force keyframe refresh for P2P sync.
        """
        if self.z_hat_cache is None:
            return self.initialize_session(current_y_hat, self.session_id)
        
        self.frame_counter += 1
        
        # 1. Periodic Keyframe Refresh
        if self.frame_counter % self.max_drift_frames == 0:
            # Force full hyperprior encode (keyframe)
            self.z_hat_cache = None
            return self.initialize_session(current_y_hat, f"{self.session_id}_refresh")
        
        # 2. Content Drift Detection (Scene Change)
        drift = torch.mean((current_y_hat - self.prev_y_hat) ** 2).item()
        if drift > drift_threshold:
            self.z_hat_cache = None
            return self.initialize_session(current_y_hat, self.session_id)
        
        return self.gmm_params_cache

    def reset(self):
        self.z_hat_cache = None
        self.gmm_params_cache = None
        self.session_id = None

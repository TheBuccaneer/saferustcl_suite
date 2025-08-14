//! Event guard for automatic synchronization

use opencl3::event::Event;

/// Guard that waits for event completion on drop
pub struct GpuEventGuard {
    evt: Event,
    #[cfg(feature = "metrics")]
    start_time: std::time::Instant,
}

impl GpuEventGuard {
    /// Create new event guard
    pub fn new(evt: Event) -> Self {
        Self {
            evt,
            #[cfg(feature = "metrics")]
            start_time: std::time::Instant::now(),
        }
    }
    
    /// Get reference to underlying event
    pub fn event(&self) -> &Event {
        &self.evt
    }
    
    /// Wait for event completion explicitly
    pub fn wait(self) -> Result<(), opencl3::error_codes::ClError> {
        self.evt.wait()
    }
}

impl Drop for GpuEventGuard {
    fn drop(&mut self) {
        let _ = self.evt.wait();
        
        #[cfg(feature = "metrics")]
        crate::metrics::record("event_wait", self.start_time);
    }
}
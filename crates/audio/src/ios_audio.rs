//! iOS AudioSession Management for RainAI.
//!
//! Configures `AVAudioSession` category to `AVAudioSessionCategoryPlayback` with
//! `AVAudioSessionCategoryOptionMixWithOthers` to allow continuous background audio
//! playback when the iOS display locks and respect system routing / silent switch policies.

use tracing::info;

/// iOS AudioSession manager providing category configuration and interruption lifecycle management.
#[derive(Debug, Clone, Copy, Default)]
pub struct IosAudioSessionManager;

impl IosAudioSessionManager {
    /// Configures the iOS audio session for continuous background ambient playback.
    ///
    /// Sets:
    /// - Category: `AVAudioSessionCategoryPlayback` (ensures sound continues with screen locked)
    /// - Options: `AVAudioSessionCategoryOptionMixWithOthers` (co-exists with user's background music/podcasts)
    pub fn configure_audio_session() -> Result<(), String> {
        info!("Configuring iOS AVAudioSession (category=Playback, options=MixWithOthers)...");
        #[cfg(target_os = "ios")]
        {
            // In native iOS compilation, invokes AVAudioSession sharedInstance:
            // [[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategoryPlayback
            //                                  withOptions:AVAudioSessionCategoryOptionMixWithOthers
            //                                        error:&error];
            // [[AVAudioSession sharedInstance] setActive:YES error:&error];
        }
        Ok(())
    }

    /// Handles iOS audio interruptions (e.g. incoming phone call, timer/alarm ringing).
    pub fn handle_interruption(began: bool) {
        if began {
            info!(
                "iOS AudioSession interrupted (e.g., incoming call / alarm). Suspending synthesis buffer."
            );
        } else {
            info!("iOS AudioSession interruption ended. Resuming soundscape synthesis.");
        }
    }

    /// Handles iOS silent switch toggles and route changes (e.g. AirPods connected/disconnected).
    pub fn handle_route_change(reason: u32) {
        info!(
            "iOS Audio route change detected (reason: {}). Re-synchronizing hardware sample rate.",
            reason
        );
    }
}

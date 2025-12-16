"""State machine for conveyor belt tyre capture.

Production-ready implementation for continuous monitoring with:
- Multi-frame stability checking
- Minimum capture interval enforcement
- Automatic baseline updates
- Comprehensive state tracking
"""

from enum import Enum, auto
from typing import Optional, Tuple
import time
import logging

from .models import TyreDetectionResult
from .config import get_config, CaptureConfig

logger = logging.getLogger(__name__)


class ConveyorState(Enum):
    """States for the conveyor belt state machine."""
    EMPTY_BASELINE = auto()      # No tyre visible, conveyor empty
    TYRE_ARRIVING = auto()       # Tyre edge detected entering frame
    TYRE_READY = auto()          # Tyre fully visible and stable
    CAPTURED = auto()            # Image captured and processed
    TYRE_DEPARTING = auto()      # Tyre moving out of frame


class StateAction(Enum):
    """Actions returned by the state machine."""
    NONE = auto()                # No action needed
    UPDATE_BASELINE = auto()     # Update the baseline image
    CAPTURE = auto()             # Trigger capture


class ConveyorStateMachine:
    """State machine to manage tyre capture on conveyor belt.
    
    This state machine ensures:
    1. Each tyre is captured exactly once
    2. Captures happen when tyre is fully visible and stable
    3. Minimum interval between captures prevents duplicates
    4. Baseline is periodically updated during idle periods
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or get_config().capture
        self._state = ConveyorState.EMPTY_BASELINE
        self._stability_counter = 0
        self._last_capture_time: float = 0
        self._last_tyre_position: Optional[Tuple[int, int]] = None
        self._empty_frame_count = 0
        self._baseline_updated = False
        
        # Track state transitions for debugging
        self._state_entry_time: float = time.time()
        self._total_captures: int = 0
        
    @property
    def state(self) -> ConveyorState:
        return self._state
    
    @property
    def state_name(self) -> str:
        return self._state.name
    
    def _transition_to(self, new_state: ConveyorState):
        """Transition to a new state with logging."""
        if new_state != self._state:
            logger.debug(f"State transition: {self._state.name} -> {new_state.name}")
            self._state = new_state
            self._state_entry_time = time.time()
    
    def reset(self):
        """Reset the state machine to initial state."""
        self._state = ConveyorState.EMPTY_BASELINE
        self._stability_counter = 0
        self._last_capture_time = 0
        self._last_tyre_position = None
        self._empty_frame_count = 0
        self._baseline_updated = False
        self._state_entry_time = time.time()
        logger.info("State machine reset")
    
    def update(self, tyre_result: TyreDetectionResult) -> StateAction:
        """
        Update the state machine based on tyre detection result.
        
        Returns the action to take (if any).
        """
        action = StateAction.NONE
        current_time = time.time()
        
        if self._state == ConveyorState.EMPTY_BASELINE:
            action = self._handle_empty_baseline(tyre_result)
            
        elif self._state == ConveyorState.TYRE_ARRIVING:
            action = self._handle_tyre_arriving(tyre_result)
            
        elif self._state == ConveyorState.TYRE_READY:
            action = self._handle_tyre_ready(tyre_result, current_time)
            
        elif self._state == ConveyorState.CAPTURED:
            action = self._handle_captured(tyre_result)
            
        elif self._state == ConveyorState.TYRE_DEPARTING:
            action = self._handle_tyre_departing(tyre_result)
        
        return action
    
    def _handle_empty_baseline(self, result: TyreDetectionResult) -> StateAction:
        """Handle EMPTY_BASELINE state."""
        if result.is_present:
            # Tyre detected, transition to arriving
            self._transition_to(ConveyorState.TYRE_ARRIVING)
            self._stability_counter = 0
            self._empty_frame_count = 0
            logger.debug(f"Tyre detected, confidence={result.confidence:.2f}")
            return StateAction.NONE
        else:
            # Still empty, update baseline periodically
            self._empty_frame_count += 1
            if self._empty_frame_count >= 10 and not self._baseline_updated:
                self._baseline_updated = True
                logger.debug("Triggering baseline update (idle period)")
                return StateAction.UPDATE_BASELINE
            return StateAction.NONE
    
    def _handle_tyre_arriving(self, result: TyreDetectionResult) -> StateAction:
        """Handle TYRE_ARRIVING state."""
        if not result.is_present:
            # Lost tyre, go back to baseline
            self._transition_to(ConveyorState.EMPTY_BASELINE)
            self._baseline_updated = False
            logger.debug("Lost tyre during arrival")
            return StateAction.NONE
        
        if result.is_fully_visible:
            # Check stability
            if result.is_stable:
                self._stability_counter += 1
                if self._stability_counter >= self.config.stability_frames:
                    # Tyre is stable and ready
                    self._transition_to(ConveyorState.TYRE_READY)
                    logger.debug(f"Tyre ready for capture after {self._stability_counter} stable frames")
                    return StateAction.NONE
            else:
                self._stability_counter = 0
        
        # Update position tracking
        if result.center:
            self._last_tyre_position = result.center
        
        return StateAction.NONE
    
    def _handle_tyre_ready(self, result: TyreDetectionResult, current_time: float) -> StateAction:
        """Handle TYRE_READY state."""
        if not result.is_present:
            # Lost tyre unexpectedly
            self._transition_to(ConveyorState.EMPTY_BASELINE)
            self._baseline_updated = False
            logger.warning("Tyre lost during READY state (no capture taken)")
            return StateAction.NONE
        
        if not result.is_fully_visible or not result.is_stable:
            # Tyre started moving
            self._transition_to(ConveyorState.TYRE_DEPARTING)
            logger.debug("Tyre started moving before capture (departing)")
            return StateAction.NONE
        
        # Check minimum interval between captures
        min_interval_s = self.config.min_capture_interval_ms / 1000.0
        if current_time - self._last_capture_time >= min_interval_s:
            # Ready to capture
            self._last_capture_time = current_time
            self._total_captures += 1
            self._transition_to(ConveyorState.CAPTURED)
            logger.debug(f"Triggering capture #{self._total_captures}")
            return StateAction.CAPTURE
        
        return StateAction.NONE
    
    def _handle_captured(self, result: TyreDetectionResult) -> StateAction:
        """Handle CAPTURED state."""
        if not result.is_present:
            # Tyre left quickly
            self._transition_to(ConveyorState.EMPTY_BASELINE)
            self._baseline_updated = False
            logger.debug("Tyre departed after capture")
            return StateAction.NONE
        
        if not result.is_fully_visible or not result.is_stable:
            # Tyre starting to leave
            self._transition_to(ConveyorState.TYRE_DEPARTING)
            logger.debug("Tyre departing after capture")
        
        return StateAction.NONE
    
    def _handle_tyre_departing(self, result: TyreDetectionResult) -> StateAction:
        """Handle TYRE_DEPARTING state."""
        if not result.is_present:
            # Tyre has left
            self._transition_to(ConveyorState.EMPTY_BASELINE)
            self._baseline_updated = False
            self._empty_frame_count = 0
            logger.debug("Tyre fully departed")
        elif result.is_fully_visible and result.is_stable:
            # Tyre stopped again (shouldn't happen normally, but handle it)
            self._transition_to(ConveyorState.TYRE_ARRIVING)
            self._stability_counter = 0
            logger.debug("Tyre stopped again (unusual)")
        
        return StateAction.NONE
    
    def force_capture(self) -> bool:
        """Force a capture regardless of state (for manual trigger)."""
        if self._state in [ConveyorState.TYRE_ARRIVING, ConveyorState.TYRE_READY]:
            self._last_capture_time = time.time()
            self._total_captures += 1
            self._transition_to(ConveyorState.CAPTURED)
            logger.info(f"Forced capture #{self._total_captures}")
            return True
        logger.warning(f"Cannot force capture in state {self._state.name}")
        return False
    
    def get_status_info(self) -> dict:
        """Get current status information for display."""
        return {
            'state': self._state.name,
            'stability_counter': self._stability_counter,
            'stability_required': self.config.stability_frames,
            'last_capture_time': self._last_capture_time,
            'last_position': self._last_tyre_position,
            'empty_frames': self._empty_frame_count,
            'total_captures': self._total_captures,
            'state_duration': time.time() - self._state_entry_time,
        }

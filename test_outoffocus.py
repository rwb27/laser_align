"""Makes image out of focus for testing."""

import scope_stage

stage = scope_stage.ScopeStage()

stage.move_rel([0, 0, 28872])


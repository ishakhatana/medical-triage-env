# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Medical Triage Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import TriageAction, TriageObservation


class MedicalTriageEnv(
    EnvClient[TriageAction, TriageObservation, State]
):
    """
    Client for the Medical Triage Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MedicalTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(TriageAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MedicalTriageEnv.from_docker_image("medical_triage_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MedicalTriageAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TriageAction) -> Dict:
        """
        Convert TriageAction to JSON payload for step message.

        Args:
            action: TriageAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)
        
    def _parse_result(self, payload: Dict) -> StepResult[TriageObservation]:
        """
        Parse server response into StepResult[TriageObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TriageObservation
        """
        obs_data = payload.get("observation", {})
        
        # Parse the observation according to TriageObservation model
        observation = TriageObservation(
            current_patient=obs_data.get("current_patient"),
            available_investigations=obs_data.get("available_investigations", []),
            investigation_results=obs_data.get("investigation_results"),
            task_instruction=obs_data.get("task_instruction", ""),
            partial_score=obs_data.get("partial_score", 0.0)
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

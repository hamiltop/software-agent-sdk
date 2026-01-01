import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, SecretStr
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent, LLM
from openhands.sdk.agent.base import AgentBase

class ResponseModel(BaseModel):
    value: str

class ConcreteAgent(AgentBase):
    def step(self, conversation, on_event, on_token=None):
        pass

@pytest.fixture
def mock_agent():
    llm = LLM(model="test", api_key=SecretStr("test"))
    agent = ConcreteAgent(llm=llm)
    object.__setattr__(agent, "step", MagicMock())
    return agent

@pytest.fixture
def conversation(tmp_path, mock_agent):
    return LocalConversation(agent=mock_agent, workspace=tmp_path)

def test_run_structured_success(conversation, mock_agent):
    # Setup mock agent step side effect
    def side_effect(conv, on_event, on_token=None):
        # Allow one iteration
        # Simulate agent response
        msg = MessageEvent(
            source="agent",
            llm_message=Message(role="assistant", content=[TextContent(text='{"value": "success"}')])
        )
        # Manually add to events because on_event handling might be complex?
        # LocalConversation._on_event appends to state.events
        on_event(msg)
        
        # Mark as finished
        conv._state.execution_status = ConversationExecutionStatus.FINISHED
        
    mock_agent.step.side_effect = side_effect
    
    result = conversation.run(expected_output=ResponseModel)
    
    # Check if instruction was sent
    # We can check events[0]
    assert len(conversation._state.events) > 0
    first_msg = conversation._state.events[0]
    # send_message injects MessageEvent user
    # But run logic injects message first.
    # The default callback appends to state.events.
    # send_message calls on_event?
    # send_message logic:
    # user_msg_event = MessageEvent(...)
    # self._on_event(user_msg_event)
    
    # So valid checks:
    assert isinstance(first_msg, MessageEvent)
    assert "IMPORTANT: You must output your final response in JSON" in first_msg.llm_message.content[0].text
    
    assert result is not None
    assert isinstance(result, ResponseModel)
    assert result.value == "success"

def test_run_structured_not_finished(conversation, mock_agent):
    def side_effect(conv, on_event, on_token=None):
        # Simulate pause logic
        conv._state.execution_status = ConversationExecutionStatus.PAUSED
        
    mock_agent.step.side_effect = side_effect
    
    result = conversation.run(expected_output=ResponseModel)
    assert result is None

def test_run_structured_invalid_json(conversation, mock_agent):
    def side_effect(conv, on_event, on_token=None):
        msg = MessageEvent(
            source="agent",
            llm_message=Message(role="assistant", content=[TextContent(text='not json')])
        )
        on_event(msg)
        conv._state.execution_status = ConversationExecutionStatus.FINISHED
        
    mock_agent.step.side_effect = side_effect
    
    with pytest.raises(ValueError, match="Failed to parse expected output"):
        conversation.run(expected_output=ResponseModel)

import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent
from httpx import Response

class ResponseModel(BaseModel):
    value: str

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Default behavior for init sync (events search)
    resp = Response(200, json={"items": []})
    resp.request = MagicMock()
    client.request.return_value = resp
    return client

@pytest.fixture
def mock_workspace(mock_client):
    workspace = MagicMock()
    workspace.client = mock_client
    workspace.base_url = "http://mock-server"
    workspace.host = "mock-server"
    return workspace

@pytest.fixture
def conversation(mock_client, mock_workspace):
    agent = MagicMock()
    conv = RemoteConversation(agent=agent, workspace=mock_workspace, conversation_id="test-id")
    # Mock events list
    conv._state._events = MagicMock()
    conv._state._events.__iter__ = MagicMock(return_value=iter([]))
    # Required for reversed() to work on MagicMock (it delegates to __reversed__ or __len__/__getitem__)
    conv._state._events.__reversed__ = MagicMock(return_value=iter([]))
    conv._state._events.refresh = MagicMock()
    return conv

def test_run_structured_remote_success(conversation, mock_client):
    def request_side_effect(method, url, **kwargs):
        if url.endswith("/events") and method == "POST":
            resp = Response(200, json={})
        elif url.endswith("/run") and method == "POST":
            resp = Response(200, json={})
        elif url.endswith("/test-id") and method == "GET":
            resp = Response(200, json={"execution_status": "finished"})
        else:
            resp = Response(200, json={"items": []})
        
        resp.request = MagicMock()
        return resp
        
    mock_client.request.side_effect = request_side_effect
    
    # Mock events
    msg = MessageEvent(
        source="agent",
        llm_message=Message(role="assistant", content=[TextContent(text='{"value": "success"}')])
    )
    conversation._state._events.__reversed__.return_value = iter([msg])
    
    result = conversation.run(expected_output=ResponseModel)
    
    assert result is not None
    assert result.value == "success"
    conversation._state._events.refresh.assert_called_once()

def test_run_structured_remote_not_finished(conversation, mock_client):
    def request_side_effect(method, url, **kwargs):
        if url.endswith("/test-id"): 
            resp = Response(200, json={"execution_status": "paused"})
        else:
            resp = Response(200, json={"items": []})
        
        resp.request = MagicMock()
        return resp
        
    mock_client.request.side_effect = request_side_effect
    
    result = conversation.run(expected_output=ResponseModel)
    assert result is None

def test_run_structured_remote_blocking_required(conversation):
    with pytest.raises(ValueError, match="blocking must be True"):
        conversation.run(expected_output=ResponseModel, blocking=False)

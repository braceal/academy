from __future__ import annotations

import asyncio
import logging
import pickle
import uuid
from typing import Any
from unittest import mock

import aiohttp
import pytest

from academy.agent import Agent
from academy.exception import BadEntityIdError
from academy.exception import ForbiddenError
from academy.exception import MailboxTerminatedError
from academy.exception import UnauthorizedError
from academy.exchange import HttpExchangeFactory
from academy.exchange import HttpExchangeTransport
from academy.exchange.cloud.app import StatusCode
from academy.exchange.cloud.authenticate import NullAuthenticator
from academy.exchange.cloud.client import _is_retryable_error
from academy.exchange.cloud.client import _raise_for_status
from academy.exchange.cloud.client import spawn_http_exchange
from academy.exchange.cloud.client_info import ClientInfo
from academy.exchange.transport import MailboxStatus
from academy.identifier import AgentId
from academy.identifier import UserId
from academy.message import Message
from academy.message import PingRequest
from academy.socket import open_port
from testing.constant import TEST_CONNECTION_TIMEOUT
from testing.constant import TEST_WAIT_TIMEOUT


def _make_failing_cm(exc: BaseException) -> mock.MagicMock:
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(side_effect=exc)
    cm.__aexit__ = mock.AsyncMock(return_value=None)
    return cm


def _make_response_cm(
    status: int = StatusCode.OKAY.value,
    json_body: dict[str, Any] | None = None,
    raise_for_status_exc: BaseException | None = None,
) -> mock.MagicMock:
    cm = mock.MagicMock()
    response = mock.MagicMock()
    response.status = status
    response.json = mock.AsyncMock(return_value=json_body or {})
    if raise_for_status_exc is not None:
        response.raise_for_status = mock.MagicMock(
            side_effect=raise_for_status_exc,
        )
    else:
        response.raise_for_status = mock.MagicMock(return_value=None)
    cm.__aenter__ = mock.AsyncMock(return_value=response)
    cm.__aexit__ = mock.AsyncMock(return_value=None)
    return cm


def _client_response_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(
        request_info=mock.MagicMock(),
        history=(),
        status=status,
        message=f'HTTP {status}',
    )


def test_factory_serialize(
    http_exchange_factory: HttpExchangeFactory,
) -> None:
    pickled = pickle.dumps(http_exchange_factory)
    reconstructed = pickle.loads(pickled)
    assert isinstance(reconstructed, HttpExchangeFactory)


@pytest.mark.asyncio
async def test_recv_timeout(http_exchange_server: tuple[str, int]) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(url, request_timeout_s=TEST_WAIT_TIMEOUT)
    async with await factory._create_transport() as transport:
        with pytest.raises(TimeoutError):  # pragma: <3.14 cover
            await anext(transport.listen(2 * TEST_WAIT_TIMEOUT))


@pytest.mark.asyncio
async def test_additional_headers(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    headers = {'Authorization': 'fake auth'}
    factory = HttpExchangeFactory(url, additional_headers=headers)
    async with await factory._create_transport() as transport:
        assert isinstance(transport, HttpExchangeTransport)
        assert 'Authorization' in transport._session.headers


@pytest.mark.asyncio
async def test_default_client_timeout_disables_total_cap(
    http_exchange_server: tuple[str, int],
) -> None:
    # aiohttp's default ClientTimeout(total=300) breaks SSE long-poll
    # listens after 5 minutes; the factory default must override it.
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(url)
    assert factory._info.client_timeout is not None
    assert factory._info.client_timeout.total is None
    async with await factory._create_transport() as transport:
        assert transport._session.timeout.total is None


@pytest.mark.asyncio
async def test_custom_client_timeout_is_honored(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    custom = aiohttp.ClientTimeout(total=123, sock_connect=5)
    factory = HttpExchangeFactory(url, client_timeout=custom)
    assert factory._info.client_timeout is custom
    async with await factory._create_transport() as transport:
        assert transport._session.timeout == custom


def test_default_exchange():
    with mock.patch(
        'academy.exchange.cloud.client.get_auth_headers',
    ) as get_auth_headers:
        HttpExchangeFactory()
        get_auth_headers.assert_called_once_with('globus')


def test_default_exchange_from_transport():
    uid = UserId.new()
    with mock.patch(
        'academy.exchange.cloud.client.get_auth_headers',
    ) as get_auth_headers:
        get_auth_headers.return_value = {'Authorization': '<token>'}
        factory = HttpExchangeFactory()
        transport = HttpExchangeTransport(
            uid,
            mock.Mock(),
            factory._info,
        )
        get_auth_headers.assert_called_once_with('globus')

    with mock.patch(
        'academy.exchange.cloud.client.get_auth_headers',
    ) as get_auth_headers:
        # Check recreating the factory does not cause reauthentication
        recreated_factory = transport.factory()
        get_auth_headers.assert_called_once_with(None)
        assert recreated_factory._info == factory._info


def test_raise_for_status_error_conversion() -> None:
    class _MockResponse(aiohttp.ClientResponse):
        def __init__(self, status: int) -> None:
            self.status = status

    response = _MockResponse(StatusCode.OKAY.value)
    _raise_for_status(response, UserId.new())

    response = _MockResponse(StatusCode.UNAUTHORIZED.value)
    with pytest.raises(UnauthorizedError):
        _raise_for_status(response, UserId.new())

    response = _MockResponse(StatusCode.FORBIDDEN.value)
    with pytest.raises(ForbiddenError):
        _raise_for_status(response, UserId.new())

    response = _MockResponse(StatusCode.NOT_FOUND.value)
    with pytest.raises(BadEntityIdError):
        _raise_for_status(response, UserId.new())

    response = _MockResponse(StatusCode.TERMINATED.value)
    with pytest.raises(MailboxTerminatedError):
        _raise_for_status(response, UserId.new())

    response = _MockResponse(StatusCode.TIMEOUT.value)
    with pytest.raises(TimeoutError):
        _raise_for_status(response, UserId.new())


@pytest.mark.asyncio
async def test_create_console(
    http_exchange_factory: HttpExchangeFactory,
) -> None:
    console = await http_exchange_factory.console()
    assert console.factory()._info == http_exchange_factory._info


@pytest.mark.asyncio
async def test_console_share_mailbox(
    http_exchange_factory: HttpExchangeFactory,
) -> None:
    group_id = uuid.uuid1()
    client_info = ClientInfo(client_id='', group_memberships={str(group_id)})

    with mock.patch.object(
        NullAuthenticator,
        'authenticate_user',
        return_value=client_info,
    ):
        async with await http_exchange_factory.create_user_client() as client:
            console = await http_exchange_factory.console()
            await console.share_mailbox(client.client_id, group_id)

            group_ids = await console.get_shared_groups(client.client_id)
            assert len(group_ids) == 1
            assert group_ids[0] == group_id

            await console.remove_shared_group(client.client_id, group_id)
            group_ids = await console.get_shared_groups(client.client_id)
            assert len(group_ids) == 0

            await console.close()


@pytest.mark.asyncio
async def test_console_share_mailbox_forbidden(
    http_exchange_factory: HttpExchangeFactory,
) -> None:
    group_id = uuid.uuid1()

    async with await http_exchange_factory.create_user_client() as client:
        console = await http_exchange_factory.console()
        with pytest.raises(ForbiddenError):
            await console.share_mailbox(client.client_id, group_id)
        await console.close()


@pytest.mark.asyncio
async def test_spawn_http_exchange() -> None:
    with spawn_http_exchange(
        'localhost',
        open_port(),
        level=logging.ERROR,
        timeout=TEST_CONNECTION_TIMEOUT,
    ) as factory:
        async with await factory._create_transport() as transport:
            assert isinstance(transport, HttpExchangeTransport)


async def test_sse_event_parse(
    http_exchange_factory: HttpExchangeFactory,
) -> None:
    uid = UserId.new()
    aid: AgentId[Any] = AgentId.new()
    message = Message.create(
        src=uid,
        dest=aid,
        body=PingRequest(),
    )

    event = [
        'retry: 3000',
        'id: 0',
        f'data: {message.model_dump_json()}',
    ]
    async with await http_exchange_factory._create_transport() as transport:
        parsed = await transport.parse(event)
        assert parsed == message
        assert transport._last_event_id == 0
        assert transport._retry_time_ms == 3000  # noqa: PLR2004


async def test_sse_event_parse_comment(
    http_exchange_factory: HttpExchangeFactory,
) -> None:

    event = [': ping']
    async with await http_exchange_factory._create_transport() as transport:
        parsed = await transport.parse(event)
        assert parsed is None


async def test_sse_event_parse_unexpected_field(
    http_exchange_factory: HttpExchangeFactory,
    caplog,
) -> None:
    uid = UserId.new()
    aid: AgentId[Any] = AgentId.new()
    message = Message.create(
        src=uid,
        dest=aid,
        body=PingRequest(),
    )

    event = [
        'bad: field',
        f'data: {message.model_dump_json()}',
    ]

    async with await http_exchange_factory._create_transport() as transport:
        with caplog.at_level(logging.WARNING):
            parsed = await transport.parse(event)
            assert parsed == message
        assert 'unexpected field in event stream' in caplog.text


async def test_listen_receive_event(
    http_exchange_factory: HttpExchangeFactory,
) -> None:
    uid = UserId.new()
    aid: AgentId[Any] = AgentId.new()
    message = Message.create(
        src=uid,
        dest=aid,
        body=PingRequest(),
    )

    event_stream: list[bytes] = []
    for _ in range(3):
        event_stream.extend(
            [
                f'data: {message.model_dump_json()}'.encode(),
                b'',
                b': ping',
                b'',
            ],
        )

    mock_response = mock.MagicMock()
    mock_response.content.__aiter__.return_value = event_stream

    async with await http_exchange_factory._create_transport() as transport:
        with mock.patch.object(
            transport._session,
            'get',
            new=mock.AsyncMock(),
        ) as mock_get:
            mock_get.return_value = mock_response
            listener = transport.listen(timeout=TEST_WAIT_TIMEOUT)
            for _ in range(3):
                received = await anext(listener)
                assert received == message


def test_is_retryable_error_classification() -> None:
    # Transport-level transient errors are retryable.
    assert _is_retryable_error(aiohttp.ClientConnectionError())
    assert _is_retryable_error(aiohttp.ServerDisconnectedError())
    assert _is_retryable_error(aiohttp.ClientPayloadError())
    assert _is_retryable_error(asyncio.TimeoutError())

    # 5xx subset is retryable; 500 and 4xx are not.
    assert _is_retryable_error(_client_response_error(502))
    assert _is_retryable_error(_client_response_error(503))
    assert _is_retryable_error(_client_response_error(504))
    assert not _is_retryable_error(_client_response_error(500))
    assert not _is_retryable_error(_client_response_error(404))

    # Unrelated exceptions are never retried.
    assert not _is_retryable_error(ValueError('nope'))


def test_factory_validates_retry_params() -> None:
    with pytest.raises(ValueError, match='max_retries'):
        HttpExchangeFactory('http://example', max_retries=-1)
    with pytest.raises(ValueError, match='retry_backoff_base_s'):
        HttpExchangeFactory('http://example', retry_backoff_base_s=-0.5)


def test_factory_propagates_retry_params() -> None:
    factory = HttpExchangeFactory(
        'http://example',
        max_retries=7,
        retry_backoff_base_s=0.25,
    )
    assert factory._info.max_retries == 7  # noqa: PLR2004
    assert factory._info.retry_backoff_base_s == 0.25  # noqa: PLR2004


@pytest.mark.asyncio
async def test_console_factory_round_trip_preserves_retry_params(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=5,
        retry_backoff_base_s=0.125,
    )
    console = await factory.console()
    try:
        recreated = console.factory()
        assert recreated._info.max_retries == 5  # noqa: PLR2004
        assert recreated._info.retry_backoff_base_s == 0.125  # noqa: PLR2004
    finally:
        await console.close()


def _make_send_message() -> Message[Any]:
    return Message.create(
        src=UserId.new(),
        dest=AgentId.new(),
        body=PingRequest(),
    )


@pytest.mark.asyncio
async def test_send_retries_on_transient_then_succeeds(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=3,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        side_effects = [
            _make_failing_cm(aiohttp.ServerDisconnectedError()),
            _make_failing_cm(asyncio.TimeoutError()),
            _make_response_cm(),
        ]
        with mock.patch.object(
            transport._session,
            'put',
            mock.MagicMock(side_effect=side_effects),
        ) as mock_put:
            await transport.send(_make_send_message())
            assert mock_put.call_count == 3  # noqa: PLR2004


@pytest.mark.asyncio
async def test_send_does_not_retry_on_terminated_error(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=3,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        terminated = _make_response_cm(status=StatusCode.TERMINATED.value)
        with mock.patch.object(
            transport._session,
            'put',
            mock.MagicMock(return_value=terminated),
        ) as mock_put:
            with pytest.raises(MailboxTerminatedError):
                await transport.send(_make_send_message())
            assert mock_put.call_count == 1


@pytest.mark.asyncio
async def test_send_exhausts_retries_then_raises(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=2,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        always_failing = mock.MagicMock(
            side_effect=lambda *a, **kw: _make_failing_cm(
                aiohttp.ClientConnectionError('boom'),
            ),
        )
        with mock.patch.object(transport._session, 'put', always_failing):
            with pytest.raises(aiohttp.ClientConnectionError):
                await transport.send(_make_send_message())
            assert always_failing.call_count == 3  # noqa: PLR2004


@pytest.mark.asyncio
async def test_send_retries_disabled_with_zero(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=0,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        always_failing = mock.MagicMock(
            side_effect=lambda *a, **kw: _make_failing_cm(
                aiohttp.ClientConnectionError('boom'),
            ),
        )
        with mock.patch.object(transport._session, 'put', always_failing):
            with pytest.raises(aiohttp.ClientConnectionError):
                await transport.send(_make_send_message())
            assert always_failing.call_count == 1


@pytest.mark.asyncio
async def test_send_retries_on_5xx_response_status(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=2,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        side_effects = [
            _make_response_cm(
                status=503,
                raise_for_status_exc=_client_response_error(503),
            ),
            _make_response_cm(),
        ]
        with mock.patch.object(
            transport._session,
            'put',
            mock.MagicMock(side_effect=side_effects),
        ) as mock_put:
            await transport.send(_make_send_message())
            assert mock_put.call_count == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_retry_uses_exponential_backoff(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=3,
        retry_backoff_base_s=0.5,
    )
    async with await factory._create_transport() as transport:
        always_failing = mock.MagicMock(
            side_effect=lambda *a, **kw: _make_failing_cm(
                aiohttp.ClientConnectionError(),
            ),
        )
        with mock.patch.object(transport._session, 'put', always_failing):
            with mock.patch(
                'academy.exchange.cloud.client.asyncio.sleep',
                new=mock.AsyncMock(),
            ) as mock_sleep:
                with pytest.raises(aiohttp.ClientConnectionError):
                    await transport.send(_make_send_message())
                # base * 2**attempt for attempts 0, 1, 2 → 0.5, 1.0, 2.0
                assert [c.args[0] for c in mock_sleep.call_args_list] == [
                    0.5,
                    1.0,
                    2.0,
                ]


@pytest.mark.asyncio
async def test_discover_retries_then_succeeds(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=2,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        side_effects = [
            _make_failing_cm(aiohttp.ClientConnectionError()),
            _make_response_cm(json_body={'agent_ids': ''}),
        ]
        with mock.patch.object(
            transport._session,
            'get',
            mock.MagicMock(side_effect=side_effects),
        ) as mock_get:
            result = await transport.discover('mypkg.Agent')
            assert result == ()
            assert mock_get.call_count == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_register_agent_retries_then_succeeds(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=2,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        side_effects = [
            _make_failing_cm(aiohttp.ClientConnectionError()),
            _make_response_cm(),
        ]
        with mock.patch.object(
            transport._session,
            'post',
            mock.MagicMock(side_effect=side_effects),
        ) as mock_post:
            registration = await transport.register_agent(Agent)
            assert isinstance(registration.agent_id, AgentId)
            assert mock_post.call_count == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_status_retries_then_succeeds(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=2,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        side_effects = [
            _make_failing_cm(aiohttp.ClientConnectionError()),
            _make_response_cm(
                json_body={'status': MailboxStatus.ACTIVE.value},
            ),
        ]
        with mock.patch.object(
            transport._session,
            'get',
            mock.MagicMock(side_effect=side_effects),
        ) as mock_get:
            status = await transport.status(UserId.new())
            assert status == MailboxStatus.ACTIVE
            assert mock_get.call_count == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_terminate_retries_then_succeeds(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=2,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        side_effects = [
            _make_failing_cm(aiohttp.ClientConnectionError()),
            _make_response_cm(),
        ]
        with mock.patch.object(
            transport._session,
            'delete',
            mock.MagicMock(side_effect=side_effects),
        ) as mock_delete:
            await transport.terminate(UserId.new())
            assert mock_delete.call_count == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_heartbeat_status_retries_then_succeeds(
    http_exchange_server: tuple[str, int],
) -> None:
    host, port = http_exchange_server
    url = f'http://{host}:{port}'
    factory = HttpExchangeFactory(
        url,
        max_retries=2,
        retry_backoff_base_s=0,
    )
    async with await factory._create_transport() as transport:
        side_effects = [
            _make_failing_cm(aiohttp.ClientConnectionError()),
            _make_response_cm(json_body={'heartbeat': 1234.5}),
        ]
        with mock.patch.object(
            transport._session,
            'get',
            mock.MagicMock(side_effect=side_effects),
        ) as mock_get:
            heartbeat = await transport.heartbeat_status(UserId.new())
            assert heartbeat == 1234.5  # noqa: PLR2004
            assert mock_get.call_count == 2  # noqa: PLR2004

"""
Unit tests for Neo4jClient and settings normalization.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.config.settings import Settings
from app.infrastructure.neo4j.client import Neo4jClient


def test_settings_neo4j_credential_stripping():
    """Verify settings strip quotes and whitespace from Neo4j credentials."""
    s = Settings(
        neo4j_uri="  neo4j+s://b4b1d22a.databases.neo4j.io  ",
        neo4j_user=" 'b4b1d22a' ",
        neo4j_password=' "some_password" ',
        database_url="postgresql://user:pass@localhost:5432/db",
        weaviate_url="https://test.weaviate.network",
        weaviate_api_key="key",
    )
    assert s.neo4j_uri == "neo4j+s://b4b1d22a.databases.neo4j.io"
    assert s.neo4j_user == "b4b1d22a"
    assert s.neo4j_password == "some_password"


@pytest.mark.asyncio
async def test_neo4j_client_connect_retry_success():
    """Verify connect retries on transient failure then succeeds."""
    client = Neo4jClient(
        uri="neo4j+s://dummy.databases.neo4j.io",
        user="neo4j",
        password="secret_password",
    )

    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()
    # First call raises OSError (DNS glitch), second succeeds
    mock_driver.verify_connectivity = AsyncMock(
        side_effect=[OSError("getaddrinfo failed"), None]
    )

    with patch("app.infrastructure.neo4j.client.AsyncGraphDatabase.driver", return_value=mock_driver):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client.connect(retries=3, initial_delay=0.01)

            assert mock_driver.verify_connectivity.call_count == 2
            assert mock_sleep.call_count == 1
            assert client._driver is mock_driver

    await client.close()


@pytest.mark.asyncio
async def test_neo4j_client_connect_retry_exhaustion():
    """Verify connect raises exception after exhausting all retries."""
    client = Neo4jClient(
        uri="neo4j+s://dummy.databases.neo4j.io",
        user="neo4j",
        password="secret_password",
    )

    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()
    mock_driver.verify_connectivity = AsyncMock(
        side_effect=OSError("getaddrinfo failed")
    )

    with patch("app.infrastructure.neo4j.client.AsyncGraphDatabase.driver", return_value=mock_driver):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(OSError, match="getaddrinfo failed"):
                await client.connect(retries=3, initial_delay=0.01)

            assert mock_driver.verify_connectivity.call_count == 3

    await client.close()


@pytest.mark.asyncio
async def test_neo4j_client_health_check_success():
    """Verify health_check runs query and returns ok status."""
    client = Neo4jClient(
        uri="neo4j+s://dummy.databases.neo4j.io",
        user="neo4j",
        password="secret_password",
    )

    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()
    mock_driver.verify_connectivity = AsyncMock()

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value={"ping": 1})
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_driver.session = MagicMock(return_value=mock_session)

    with patch("app.infrastructure.neo4j.client.AsyncGraphDatabase.driver", return_value=mock_driver):
        await client.connect(retries=1)
        hc = await client.health_check()
        assert hc["status"] == "ok"
        assert "latency_ms" in hc

    await client.close()

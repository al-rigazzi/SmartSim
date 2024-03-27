import pathlib
import pytest

from smartsim._core.config.config import get_config
from smartsim._core.utils.security import KeyManager, KeyLocator


def test_key_manager_dir_preparation(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the KeyManager creates the appropriate directory
    structure required for public/private key pairs."""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg)

        km.create_directories()

        # verify the expected paths are created
        server_locator = KeyLocator(pathlib.Path(test_dir), "server")
        client_locator = KeyLocator(pathlib.Path(test_dir), "client")

        locators = [server_locator, client_locator]

        for locator in locators:
            assert locator.public_dir.exists()
            assert locator.private_dir.exists()


def test_key_manager_get_existing_keys_only(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager cannot load keys when
    directed not to create missing keys."""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg)

        # use no_create=True to only load pre-existing keys
        key_set = km.get_keys(create=False)

        assert key_set[0] is None
        assert key_set[1] is None


def test_key_manager_get_or_create_keys_default(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager creates keys when none can be loaded"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg)

        key_set = km.get_keys()

        # public keys are returned by default
        assert key_set[0].public is not None
        assert key_set[1].public is not None

        # default behavior will only return public keys
        assert not key_set[0].private
        assert not key_set[1].private


def test_key_manager_server_context(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager does not load private client
    keys when the context is set to server=True"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, as_server=True)

        server_keyset, client_keyset = km.get_keys()

        # as_server=True returns pub/priv server keys...
        assert len(server_keyset.public) > 0
        assert len(server_keyset.private) > 0

        # as_server=True returns only public client key
        assert len(client_keyset.public) > 0
        assert len(client_keyset.private) == 0


def test_key_manager_client_context(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager does not load private client
    keys when the context is set to client=True"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, as_client=True)

        server_keyset, client_keyset = km.get_keys()

        # as_client=True returns pub/priv client keys...
        assert len(server_keyset.public) > 0
        assert len(server_keyset.private) == 0

        # e=True returns only public server key
        assert len(client_keyset.public) > 0
        assert len(client_keyset.private) > 0

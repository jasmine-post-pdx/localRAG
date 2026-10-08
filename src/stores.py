"""Manages the local RAG system's data stores.

Run from the project root with ``python -m src.stores <command>``:

* ``list`` (the default): every store, its models, and what's been ingested.
* ``info [NAME]``: a store's settings, where each comes from, its paths, and
  whether its vectors match its settings.
* ``create NAME``: a new store whose ingest settings are pinned in its
  ``store.toml``, copied from ``stores/defaults.toml`` or another store.
* ``use NAME``: makes a store the default by saving ``RAG_STORE`` in ``.env``.

Attributes:
    STORE_NAME_PATTERN (re.Pattern): Allowed store names: letters, digits,
        ``-``, and ``_``, starting with a letter or digit.
"""
import argparse
import dataclasses
import os
import re
import shutil

import tomlkit
from dotenv import set_key
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import config
from src.console import console
from src.ingest import find_document_files
from src.manifest import find_mismatches, has_vectors, read_manifest

STORE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def ingest_status(store):
    """Describes what has been ingested into a store, in a few words."""
    manifest = read_manifest(store)
    if manifest is not None:
        return f"{manifest['chunk_count']} chunks, {manifest['updated_at'][:10]}"
    return "not recorded" if has_vectors(store) else "not ingested"


def create_store(
    name,
    description="",
    embedding_model=None,
    llm_model=None,
    copy_from=None,
    stores_dir=config.STORES_DIR,
):
    """Creates a new data store.

    Makes ``<stores_dir>/<name>/`` with a ``documents/`` folder and a
    ``store.toml`` that sets every ``[ingest]`` and ``[query]`` setting, so
    later changes to ``defaults.toml`` can't silently change the store. The
    settings come from ``copy_from`` if given, otherwise from
    ``defaults.toml``. A ``copy_from`` store's own ``prompt.txt`` is copied
    too; its documents, vectors, and paths are not.

    Args:
        name (str): Name of the new store.
        description (str): Short note about what the store will hold.
        embedding_model (str | None): Embedding model to use instead of the
            copied one.
        llm_model (str | None): Chat model to use instead of the copied one.
        copy_from (str | None): Existing store to copy settings from.
        stores_dir (Path): Folder holding the stores. Defaults to
            ``config.STORES_DIR``.

    Returns:
        StoreConfig: The new store.

    Raises:
        RuntimeError: If the name isn't allowed, the store already exists,
            or the settings to copy can't be loaded.
    """
    if not STORE_NAME_PATTERN.fullmatch(name):
        raise RuntimeError(
            f"{name!r} isn't a valid store name. Use letters, digits, - and _, "
            "starting with a letter or digit."
        )
    folder = stores_dir / name
    if folder.exists():
        raise RuntimeError(f"{folder} already exists.")

    folder.mkdir(parents=True)
    try:
        if copy_from:
            base = config.load_store(copy_from, stores_dir, shell_env={})
        else:
            (folder / "store.toml").write_text("", encoding="utf-8")
            base = config.load_store(name, stores_dir, shell_env={})
        ingest = dataclasses.asdict(base.ingest)
        query = dataclasses.asdict(base.query)
        if embedding_model:
            ingest["embedding_model"] = embedding_model
        if llm_model:
            query["llm_model"] = llm_model

        document = tomlkit.document()
        document.add(tomlkit.comment("store.toml"))
        document.add(tomlkit.comment(""))
        document.add(tomlkit.comment(
            f'Settings for the "{name}" data store. Anything left out is taken from'
        ))
        document.add(tomlkit.comment("stores/defaults.toml, which explains each setting."))
        document.add(tomlkit.nl())
        document.add("description", description)
        document.add(tomlkit.nl())
        ingest_table = tomlkit.table()
        ingest_table.add(tomlkit.comment(
            "Fixed once documents are ingested. Changing embedding_model, chunk_size,"
        ))
        ingest_table.add(tomlkit.comment(
            "or chunk_overlap later requires deleting the contents of vectorstore/."
        ))
        for key, value in ingest.items():
            ingest_table.add(key, value)
        document.add("ingest", ingest_table)
        query_table = tomlkit.table()
        query_table.add(tomlkit.comment("Safe to change at any time without re-ingesting."))
        for key, value in query.items():
            query_table.add(key, value)
        document.add("query", query_table)
        (folder / "store.toml").write_text(tomlkit.dumps(document), encoding="utf-8")

        (folder / "documents").mkdir()
        own_prompt = stores_dir / base.name / config.PROMPT_FILENAME
        if copy_from and own_prompt.is_file():
            shutil.copyfile(own_prompt, folder / config.PROMPT_FILENAME)
        return config.load_store(name, stores_dir, shell_env={})
    except Exception:
        shutil.rmtree(folder)
        raise


def use_store(name, env_path=config.ENV_PATH, stores_dir=config.STORES_DIR):
    """Makes a store the default by saving ``RAG_STORE`` in ``.env``.

    Args:
        name (str): The store to use by default.
        env_path (Path): The ``.env`` file to update, created if needed.
            Defaults to ``config.ENV_PATH``.
        stores_dir (Path): Folder holding the stores. Defaults to
            ``config.STORES_DIR``.

    Raises:
        RuntimeError: If the store doesn't exist.
    """
    available = config.list_stores(stores_dir)
    if name not in available:
        raise RuntimeError(
            f"Store {name!r} doesn't exist. Available stores: {', '.join(available) or 'none'}."
        )
    env_path.touch(exist_ok=True)
    set_key(env_path, "RAG_STORE", name, quote_mode="never")


def list_command(args):
    """Prints a table of every store."""
    app_settings = config.load_app_settings()
    table = Table(title="Data stores", title_justify="left")
    for column in ("Store", "Description", "Embedding model", "Chat model", "Ingested"):
        table.add_column(column)
    for name in config.list_stores():
        is_default = name == app_settings.default_store
        label = Text(f"{name} (default)" if is_default else name)
        try:
            store = config.load_store(name)
        except RuntimeError as error:
            table.add_row(label, Text(f"Invalid settings: {error}"), style="warning")
            continue
        table.add_row(
            label,
            Text(store.description),
            Text(store.ingest.embedding_model),
            Text(store.query.llm_model),
            Text(ingest_status(store)),
            style="success" if is_default else None,
        )
    console.print(table)
    console.print(
        "Use a store for one command with --store <name>, or make it the default with "
        "`python -m src.stores use <name>`.",
        style="info",
    )


def info_command(args):
    """Prints a store's settings, paths, and ingestion status."""
    app_settings = config.load_app_settings()
    store = config.load_store(args.name or app_settings.default_store)
    default_note = " (default)" if store.name == app_settings.default_store else ""
    console.rule(f"Data store: {store.name}{default_note}")
    if store.description:
        console.print(store.description)

    table = Table(title="Settings", title_justify="left")
    for column in ("Setting", "Value", "Set in"):
        table.add_column(column)
    for name, (section, key, _) in config.STORE_SETTINGS.items():
        dotted = f"{section}.{key}" if section else key
        source = store.sources[name]
        if source == config.SHELL:
            source = f"the shell ({name})"
        table.add_row(Text(dotted), Text(str(store.get(name))), Text(source))
    console.print(table)

    waiting = len(find_document_files(store.documents_path))
    console.print(f"Settings file:  {store.config_path}")
    console.print(f"Prompt:         {store.prompt_path}")
    console.print(f"Documents:      {store.documents_path} ({waiting} file(s) waiting to ingest)")
    console.print(f"Vector store:   {store.vectorstore_path} ({ingest_status(store)})")

    manifest = read_manifest(store)
    mismatches = find_mismatches(store, manifest) if manifest else []
    if mismatches:
        console.print(Panel(
            "\n".join(mismatches),
            title="Settings changed since this store was built", border_style="yellow",
        ))


def create_command(args):
    """Creates a store from command-line arguments and says what to do next."""
    store = create_store(
        args.name,
        description=args.description,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        copy_from=args.copy_from,
    )
    console.print(f"Created {store.name} in {store.root}", style="success")
    console.print(
        f"Next: add files to {store.documents_path}, then run "
        f"`python -m src.ingest --store {store.name}`. To pick its models "
        f"interactively first, run `python -m src.util --store {store.name}`.",
        style="info",
    )


def use_command(args):
    """Makes a store the default and warns if the shell would override it."""
    use_store(args.name)
    console.print(f"Saved RAG_STORE={args.name} to {config.ENV_PATH.name}", style="success")
    shell_value = os.environ.get("RAG_STORE", "").strip()
    if shell_value and shell_value != args.name:
        console.print(
            f"RAG_STORE={shell_value} is also exported in your shell, which takes "
            "precedence over .env. Run `unset RAG_STORE` for the new default to take effect.",
            style="warning",
        )


def main(argv=None):
    """Runs a store command from the command line."""
    parser = argparse.ArgumentParser(description="Manage data stores.")
    parser.set_defaults(func=list_command)
    commands = parser.add_subparsers(title="commands")

    commands.add_parser("list", help="list every store (the default)").set_defaults(
        func=list_command,
    )

    info = commands.add_parser("info", help="show a store's settings and status")
    info.add_argument("name", nargs="?", help="store to show (default: RAG_STORE)")
    info.set_defaults(func=info_command)

    create = commands.add_parser("create", help="create a new store")
    create.add_argument("name", help="name of the new store")
    create.add_argument("--description", default="", help="what the store will hold")
    create.add_argument("--embedding-model", help="embedding model (default: from defaults.toml)")
    create.add_argument("--llm-model", help="chat model (default: from defaults.toml)")
    create.add_argument(
        "--copy-from", metavar="STORE", help="copy settings and prompt from another store",
    )
    create.set_defaults(func=create_command)

    use = commands.add_parser("use", help="make a store the default (saved in .env)")
    use.add_argument("name", help="store to use by default")
    use.set_defaults(func=use_command)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RuntimeError as error:
        console.print(Panel(str(error), title="Error", border_style="red"))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

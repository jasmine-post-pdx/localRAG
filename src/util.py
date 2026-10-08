"""Maintenance tasks for the local RAG system.

Currently lets the user pick the embedding and chat models for a data store
from the models installed in Ollama, saving any change to the store's
``store.toml``.

Run from the project root with ``python -m src.util``, adding
``--store <name>`` to change a store other than the default.
"""
import argparse

import tomlkit
from ollama import Client
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import config
from src.console import console


def _with_tag(name):
    """Returns a model name with Ollama's implicit ``:latest`` tag made explicit."""
    return name if ":" in name else f"{name}:latest"


def _setting_key(name):
    """Returns a setting's dotted key in ``store.toml``, e.g. ``"query.llm_model"``."""
    table, key, _ = config.STORE_SETTINGS[name]
    return f"{table}.{key}" if table else key


def save_store_setting(name, value, path):
    """Writes a per-store setting to a store's ``store.toml``.

    Comments and layout in the file are kept. The setting's table (e.g.
    ``[query]``) is added at the end of the file if it doesn't exist yet.

    Args:
        name (str): Environment name of the setting, e.g. ``"LLM_MODEL"``.
        value (str | int | float): The new value.
        path (Path): The ``store.toml`` to update, typically a store's
            ``config_path``.
    """
    table, key, _ = config.STORE_SETTINGS[name]
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    target = document
    if table:
        if table not in document:
            document.add(table, tomlkit.table())
        target = document[table]
    target[key] = value
    path.write_text(tomlkit.dumps(document), encoding="utf-8")


def list_models(client, capability):
    """Lists installed Ollama models that have a given capability.

    Args:
        client (Client): Ollama client.
        capability (str): Capability Ollama reports for the model, such as
            ``"embedding"`` or ``"completion"`` (chat). Models from Ollama
            versions that don't report capabilities are always included.

    Returns:
        list[ListResponse.Model]: The matching models, sorted by name.

    Raises:
        ConnectionError: If the Ollama server can't be reached.
    """
    models = []
    for model in client.list().models:
        capabilities = client.show(model.model).capabilities
        if capabilities is None or capability in capabilities:
            models.append(model)
    return sorted(models, key=lambda m: m.model)


def print_models(title, models, current):
    """Prints models as a numbered table, highlighting the current one."""
    table = Table(title=title, title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Parameters", justify="right")
    table.add_column("Family")
    table.add_column("Size on disk", justify="right")
    for number, model in enumerate(models, start=1):
        is_current = _with_tag(model.model) == _with_tag(current)
        name = f"{model.model} (current)" if is_current else model.model
        table.add_row(
            Text(str(number)),
            Text(name),
            Text(model.details.parameter_size or ""),
            Text(model.details.family or ""),
            Text(f"{model.size / 1e9:.1f} GB"),
            style="success" if is_current else None,
        )
    console.print(table)


def choose_model(label, models, current):
    """Asks the user to pick a model by number or name.

    Keeps asking until the answer matches a row in the table. An empty answer,
    or end of input, keeps ``current``.

    Args:
        label (str): What the model is for, used in the prompt.
        models (list[ListResponse.Model]): The models shown in the table.
        current (str): The model in use now.

    Returns:
        str: The chosen model name, or ``current`` if the user kept it.
    """
    prompt = Text(
        f"\nChoose a new {label} by number or name (press Enter to keep {current}): ",
        style="question",
    )
    while True:
        try:
            answer = console.input(prompt).strip()
        except EOFError:
            return current
        if not answer:
            return current
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1].model
        for model in models:
            if _with_tag(answer) == _with_tag(model.model):
                return model.model
        console.print(
            f"{answer!r} isn't in the table. Enter a number from 1 to {len(models)}, "
            "a model name, or press Enter.",
            style="warning",
        )


def update_model_setting(client, store, name, label, capability):
    """Shows a store's current model for a setting, lets the user change it, and saves it.

    Lists the installed models that have ``capability``, asks for a choice
    with ``choose_model``, and writes a changed value to the store's
    ``store.toml``. ``stores/defaults.toml`` is never modified.

    Args:
        client (Client): Ollama client.
        store (StoreConfig): The store whose model to change.
        name (str): Setting to update, e.g. ``"EMBEDDING_MODEL"``.
        label (str): What the model is for, e.g. ``"embedding model"``.
        capability (str): Capability the listed models must have.

    Returns:
        bool: Whether the setting was changed.

    Raises:
        ConnectionError: If the Ollama server can't be reached.
    """
    current = store.get(name)
    source = store.sources[name]
    console.rule(label.capitalize())
    console.print(f"Current {label}: {current} (set in {source})", style="info")

    models = list_models(client, capability)
    if not models:
        console.print(Panel(
            f"None of the installed Ollama models is a {label}.\n"
            f"Install one with `ollama pull <model>`, then run this again.",
            title="Nothing to choose from", border_style="yellow",
        ))
        return False

    print_models(f"Installed {label}s", models, current)
    if not any(_with_tag(m.model) == _with_tag(current) for m in models):
        console.print(
            f"The current {label} isn't installed. Run `ollama pull {current}` "
            "or choose another one.",
            style="warning",
        )

    chosen = choose_model(label, models, current)
    if _with_tag(chosen) == _with_tag(current):
        console.print(f"Keeping {current}.", style="info")
        return False

    save_store_setting(name, chosen, store.config_path)
    console.print(
        f"Saved {_setting_key(name)} = {chosen} to {store.config_path}",
        style="success",
    )
    if source == config.SHELL:
        console.print(
            f"{name} is also exported in your shell, which takes precedence over "
            f"store.toml. Run `unset {name}` for the new value to take effect.",
            style="warning",
        )
    return True


def main(argv=None):
    """Lets the user change a store's embedding model, then its chat model.

    Changes the store named by ``--store``, or ``RAG_STORE``.
    """
    parser = argparse.ArgumentParser(description="Pick a data store's embedding and chat models.")
    parser.add_argument("--store", help="data store to change (default: RAG_STORE)")
    args = parser.parse_args(argv)

    app_settings = config.load_app_settings()
    store = config.load_store(args.store or app_settings.default_store)
    console.print(f"Data store: {store.name} ({store.config_path})", style="info")
    client = Client(host=app_settings.ollama_host)
    try:
        embedding_changed = update_model_setting(
            client, store, "EMBEDDING_MODEL", "embedding model", "embedding",
        )
        has_vectors = (store.vectorstore_path / "chroma.sqlite3").exists()
        if embedding_changed and has_vectors:
            console.print(Panel(
                "Vectors from different embedding models can't be compared, so this "
                "store's existing vector store won't work with the new model.\n"
                f"Delete the contents of {store.vectorstore_path}, then run "
                f"`python -m src.ingest --store {store.name}`. To keep this store's "
                "data instead, change the model back and create a new store for the "
                "new model.",
                title="Rebuild the vector store", border_style="yellow",
            ))
        update_model_setting(client, store, "LLM_MODEL", "chat model", "completion")
    except ConnectionError:
        console.print(Panel(
            f"Can't reach Ollama at {app_settings.ollama_host}.\n"
            "Start it with `ollama serve` (or open the Ollama app) and run this again.",
            title="Error", border_style="red",
        ))
        raise SystemExit(1) from None
    console.rule("Done")


if __name__ == "__main__":
    main()

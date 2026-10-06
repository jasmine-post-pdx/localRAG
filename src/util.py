"""Maintenance tasks for the local RAG system.

Currently lets the user pick the embedding and chat models from the models
installed in Ollama, saving any change to ``.env``.

Run from the project root with ``python -m src.util``.
"""
from dotenv import dotenv_values, set_key
from ollama import Client
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import config
from src.console import console


def _with_tag(name):
    """Returns a model name with Ollama's implicit ``:latest`` tag made explicit."""
    return name if ":" in name else f"{name}:latest"


def setting_source(name, value):
    """Reports where the current value of a setting came from.

    Args:
        name (str): Environment variable name, e.g. ``"LLM_MODEL"``.
        value (str): The value ``config`` loaded for it.

    Returns:
        str: ``".env"``, ``".env.default"``, or ``"shell"`` if neither file
        holds that value, meaning an exported shell variable set it.
    """
    if dotenv_values(config.ENV_PATH).get(name) == value:
        return ".env"
    if dotenv_values(config.DEFAULT_ENV_PATH).get(name) == value:
        return ".env.default"
    return "shell"


def list_models(client, capability):
    """Lists installed Ollama models that have a given capability.

    Args:
        client (Client): Ollama client connected to ``config.OLLAMA_HOST``.
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


def update_model_setting(client, name, label, capability):
    """Shows the current model for a setting, lets the user change it, and saves it.

    Lists the installed models that have ``capability``, asks for a choice
    with ``choose_model``, and writes a changed value to ``.env``, creating
    the file if needed. ``.env.default`` is never modified.

    Args:
        client (Client): Ollama client connected to ``config.OLLAMA_HOST``.
        name (str): Setting to update, e.g. ``"EMBEDDING_MODEL"``.
        label (str): What the model is for, e.g. ``"embedding model"``.
        capability (str): Capability the listed models must have.

    Returns:
        bool: Whether the setting was changed.

    Raises:
        ConnectionError: If the Ollama server can't be reached.
    """
    current = getattr(config, name)
    source = setting_source(name, current)
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

    config.ENV_PATH.touch(exist_ok=True)
    set_key(config.ENV_PATH, name, chosen, quote_mode="never")
    console.print(f"Saved {name}={chosen} to {config.ENV_PATH.name}", style="success")
    if source == "shell":
        console.print(
            f"{name} is also exported in your shell, which takes precedence over "
            f".env. Run `unset {name}` for the new value to take effect.",
            style="warning",
        )
    return True


def main():
    """Lets the user change the embedding model, then the chat model."""
    client = Client(host=config.OLLAMA_HOST)
    try:
        embedding_changed = update_model_setting(
            client, "EMBEDDING_MODEL", "embedding model", "embedding",
        )
        if embedding_changed:
            console.print(Panel(
                "Vectors from different embedding models can't be compared, so the "
                "existing vector store won't work with the new model.\n"
                f"Delete the contents of {config.VECTORSTORE_PATH} (or set a new "
                "VECTORSTORE_PATH in .env), then run `python -m src.ingest`.",
                title="Rebuild the vector store", border_style="yellow",
            ))
        update_model_setting(client, "LLM_MODEL", "chat model", "completion")
    except ConnectionError:
        console.print(Panel(
            f"Can't reach Ollama at {config.OLLAMA_HOST}.\n"
            "Start it with `ollama serve` (or open the Ollama app) and run this again.",
            title="Error", border_style="red",
        ))
        raise SystemExit(1) from None
    console.rule("Done")


if __name__ == "__main__":
    main()

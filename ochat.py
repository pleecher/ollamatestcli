#!/usr/bin/env python3
import ollama
import sys
import re
import time
import datetime
import difflib


class Color:
    GREEN  = "\033[92m"   # User input
    PURPLE = "\033[95m"   # Model thinking
    BLUE   = "\033[94m"   # Model response
    YELLOW = "\033[93m"   # System messages
    RED    = "\033[91m"   # Errors
    CYAN   = "\033[96m"   # Info / headers
    DIM    = "\033[2m"    # Dimmed text
    BOLD   = "\033[1m"    # Bold
    RESET  = "\033[0m"    # Reset

def cprint(text, color=Color.RESET, end="\n"):
    """Print colored text."""
    print(f"{color}{text}{Color.RESET}", end=end, flush=True)

def connect2server(host: str) -> ollama.Client:
    """Create client and verify connection."""
    if not host.startswith("http"):
        host = f"http://{host}"
    client = ollama.Client(host=host)
    try:
        client.list()
        return client
    except Exception as e:
        cprint(f"Error: Could not connect to Ollama at {host}", Color.RED)
        cprint(f"  {e}", Color.DIM)
        sys.exit(1)

def listModels(client: ollama.Client) -> list[str]:
    """Retrieve available model names."""
    response = client.list()
    models = []
    for m in response.get("models", []):
        name = m.get("name", m.get("model", ""))
        if name:
            models.append(name)
    return sorted(models)


def selectModelX(query: str, models: list[str]) -> str | None:
    if not models:
        return None

    query_lower = query.lower().strip()

    for m in models:
        if m.lower() == query_lower:
            return m

    starts = [m for m in models if m.lower().startswith(query_lower)]
    if len(starts) == 1:
        return starts[0]

    try:
        pattern = re.compile(query_lower, re.IGNORECASE)
        regex_matches = [m for m in models if pattern.search(m)]
        if len(regex_matches) == 1:
            return regex_matches[0]
        if regex_matches:
            return min(regex_matches, key=len)
    except re.error:
        pass

    def subsequence_match(query_str, target):
        qi = 0
        for ch in target.lower():
            if qi < len(query_str) and ch == query_str[qi]:
                qi += 1
        return qi == len(query_str)

    subseq = [m for m in models if subsequence_match(query_lower.replace("-", ""), m.lower().replace("-", ""))]
    if len(subseq) == 1:
        return subseq[0]
    if subseq:
        return min(subseq, key=len)

    close = difflib.get_close_matches(query_lower, [m.lower() for m in models], n=1, cutoff=0.4)
    if close:
        idx = [m.lower() for m in models].index(close[0])
        return models[idx]
    return None


def selectModel(client: ollama.Client, models: list[str], prompt_text: str = "Select model") -> str:
    """Interactive model selection with fuzzy matching."""
    cprint(f"\n{'─' * 50}", Color.DIM)
    cprint("Available models:", Color.CYAN)
    for i, m in enumerate(models, 1):
        cprint(f"  {i:3d}. {m}", Color.CYAN)
    cprint(f"{'─' * 50}", Color.DIM)

    while True:
        cprint(f"\n{prompt_text} (name, number, or pattern): ", Color.YELLOW, end="")
        user_input = input().strip()
        if not user_input:
            continue

        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                cprint(f"  Number out of range (1-{len(models)})", Color.RED)
                continue
        except ValueError:
            pass

        match = selectModelX(user_input, models)
        if match:
            cprint(f"  → Matched: {match}", Color.YELLOW)
            return match
        else:
            cprint(f"  No match found for '{user_input}'. Try again.", Color.RED)

def loadModelX(client: ollama.Client, model_name: str):
    """
    Trigger model load by sending a minimal generate request,
    and show a loading indicator while the model loads.
    We use a streaming generate with an empty prompt and num_predict=0
    to just trigger loading without generating tokens.
    """
    cprint(f"\nLoading model '{model_name}'...", Color.YELLOW)

    try:
        # Use pull to check if model exists / download if needed
        # Then use generate with num_predict=0 to load into memory
        start_time = time.time()
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        spin_idx = 0

        # First, stream a pull to show download progress (if not already downloaded)
        try:
            for chunk in client.pull(model_name, stream=True):
                status = chunk.get("status", "")
                total = chunk.get("total", 0)
                completed = chunk.get("completed", 0)

                if total > 0 and completed > 0:
                    pct = (completed / total) * 100
                    bar_len = 30
                    filled = int(bar_len * completed // total)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r{Color.YELLOW}  {status}: [{bar}] {pct:5.1f}%{Color.RESET}", end="", flush=True)
                elif status:
                    print(f"\r{Color.YELLOW}  {spinner[spin_idx]} {status:<50}{Color.RESET}", end="", flush=True)
                    spin_idx = (spin_idx + 1) % len(spinner)
            print()  # newline after pull
        except Exception:
            pass  # Model might already be available

        # Now trigger actual model load into memory
        cprint("  Loading into memory...", Color.YELLOW, end="")
        loading_done = False

        # Use a generate call with num_predict=1 to force loading
        import threading

        def do_load():
            nonlocal loading_done
            try:
                client.generate(model=model_name, prompt="Hi", options={"num_predict": 1})
            except Exception:
                pass
            loading_done = True

        t = threading.Thread(target=do_load, daemon=True)
        t.start()

        while not loading_done:
            elapsed = time.time() - start_time
            print(f"\r{Color.YELLOW}  {spinner[spin_idx]} Loading into memory... ({elapsed:.0f}s){Color.RESET}", end="", flush=True)
            spin_idx = (spin_idx + 1) % len(spinner)
            time.sleep(0.5)

        elapsed = time.time() - start_time
        print(f"\r{Color.YELLOW}  ✓ Model loaded in {elapsed:.1f}s{' ' * 20}{Color.RESET}")

    except ollama.ResponseError as e:
        cprint(f"\n  Error loading model: {e}", Color.RED)
        sys.exit(1)



def format4export(history: list[dict], model_name: str) -> str:
    """Format conversation history for /write export."""
    lines = []
    lines.append(f"# Ollama Chat Export")
    lines.append(f"# Model: {model_name}")
    lines.append(f"# Date:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# {'=' * 50}")
    lines.append("")

    for msg in history:
        role = msg["role"]
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")

        if role == "user":
            lines.append(f"user: {content}")
        elif role == "assistant":
            if thinking:
                lines.append(f"assist(think): {thinking}")
            lines.append(f"assist: {content}")
        elif role == "system":
            lines.append(f"system: {content}")
        lines.append("")

    return "\n".join(lines)


def showHelp():
    """Display available commands."""
    cprint("\n  Available Commands:", Color.CYAN)
    cprint("  ─────────────────────────────────────", Color.DIM)
    cprint("  /model  - change model", Color.CYAN)
    cprint("  /write  - write to a  file", Color.CYAN)
    cprint("  /bench  - show token/s of last response", Color.CYAN)
    cprint("  /help   - this message", Color.CYAN)
    cprint("  /quit   - bye bye...", Color.CYAN)
    cprint("  ─────────────────────────────────────\n", Color.DIM)


def chatter(client: ollama.Client, model_name: str, models: list[str]):
    """Main chat loop with streaming, thinking support, and commands."""
    history: list[dict] = []
    last_bench: dict | None = None  # stores last response metrics

    cprint(f"\n{'═' * 50}", Color.DIM)
    cprint(f"  Chat with: {model_name}", Color.BOLD + Color.CYAN)
    cprint(f"  Type /help for commands, /quit to exit", Color.DIM)
    cprint(f"{'═' * 50}\n", Color.DIM)

    while True:
        try:
            cprint("You: ", Color.GREEN, end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            cprint("\n\nGoodbye!", Color.YELLOW)
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.lower() == "/quit" or stripped.lower() == "/exit":
            cprint("Goodbye!", Color.YELLOW)
            break

        if stripped.lower() == "/help":
            showHelp()
            continue

        if stripped.lower() == "/model":
            models = listModels(client)
            if not models:
                cprint("  No models available.", Color.RED)
                continue
            new_model = selectModel(client, models, "Switch to model")
            if new_model != model_name:
                model_name = new_model
                loadModelX(client, model_name)
                cprint(f"\n  Switched to: {model_name}\n", Color.YELLOW)
            continue

        if stripped.lower() == "/write":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{timestamp}.txt"
            content = format4export(history, model_name)
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                cprint(f"  Conversation saved to: {filename}", Color.YELLOW)
            except IOError as e:
                cprint(f"  Error writing file: {e}", Color.RED)
            continue

        if stripped.lower() == "/bench":
            if last_bench is None:
                cprint("  No response data yet. Send a message first.", Color.YELLOW)
            else:
                eval_count = last_bench.get("eval_count", 0)
                eval_duration_ns = last_bench.get("eval_duration", 0)
                prompt_eval_count = last_bench.get("prompt_eval_count", 0)
                prompt_eval_duration_ns = last_bench.get("prompt_eval_duration", 0)
                total_duration_ns = last_bench.get("total_duration", 0)

                cprint("  ┌─────────────────────────────────────┐", Color.CYAN)
                if eval_duration_ns > 0:
                    tok_per_sec = eval_count / (eval_duration_ns / 1e9)
                    cprint(f"  │ Generation:  {tok_per_sec:8.2f} token/s      │", Color.CYAN)
                    cprint(f"  │ Tokens out:  {eval_count:8d}              │", Color.CYAN)
                    eval_sec = eval_duration_ns / 1e9
                    cprint(f"  │ Gen time:    {eval_sec:8.2f}s             │", Color.CYAN)
                if prompt_eval_duration_ns > 0:
                    prompt_tok_s = prompt_eval_count / (prompt_eval_duration_ns / 1e9)
                    cprint(f"  │ Prompt eval: {prompt_tok_s:8.2f} token/s      │", Color.CYAN)
                    cprint(f"  │ Tokens in:   {prompt_eval_count:8d}              │", Color.CYAN)
                if total_duration_ns > 0:
                    total_sec = total_duration_ns / 1e9
                    cprint(f"  │ Total time:  {total_sec:8.2f}s             │", Color.CYAN)
                cprint("  └─────────────────────────────────────┘", Color.CYAN)
            continue

        history.append({"role": "user", "content": stripped})

        response_content = []
        thinking_content = []
        in_thinking = False
        first_content_token = True
        first_thinking_token = True

        try:
            stream = client.chat(
                model=model_name,
                messages=history,
                stream=True,
                think=True,
            )

            for chunk in stream:
                msg = chunk.get("message", {})
                thinking = msg.get("thinking", "")
                content = msg.get("content", "")

                if thinking:
                    if first_thinking_token:
                        cprint("\n💭 ", Color.PURPLE, end="")
                        first_thinking_token = False
                        in_thinking = True
                    print(f"{Color.PURPLE}{thinking}{Color.RESET}", end="", flush=True)
                    thinking_content.append(thinking)

                if content:
                    if in_thinking:
                        # Transition from thinking to content
                        print()  # newline after thinking
                        in_thinking = False
                    if first_content_token:
                        cprint("\n🤖 ", Color.BLUE, end="")
                        first_content_token = False
                    print(f"{Color.BLUE}{content}{Color.RESET}", end="", flush=True)
                    response_content.append(content)

                if chunk.get("done", False):
                    last_bench = {
                        "eval_count": chunk.get("eval_count", 0),
                        "eval_duration": chunk.get("eval_duration", 0),
                        "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                        "prompt_eval_duration": chunk.get("prompt_eval_duration", 0),
                        "total_duration": chunk.get("total_duration", 0),
                    }

            print("\n") 

            full_content = "".join(response_content)
            full_thinking = "".join(thinking_content)
            entry = {"role": "assistant", "content": full_content}
            if full_thinking:
                entry["thinking"] = full_thinking
            history.append(entry)

        except ollama.ResponseError as e:
            cprint(f"\n  Error: {e}\n", Color.RED)
            if history and history[-1]["role"] == "user":
                history.pop()
        except KeyboardInterrupt:
            cprint("\n  (Response interrupted)\n", Color.YELLOW)
            partial = "".join(response_content)
            if partial:
                entry = {"role": "assistant", "content": partial}
                partial_think = "".join(thinking_content)
                if partial_think:
                    entry["thinking"] = partial_think
                history.append(entry)
#######################
def main():
    host = "127.0.0.1:11434"
    if len(sys.argv) > 1:
        host = sys.argv[1]
    cprint("  Ollama Chat", Color.BOLD + Color.CYAN)
    cprint(f"  Connecting to: {host}", Color.YELLOW)

    client = connect2server(host)
    cprint("  ✓ Connected!", Color.YELLOW)

    models = listModels(client)
    if not models:
        cprint("\n  No models ...", Color.RED)
        sys.exit(1)

    modelName = selectModel(client, models)

    loadModelX(client, modelName)
    chatter(client, modelName, models)  # L00P


if __name__ == "__main__":
    main()

from datetime import datetime
import shlex

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.widget import Widget
from textual.widgets import RichLog, Static, Input, TabbedContent, TabPane, Collapsible, \
    Header

from extensions.cli.src.cli import CLI_Connector
from core.utils.callbacks import callback_definition, CallbackContainer

# ======================================================================================================================
log_function = None


def log(message):
    if log_function is not None and callable(log_function):
        log_function(message)


# ======================================================================================================================
# OVERVIEW WIDGET
# ======================================================================================================================
class OverviewLeft(Widget):
    def compose(self) -> ComposeResult:
        with VerticalScroll(can_focus=False):
            for i in range(0, 5):
                with Collapsible(title=f"Collapsible #{i}"):
                    for j in range(0, 10):
                        yield Static(f"Overview Left {i}:{j}", classes="test-class")


class OverviewWidget(Widget):
    logging_function = None
    log = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="overview-upper"):
                yield OverviewLeft(id="overview-left")
                yield RichLog(wrap=True, highlight=False, markup=True, id="overview-log")
            yield Static("INPUT", id='overview-input', expand=True)

    def on_mount(self):
        self.log = self.query_one("#overview-log", RichLog)
        self.log.can_focus = False


# ======================================================================================================================
# LOG TAB WIDGET
# ======================================================================================================================
class CommandInput(Input):
    """Custom Input widget with command history and dynamic suggestions."""
    commands: list
    command_history = []
    history_index = -1
    suggestions_box: Static = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = []  # not used for suggestions now
        self.suggestions_box = None

    def setCommands(self, commands: list):
        self.commands = commands

    def on_mount(self) -> None:
        """Initialize suggestions on mount."""
        self.update_dynamic_suggestions("")

    def update_dynamic_suggestions(self, value: str):
        """Ask the parent App for a suggestion string based on the given value."""
        if self.suggestions_box and hasattr(self.app, "get_suggestions_for_input"):
            suggestions = self.app.get_suggestions_for_input(value)
            self.suggestions_box.update(suggestions)

    def on_key(self, event: events.Key) -> None:
        # Optional history navigation:
        if event.key == "up":
            if self.command_history and self.history_index > 0:
                self.history_index -= 1
                self.value = self.command_history[self.history_index]
                self.cursor_position = len(self.value)
                self.update_dynamic_suggestions(self.value)
        elif event.key == "down":
            if self.command_history and self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.value = self.command_history[self.history_index]
                self.cursor_position = len(self.value)
                self.update_dynamic_suggestions(self.value)
            else:
                self.history_index = len(self.command_history)
                self.value = ""
                self.update_dynamic_suggestions(self.value)
        elif event.key == "enter":
            if self.value.strip():
                self.command_history.append(self.value)
                self.history_index = len(self.command_history)
            # Submission is handled elsewhere.
        # For keys that change the input, update suggestions.
        elif event.key in ("backspace", "space") or (len(event.key) == 1 and event.key.isalnum()) or event.key == "-":
            # We do not recalc the entire value; instead use self.value plus the key if applicable.
            if event.key == "backspace":
                new_val = self.value[:-1]
            elif event.key == "space":
                new_val = self.value + " "
            else:
                new_val = self.value + event.key
            self.update_dynamic_suggestions(new_val)
        else:
            self.update_dynamic_suggestions(self.value)
        # super().on_key(event)


class LogTabWidget(Widget):
    logging_function = None
    log = None

    input: CommandInput
    suggestions_box: Static

    def compose(self) -> ComposeResult:
        with Vertical():
            yield RichLog(highlight=False, markup=True, id="log-log")
            yield Static("here comes the path", id="log-path-box", expand=False)
            yield Static("here come the suggestions", id="log-suggestions-box", expand=False)
            yield CommandInput(placeholder="Enter command...", id="log-input")

    def on_mount(self):
        self.log = self.query_one("#log-log", RichLog)
        self.log.can_focus = False
        self.input = self.query_one("#log-input", CommandInput)
        self.suggestions_box = self.query_one("#log-suggestions-box", Static)
        self.input.suggestions_box = self.suggestions_box
        self.input.on_mount()


# ======================================================================================================================
@callback_definition
class CLI_GUI_App_Callbacks:
    command: CallbackContainer


class CLI_GUI_App(App):
    CSS = """

    Screen {
        background: transparent;
    }

    Toast {
        # background: red;
        # opacity: 0.5;
    }
    #overview-widget {
        height: 100%;
        width: 100%;
    }
    #overview-upper {
        height: 80%;
    }
    #overview-left {
        height: 100%;
        width: 50%;
        border: round #8e8e8e;
        padding: 0 1;
        text-align: center;
        content-align: center middle;
    }
    #overview-log {
        height: 100%;
        width: 50%;
        border: round #8e8e8e;
        padding: 0 1;
        color: rgb(200,200,200);
    }
    #overview-input {
        height: 20%;
        width: 100%;
        border: round #8e8e8e;
        text-align: center;
        content-align: center middle;
    }

    #log-widget {
        background: transparent;
        height: 100%;
        width: 100%;
    }
    #log-log {
        background: transparent;
        border: round #8e8e8e;
        color: rgb(200,200,200);
        # color: red;
    }
    #log-path-box {
        height: 8%;
        min-height: 3;
        border: round #8e8e8e;
        padding: 0 3;
        content-align: left middle;
    }
    #log-suggestions-box {
        height: 12%;
        min-height: 3;
        border: round #8e8e8e;
        padding: 0 3;
        content-align: left middle;
    }
    #log-input {
        height: 10%;
        min-height: 3;
        content-align: center middle;
        padding: 0 3;
        background: rgb(40,40,40);
    }

    TabbedContent {
        height: 100%;
        background: transparent;
    }
    RichLog {
        border: round red;
        background: transparent;
        # color: white;
    }
    Tabs {
        background: transparent;
        color: white;
    }

    #robots_widget {
        height: 80%;
    }
    #robots_log {
        height: 20%;
        border: round white;
        padding: 0;
    }
    """

    callbacks = CLI_GUI_App_Callbacks()
    robots: dict[str, dict]
    logs: list[RichLog]
    cli_connector: CLI_Connector

    def __init__(self):
        super().__init__()
        # Assume the CLI-Connector already has a serialized command tree.
        self.cli_connector = CLI_Connector()
        self.logs = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with TabbedContent():
                # with TabPane("Overview", id="overview-tab"):
                #     with Vertical():
                #         yield OverviewWidget(id="overview-widget")
                with TabPane("Log", id="log-tab"):
                    with Vertical():
                        yield LogTabWidget(id="log-widget")
                # with TabPane("Robots", id="robots-tab"):
                #     yield Static("Robots", expand=True)
                # with TabPane("Devices", id="devices-tab"):
                #     yield Static("Devices", expand=True)
                # with TabPane("Optitrack", id="optitrack-tab"):
                #     yield Static("Optitrack", expand=True)
                # with TabPane("Experiments", id="experiments-tab"):
                #     yield Static("Experiments", expand=True)
                # with TabPane("Help", id="help-tab"):
                #     yield Static("Help", expand=True)

    def on_mount(self):
        self.title = "Robot App"

    def on_ready(self) -> None:
        """Called when the DOM is ready."""
        global log_function
        log_function = self.addLog

        # self.notify("Info", timeout=5)
        # self.notify("Warning", timeout=7, severity='warning')
        # self.notify("Error", timeout=9, severity='error')

        # overview_widget = self.query_one('#overview-widget', OverviewWidget)
        # self.logs.append(overview_widget.log)

        log_widget = self.query_one("#log-widget", LogTabWidget)
        self.logs.append(log_widget.log)

        log("Start GUI")
        self.update_log_widgets()

    def on_input_submitted(self, event: Input.Submitted):
        """Handle user input on submission."""
        user_input = event.value.strip()
        event.input.value = ""
        if user_input:
            # Special "clear" command: clear all logs immediately.
            if user_input.lower() == "clear":
                for richlog in self.logs:
                    richlog.clear()
                self.update_log_widgets()
                return

            self.addLog(f">> {user_input}", color="cyan")

            parsed_command = self.cli_connector.parseCommand(user_input)
            if not parsed_command['success']:
                self.addLog(parsed_command['error'], color='red')
            # Additional command processing can be added here.

            if parsed_command['success'] and parsed_command['type'] == 'command':
                self.callbacks.command.call(parsed_command['command'])

        self.update_log_widgets()

    def setCommands(self, commands: dict[str, dict]):
        self.cli_connector.setCommandSets(commands)
        if commands is None:
            self.cli_connector.active_set = [self.cli_connector.root_set]
        self.update_log_widgets()

    def setRobotData(self, data: dict):
        # Update the robots dict and refresh the related widget.
        pass

    def addLog(self, message, color=None, bold=False):
        if color is None:
            color = "rgb(200,200,200)"

        timestamp = datetime.now().strftime("%H:%M:%S")
        style = f"[bold {color}]" if bold else f"[{color}]"
        timestamp = f"{style}[{timestamp}][/]"
        formatted_message = f"{timestamp} {style} {message}"
        for richlog in self.logs:
            richlog.write(formatted_message)

    # -------------------- Helper methods for dynamic CLI suggestions --------------------
    def get_current_command_set(self) -> dict:
        """
        Navigate the serialized command tree using the CLI connector’s active_set.
        """
        desc = self.cli_connector.command_sets
        if not desc:
            return {}
        current = desc
        for part in self.cli_connector.active_set[1:]:
            if "child_sets" in current and part in current["child_sets"]:
                current = current["child_sets"][part]
            else:
                break
        return current

    def get_command_set_by_path(self, path: list) -> dict:
        """
        Return the command set at the given path in the serialized tree.
        """
        desc = self.cli_connector.command_sets
        if not desc:
            return {}
        current = desc
        for part in path[1:]:
            if "child_sets" in current and part in current["child_sets"]:
                current = current["child_sets"][part]
            else:
                break
        return current

    def get_suggestions_for_input(self, input_text: str) -> str:
        """
        Build a single-row suggestions string (using markup) based on the current active command set
        and the current input text.

        Special tokens:
         - If the first token is ".." (and there are more tokens) the parent's command set is used.
         - If the first token is "." (and there are more tokens) the root command set is used.

        Behavior changes:
         1. When entering a subset (or command) partially, show matching candidates (filtering the current possibilities).
         2. When a valid command has been fully entered (or when arguments have begun via a trailing space), show the input (argument) suggestions.
        """
        try:
            tokens = shlex.split(input_text)
        except Exception as e:
            return "[red]Invalid input[/red]"

        if not self.cli_connector.command_sets:
            return ""

        complete_token = input_text.endswith(" ")

        # Determine starting set and adjust tokens if needed.
        if tokens:
            if tokens[0] in ["..", "."]:
                if tokens[0] == "..":
                    if len(self.cli_connector.active_set) > 1:
                        starting_set = self.get_command_set_by_path(self.cli_connector.active_set[:-1])
                    else:
                        starting_set = self.get_current_command_set()
                else:  # tokens[0] == "."
                    starting_set = self.cli_connector.command_sets
                tokens = tokens[1:]
            else:
                starting_set = self.get_current_command_set()
        else:
            starting_set = self.get_current_command_set()

        # Separate fully typed tokens from the last (partial) token.
        if tokens:
            if complete_token:
                tokens_to_traverse = tokens
                partial_token = None
            else:
                tokens_to_traverse = tokens[:-1]
                partial_token = tokens[-1]
        else:
            tokens_to_traverse = []
            partial_token = None

        valid_path = True
        recognized_command = None
        current_set = starting_set

        # Traverse the fully typed tokens.
        for token in tokens_to_traverse:
            if token in current_set.get("child_sets", {}):
                current_set = current_set["child_sets"][token]
            elif token in current_set.get("commands", {}):
                recognized_command = current_set["commands"][token]
                # Once a command is recognized, we stop traversing.
                break
            else:
                valid_path = False
                break

        if not valid_path:
            return ""

        # If a valid command has been recognized, show its argument suggestions.
        if recognized_command is not None:
            # We consider the command as complete if it was typed fully,
            # or if additional tokens (arguments) are being entered.
            args = recognized_command.get("arguments", {})
            if args:
                arg_details = []
                for arg_name, arg_def in args.items():
                    arg_type = arg_def.get("type", "str")
                    if arg_type.lower() in ("flag", "bool"):
                        arg_type = "FLAG"
                    if "short_name" in arg_def and arg_def["short_name"]:
                        arg_details.append(f"--{arg_name} (-{arg_def['short_name']}, {arg_type})")
                    else:
                        arg_details.append(f"--{arg_name} ({arg_type})")
                return f"Inputs: {' '.join(arg_details)}"
            else:
                return ""

        # Otherwise, we are still selecting a child set/command.
        if partial_token is not None:
            filter_str = partial_token.lower()
            sets = [s for s in current_set.get("child_sets", {}) if s.lower().startswith(filter_str)]
            cmds = [c for c in current_set.get("commands", {}) if c.lower().startswith(filter_str)]
        else:
            sets = list(current_set.get("child_sets", {}).keys())
            cmds = list(current_set.get("commands", {}).keys())

        extras = []
        if len(self.cli_connector.active_set) > 1:
            if partial_token is None or "..".startswith(partial_token):
                extras.append("..")
        if partial_token is None or ".".startswith(partial_token):
            extras.append(".")

        suggestions = [f"[yellow]{s}[/yellow]" for s in extras + sets] + [f"[cyan]{c}[/cyan]" for c in cmds]
        result = " ".join(suggestions)
        return f"Commands: {result}" if result else ""

    def update_log_widgets(self):
        """
        Update the log-path-box with the current command-set path and update the suggestions.
        """
        log_widget = self.query_one("#log-widget", LogTabWidget)
        if log_widget:
            path_box = log_widget.query_one("#log-path-box", Static)
            current_path = " / ".join(self.cli_connector.active_set)
            path_box.update(f"Current path: [bold yellow]{current_path}[/bold yellow]")
            input_widget = log_widget.query_one("#log-input", CommandInput)
            if input_widget and input_widget.suggestions_box:
                suggestions = self.get_suggestions_for_input(input_widget.value)
                input_widget.suggestions_box.update(suggestions)

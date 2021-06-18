import collections
import io
from typing import List

import click
from rich import print
from rich.console import Console

import ehrapy
from ehrapy.cli.custom_cli.levenstein_distance import most_similar_command

# ehrapy's main commands
MAIN_COMMANDS = {"read"}


class HelpErrorHandling(click.Group):
    """Customise the help command."""

    def __init__(self, name=None, commands=None, **kwargs):
        super(HelpErrorHandling, self).__init__(name, commands, **kwargs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx: click.Context):
        return self.commands

    def main_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Load the main options and display them in a customized option section.
        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        ehrapy_main_options: List = []
        # NOTE: this only works for options as arguments do not have a help attribute per default
        for p in ctx.command.params:
            ehrapy_main_options.append(("--" + p.name + ": ", p.help))  # type: ignore
        ehrapy_main_options.append(("--help: ", "   Get detailed info on a command."))
        with formatter.section(HelpErrorHandling.get_rich_value("Options")):
            for t in ehrapy_main_options:
                formatter.write_text(f"{t[0] + t[1]}")

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Call format_help function with ehrapys customized functions.

        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        self.format_usage(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_commands(ctx, formatter)

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Overwrite format_usage method of class MultiCommand for customized usage section output.

        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        formatter.write_text(
            f'{HelpErrorHandling.get_rich_value("Usage:")} ehrapy {" ".join(super().collect_usage_pieces(ctx))}'
        )

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Overwrite the format_options method of class MultiCommand for customized option output.

        This is internally called by format_help() which itself is called by get_help().

        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        self.main_options(ctx, formatter)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Overwrite the format_commands method of class MultiCommand for customized commands output.

        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        formatter.width = 120

        with formatter.section(HelpErrorHandling.get_rich_value("General Commands")):
            formatter.write_text(
                f"{self.commands.get('upgrade').name}\t\t{self.commands.get('upgrade').get_short_help_str(limit=150)}"
            )

        with formatter.section(HelpErrorHandling.get_rich_value("Commands to work with EHR data")):
            formatter.write_text(
                f"{self.commands.get('read').name}\t\t{self.commands.get('read').get_short_help_str(limit=150)}"
            )

        # with formatter.section(HelpErrorHandling.get_rich_value("Special commands")):
        #     formatter.write_text(
        #         f"{self.commands.get('warp').name}\t\t{self.commands.get('warp').get_short_help_str(limit=150)}"
        #     )

        with formatter.section(HelpErrorHandling.get_rich_value("Examples")):
            formatter.write_text("$ ehrapy read data.csv")

        with formatter.section(HelpErrorHandling.get_rich_value("Learn more")):
            formatter.write_text(
                "Use ehrapy <command> --help for more information about a command."
                "You may also want to take a look at our docs at https://ehrapy.readthedocs.io/."
            )

        # with formatter.section(HelpErrorHandling.get_rich_value("Feedback")):
        #     formatter.write_text(
        #         "We are always curious about your opinion on ehrapy. Join our Discord at "
        #         "https://discord.gg/CwRXMdSg and drop us a message: cookies await you."
        #     )

    def get_command(self, ctx, cmd_name):
        """Override the get_command of Click.

        If an unknown command is given, try to determine a similar command.
         If no similar command could´ve been found, exit with an error message.
         Else use the most similar command while printing a status message for the user.

         Args:
             ctx: Click context
             cmd_name: The by Click invoked command
        """
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        sim_commands, action = most_similar_command(cmd_name, MAIN_COMMANDS)

        matches = [cmd for cmd in self.list_commands(ctx) if cmd in sim_commands]

        # no similar commands could be found
        if not matches:
            ctx.fail(click.style("Unknown command and no similar command was found!", fg="red"))
        elif len(matches) == 1 and action == "use":
            print(f"[bold red]Unknown command! Will use best match [green]{matches[0]}")
            return click.Group.get_command(self, ctx, matches[0])
        elif len(matches) == 1 and action == "suggest":
            ctx.fail(
                click.style("Unknown command! Did you mean ", fg="red") + click.style(f"{matches[0]}?", fg="green")
            )

        # a few similar commands were found, print a info message
        ctx.fail(
            click.style("Unknown command. Most similar commands were", fg="red")
            + click.style(f'{", ".join(sorted(matches))}', fg="red")
        )

    @staticmethod
    def args_not_provided(ctx, cmd: str) -> None:
        """Print a failure message depending on the command.

        Args:
            ctx: Click context
            cmd: The by Click invoked command
        """
        pass
        # if cmd == "info":
        #     print(
        #         f"[bold red]Failed to execute [bold green]{cmd}.\n"
        #         f"[bold blue]Please provide a valid handle like [bold green]cli [bold blue]as argument."
        #     )
        #     sys.exit(1)

    @staticmethod
    def get_rich_value(output: str, is_header: bool = True) -> str:
        """Return a string which contains the output to console rendered by rich for the click formatter.

        Args:
            output: Output string to be rendered by Rich
            is_header: Whether the output string is a header section

        Returns:
            Rendered Rich string
        """
        sio = io.StringIO()
        console = Console(file=sio, force_terminal=True)
        if is_header:
            console.print(f"[bold #1874cd]{output}")

        return sio.getvalue().replace("\n", "")


class CustomHelpSubcommand(click.Command):
    """Customize the help output for each subcommand"""

    def __init__(self, *args, **kwargs):
        super(CustomHelpSubcommand, self).__init__(*args, **kwargs)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Custom implementation of formatting help for each subcommand.

        Use the overwritten format functions this class provides to output help for each subcommand ehrapy provides.

        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        formatter.width = 120
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Custom implementation if formatting the usage of each subcommand.

        Usage section with a styled header will be printed.

        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        formatter.write_text(
            f'{HelpErrorHandling.get_rich_value("Usage: ")}ehrapy {self.name} {" ".join(self.collect_usage_pieces(ctx))}'
        )

    def format_help_text(self, ctx: click.Context, formatter: click.HelpFormatter):
        """
        Custom implementation of formatting the help text of each subcommand.
        The help text will be printed as normal. A separate arguments section will be added below with all arguments and
        a short help message for each of them and a styled header in order to keep things separated.
        """
        formatter.write_paragraph()
        formatter.write_text(self.help)
        args = [("--" + param.name, param.helpmsg) for param in self.params if type(param) == CustomArg]  # type: ignore
        if args:
            with formatter.section(HelpErrorHandling.get_rich_value("Arguments")):
                formatter.write_dl(args)

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Custom implementation of formatting the options of each subcommand.

        The options will be displayed in their relative order with their corresponding help message and a styled header.

        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        options = [
            ("--" + param.name.replace("_", "-"), param.help)  # type: ignore
            for param in self.params
            if type(param) == click.core.Option
        ]
        help_option = self.get_help_option(ctx)
        options.append(("--" + help_option.name, help_option.help))
        with formatter.section(HelpErrorHandling.get_rich_value("Options")):
            formatter.write_dl(options)

    def get_rich_value(self, output: str, is_header: bool = True) -> str:
        """Return a string which contains the output to console rendered by Rich for the Click formatter.

        Args:
            output: Output string rendered by Rich
            is_header:  Whether the string is a part of a header section

        Returns:
            The Rich rendered string
        """
        sio = io.StringIO()
        console = Console(file=sio, force_terminal=True)
        if is_header:
            console.print(f"[bold #1874cd]{output}")

        return sio.getvalue().replace("\n", "")


class CustomArg(click.Argument):
    """A custom argument implementation of click.Argument class to
    provide a short help message for each argument of a command."""

    def __init__(self, *args, **kwargs):
        self.helpmsg = kwargs.pop("helpmsg")
        super().__init__(*args, **kwargs)


def print_ehrapy_version(ctx, param, value) -> None:
    """Print ehrapy version styled with Rich.

    Args:
        ctx: Click context
        param:
        value:
    """
    # if context uses resilient parsing (no changes of execution flow) or no flag value is provided, do nothing
    if not value or ctx.resilient_parsing:
        return
    try:
        print(f"[bold blue]ehrapy version: {ehrapy.__version__}")
        ctx.exit()
    except click.ClickException:
        ctx.fail("An error occurred fetching ehrapy's version!")

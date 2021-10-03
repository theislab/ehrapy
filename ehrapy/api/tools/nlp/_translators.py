from typing import Dict, List, Union

import deepl
from anndata import AnnData
from deepl import Formality, GlossaryInfo, TextResult
from rich import print

from ehrapy.api._util import get_column_indices, get_column_values


class DeepL:
    """Implementation of the DeepL translator"""

    def __init__(self, authentication_key: str):
        self.translator = deepl.Translator(authentication_key)

    def _check_usage(function) -> ():  # type: ignore # noqa
        """Checks the usage limit of the DeepL Account.

        Prints a warning if the DeepL usage limit is exceeded.

        Args:
            function: The function to actually call
        """

        def wrapper(self, *args, **kwargs) -> None:
            usage = self.translator.get_usage()
            if usage.any_limit_exceeded:
                print("[bold red]DeepL limit exceeded. Please increase your quota")
            else:
                if (usage.character.count / usage.character.limit) > 0.9:
                    print(
                        "[bold yellow]Reached 90% of the character translation limit. "
                        "Ensure that you have enough quota."
                    )
                    print(f"[bold yellow]{self.translator.get_usage}")
                elif usage.document.limit is not None and (usage.document.count / usage.document.limit) > 0.9:
                    print(
                        "[bold yellow]Reached 90% of the document translation limit. "
                        "Ensure that you have enough quota."
                    )
                    print(f"[bold yellow]Current usage: {usage.document.count}")
                elif (
                    usage.team_document.limit is not None
                    and (usage.team_document.count / usage.team_document.limit) > 0.9
                ):
                    print(
                        "[bold yellow]Reached 90% of the team document translation limit "
                        "Ensure that you have enough quota"
                    )
                    print(f"[bold yellow]Current usage: {usage.team_document.count}")

                return function(self, *args, **kwargs)  # type: ignore

        return wrapper

    @_check_usage  # type: ignore
    def authenticate(self, authentication_key: str) -> None:
        """Authenticates the DeepL user

        Args:
            authentication_key: DeepL authentication key
        """
        self.translator = deepl.Translator(authentication_key)

    def print_source_languages(self) -> None:
        """prints all possible source languages to translate from

        Example: "DE (German)"
        """
        for language in self.translator.get_source_languages():
            print(f"{language.code} ({language.name})")

    def print_target_languages(self) -> None:
        """Prints all possible target languages to translate to"""
        for language in self.translator.get_target_languages():
            if language.supports_formality:
                print(f"{language.code} ({language.name}) supports formality")
            else:
                print(f"{language.code} ({language.name})")

    @_check_usage  # type: ignore
    def translate_text(self, text: Union[str, List], target_language: str) -> Union[TextResult, List[TextResult]]:
        """Translates the provided text into the target language

        Args:
            text: The text to translate
            target_language: The target language to translate the Text into, e.g. EN-GB

        Returns:
            A :class:`~deepl.TextResult` object
        """
        return self.translator.translate_text(text, target_lang=target_language)

    @_check_usage  # type: ignore
    def translate_document(
        self, input_file_path: str, output_path: str, target_language: str, formality: str = Formality.DEFAULT
    ) -> None:
        """Translate a complete document into the target language

        Args:
            input_file_path: File path to the document to translate
            output_path: Output path to write the translation to
            target_language: Target language to translate the document into
            formality: Desired formality for translation, as Formality enum, "less" or "more".
        """
        self.translator.translate_document_from_filepath(
            input_file_path, output_path, target_lang=target_language, formality=formality
        )

    @_check_usage  # type: ignore
    def create_glossary(
        self, glossary_name: str, source_language: str, target_language: str, entries: Dict[str, str]
    ) -> GlossaryInfo:
        """Creates a DeepL Glossary to translate with.

        A Glossary may help ensuring that specific words get translated into specific translations

        Args:
            glossary_name: Name of the Glossary
            source_language: The source language of the Glossary
            target_language: The target language of the Glossary
            entries: A Dictionary of Glossary entries

        Returns:
            A :class:`~deepl.GlossaryInfo` object
        """
        return self.translator.create_glossary(glossary_name, source_language, target_language, entries)

    @_check_usage  # type: ignore
    def translate_with_glossary(
        self, text: Union[str, List], glossary: GlossaryInfo
    ) -> Union[TextResult, List[TextResult]]:
        """Translates text with a provided Glossary

        Args:
            text: Text to translate
            glossary: A :class:`~deepl.GlossaryInfo` object

        Returns:
            A :class:`~deepl.TextResult` object
        """
        return self.translator.translate_text_with_glossary(text, glossary)

    def translate_obs_column(
        self,
        adata: AnnData,
        target_language: str,
        columns=Union[str, List],
        translate_column_name: bool = False,
        inplace: bool = False,
    ) -> None:
        """Translates a single obs column and optionally replaces the original values

        Args:
            adata: :class:`~anndata.AnnData` object containing the obs column to translate
            target_language: The target language to translate into, e.g. EN-US
            columns: The columns to translate. Can be either a single column (str) or a list of columns
            translate_column_name: Whether to translate the column name itself
            inplace: Whether to replace the obs values or add a new obs column
        """
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            # as of Pandas 1.1.0 the default for new string column is still 'object'
            if adata.obs[column].dtype != str and adata.obs[column].dtype != object:
                raise ValueError("Attempted to translate column {column} which does not contain only strings.")
            target_column = column
            if translate_column_name:
                target_column = self.translator.translate_text(column, target_lang=target_language).text
            if not inplace:
                f"{target_column}_{target_language}"

            adata.obs[target_column] = adata.obs[column].apply(
                lambda text: self.translator.translate_text(text, target_lang=target_language).text
            )

    def translate_var_column(
        self,
        adata: AnnData,
        target_language: str,
        columns=Union[str, List],
        translate_column_name: bool = False,
        inplace: bool = False,
    ) -> None:
        """Translates a single var column and optionally replaces the original values

        Args:
            adata: :class:`~anndata.AnnData` object containing the obs column to translate
            target_language: The target language to translate into, e.g. EN-US
            columns: The columns to translate. Can be either a single column (str) or a list of columns
            translate_column_name: Whether to translate the column name itself
            inplace: Whether to replace the obs values or add a new obs column
        """
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            # as of Pandas 1.1.0 the default for new string column is still 'object'
            if adata.var[column].dtype != str and adata.var[column].dtype != object:
                raise ValueError("Attempted to translate column {column} which does not contain only strings.")
            target_column = column
            if translate_column_name:
                target_column = self.translator.translate_text(column, target_lang=target_language).text
            if not inplace:
                f"{target_column}_{target_language}"

            adata.var[target_column] = adata.var[column].apply(
                lambda text: self.translator.translate_text(text, target_lang=target_language).text
            )

    def translate_X_column(
        self,
        adata: AnnData,
        target_language: str,
        columns=Union[str, List],
        translate_column_name: bool = False,
        inplace: bool = False,
    ) -> None:
        """Translates a X column into the target language

        Args:
            adata: :class:`~anndata.AnnData` object containing the var column to translate
            target_language: The target language to translate into, e.g. EN-US
            columns: The columns to translate. Can be either a single column (str) or a list of columns
            translate_column_name: Whether to translate the column name itself
            inplace: Whether to replace the obs values or add a new obs column
        """
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            target_column = column
            if translate_column_name:
                target_column = self.translator.translate_text(column, target_lang=target_language).text
            if not inplace:
                f"{target_column}_{target_language}"

            index = get_column_indices(adata, column)
            column_values = get_column_values(adata, index)

            if column_values.dtype != str and column_values.dtype != object:
                raise ValueError("Attempted to translate column {column} which does not contain only strings.")

            f = lambda x: x + "blub"
            translated_column_values = f(column_values)

            print(translated_column_values)

            # Replace the column in X with the column here
            # Also possibly the var name

from __future__ import annotations

from functools import wraps
from typing import List, Union

import deepl
import numpy as np
from anndata import AnnData
from deep_translator import GoogleTranslator
from deepl import Formality, GlossaryInfo, TextResult
from rich import print

from ehrapy.api.anndata_ext import get_column_indices, get_column_values


class Translator:
    """Class providing an interface to all translation functions. Requires a flavour."""

    def __init__(
        self, flavour: str = "deepl", source: str = "de", target: str = "en", token: str = None
    ) -> None:  # pragma: no cover
        if flavour == "deepl":
            self.translator = DeepL(token)
        elif flavour == "googletranslate":
            self.translator = GoogleTranslate(source, target)  # type: ignore
        else:
            raise NotImplementedError(f"Flavour '{flavour}' is not supported.")
        self.flavour = flavour
        self.source_language = source
        self.target_language = target

    def translate_text(self, text: str | list, target_language: str = None) -> str | list[str]:  # pragma: no cover
        """Translates the provided text into the target language.

        Args:
            text: The text to translate
            target_language: The target language to translate the Text into, e.g. EN-GB

        Returns:
            A :class:`~deepl.TextResult` object
        """
        if target_language is None:
            target_language = self.target_language

        return self.translator.translate_text(text, target_language=target_language)

    def translate_obs_column(
        self,
        adata: AnnData,
        columns=Union[str, List],
        translate_column_name: bool = False,
        inplace: bool = False,
    ) -> None:
        """Translates a single obs column and optionally replaces the original values

        Args:
            adata: :class:`~anndata.AnnData` object containing the obs column to translate
            target_language: The target language to translate into (default: EN-US)
            columns: The columns to translate. Can be either a single column (str) or a list of columns
            translate_column_name: Whether to translate the column name itself
            inplace: Whether to replace the obs values or add a new obs column
        """
        if isinstance(columns, str):  # pragma: no cover
            columns = [columns]

        translate_text = self.translate_text

        for column in columns:
            # as of Pandas 1.1.0 the default for new string column is still 'object'
            if adata.obs[column].dtype != str and adata.obs[column].dtype != object:  # pragma: no cover
                raise ValueError("Attempted to translate column {column} which does not contain only strings.")
            target_column = column
            if translate_column_name:  # TODO This requires a test
                target_column = translate_text(column)
            if not inplace:  # pragma: no cover
                target_column = f"{target_column}_{self.target_language}"

            adata.obs[target_column] = adata.obs[column].apply(translate_text)

    def translate_var_column(
        self,
        adata: AnnData,
        columns=Union[str, List],
        translate_column_name: bool = False,
        inplace: bool = False,
    ) -> None:
        """Translates a single var column and optionally replaces the original values

        Args:
            adata: :class:`~anndata.AnnData` object containing the obs column to translate
            target_language: The target language to translate into (default: EN-US)
            columns: The columns to translate. Can be either a single column (str) or a list of columns
            translate_column_name: Whether to translate the column name itself
            inplace: Whether to replace the obs values or add a new obs column
        """
        if isinstance(columns, str):  # pragma: no cover
            columns = [columns]

        translate_text = self.translate_text

        for column in columns:
            # as of Pandas 1.1.0 the default for new string column is still 'object'
            if adata.var[column].dtype != str and adata.var[column].dtype != object:  # pragma: no cover
                raise ValueError("Attempted to translate column {column} which does not contain only strings.")
            target_column = column
            if translate_column_name:  # TODO this requires a test
                target_column = translate_text(column)
            if not inplace:  # pragma: no cover
                target_column = f"{target_column}_{self.target_language}"

            adata.var[target_column] = adata.var[column].apply(translate_text)

    def translate_X_column(
        self,
        adata: AnnData,
        columns=Union[str, List],
        translate_column_name: bool = False,
    ) -> None:
        """Translates a X column into the target language in place.

        Note that the translation of a column in X is **always** in place.

        Args:
            adata: :class:`~anndata.AnnData` object containing the var column to translate
            target_language: The target language to translate into (default: EN-US)
            columns: The columns to translate. Can be either a single column (str) or a list of columns
            translate_column_name: Whether to translate the column name itself (only translates var_names, not var)
        """
        if isinstance(columns, str):  # pragma: no cover
            columns = [columns]

        translate_text = self.translate_text
        indices = get_column_indices(adata, columns)

        for column, index in zip(columns, indices):
            column_values = get_column_values(adata, index)

            if column_values.dtype != str and column_values.dtype != object:  # pragma: no cover
                raise ValueError("Attempted to translate column {column} which does not only contain strings.")

            if translate_column_name:  # TODO This requires a test
                translated_column_name = translate_text(column)
                index_values = adata.var_names.tolist()
                index_values[index] = translated_column_name
                adata.var_names = index_values

            translated_column_values: list = translate_text(
                column_values  # type: ignore
            )  # TODO: Check that structure is still ok
            # translated_column_values = list(map(lambda text_result: text_result.text, translated_column_values))

            adata.X[:, index] = translated_column_values


class DeepL:
    """Implementation of the DeepL translator"""

    def __init__(self, authentication_key: str):
        self.translator = deepl.Translator(authentication_key)

    def _check_usage(function) -> ():  # type: ignore # noqa # pragma: no cover
        """Checks the usage limit of the DeepL Account.

        Prints a warning if the DeepL usage limit is exceeded.

        Args:
            function: The function to actually call
        """

        @wraps(function)  # type: ignore
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

    @_check_usage  # type: ignore # pragma: no cover
    def authenticate(self, authentication_key: str) -> None:
        """Authenticates the DeepL user

        Args:
            authentication_key: DeepL authentication key
        """
        self.translator = deepl.Translator(authentication_key)

    def print_source_languages(self) -> None:  # pragma: no cover
        """prints all possible source languages to translate from

        Example: "DE (German)"
        """
        for language in self.translator.get_source_languages():
            print(f"{language.code} ({language.name})")

    def print_target_languages(self) -> None:  # pragma: no cover
        """Prints all possible target languages to translate to"""
        for language in self.translator.get_target_languages():
            if language.supports_formality:
                print(f"{language.code} ({language.name}) supports formality")
            else:
                print(f"{language.code} ({language.name})")

    @_check_usage  # type: ignore
    def translate_text(self, text: str | list, target_language: str) -> list[np.ndarray] | str:
        """Translates the provided text into the target language

        Args:
            text: The text to translate
            target_language: The target language to translate the Text into, e.g. EN-GB

        Returns:
            A :class:`~deepl.TextResult` object
        """
        if isinstance(text, List) or isinstance(text, np.ndarray):
            return [
                self.translator.translate_text(translation, target_lang=target_language).text for translation in text
            ]
        return self.translator.translate_text(text, target_lang=target_language).text

    @_check_usage  # type: ignore # pragma: no cover
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

    @_check_usage  # type: ignore # pragma: no cover
    def create_glossary(
        self, glossary_name: str, source_language: str, target_language: str, entries: dict[str, str]
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

    @_check_usage  # type: ignore # pragma: no cover
    def translate_with_glossary(self, text: str | list, glossary: GlossaryInfo) -> TextResult | list[TextResult]:
        """Translates text with a provided Glossary

        Args:
            text: Text to translate
            glossary: A :class:`~deepl.GlossaryInfo` object

        Returns:
            A :class:`~deepl.TextResult` object
        """
        return self.translator.translate_text_with_glossary(text, glossary)


class GoogleTranslate:
    def __init__(self, source="auto", target="en"):
        self.translator = GoogleTranslator(source, target)

    def print_source_languages(self) -> None:  # pragma: no cover
        """prints all possible source languages to translate from

        Example: "DE (German)"
        """
        for code, language in self.translator.get_supported_languages(as_dict=True).items():
            print(f"{code} ({language})")

    def print_target_languages(self) -> None:  # pragma: no cover
        """Prints all possible target languages to translate to"""
        for code, language in self.translator.get_supported_languages(as_dict=True).items():
            print(f"{code} ({language})")

    # @_check_usage  # type: ignore
    def translate_text(self, text: str | list, target_language: str) -> str | list[str]:
        """Translates the provided text into the target language

        Args:
            text: The text to translate
            target_language: The target language to translate the Text into, e.g. EN-GB

        Returns:
            A :class:`~deepl.TextResult` object
        """
        if isinstance(text, List) or isinstance(text, np.ndarray):
            return [self.translator.translate(word, target_lang=target_language) for word in text]
        return self.translator.translate(text, target_lang=target_language)

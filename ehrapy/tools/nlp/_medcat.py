from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
from medcat.vocab import Vocab
from rich import print
from rich.progress import Progress, SpinnerColumn
from spacy import displacy
from spacy.tokens.doc import Doc

from ehrapy import settings
from ehrapy.core.tool_available import check_module_importable


# TODO: State in docs those models are only needed when not using a model pack (cdb and vocab separatly)
# so this check can be removed
spacy_models_modules: list[str] = list(map(lambda model: model.replace("-", "_"), ["en-core-web-md"]))
for model in spacy_models_modules:
    if not check_module_importable(model):
        print(
            f"[bold yellow]Model {model} is not installed. Refer to the ehrapy installation instructions if required."
        )


# TODO: Discuss this could be used as a custom class for results
@dataclass
class AnnotationResult:
    all_medcat_annotation_results: list | None
    entities_pretty: dict[str, list[str]]
    cui_locations: dict | None
    type_ids_location: dict | None


class MedCAT:
    def __init__(self, vocabulary: Vocab = None, concept_db: CDB = None, model_pack_path=None):
        if not check_module_importable("medcat"):
            raise RuntimeError("medcat is not importable. Please install via pip install medcat")
        self.vocabulary = vocabulary
        self.concept_db = concept_db
        if self.vocabulary is not None and self.concept_db is not None:
            self.cat = CAT(cdb=concept_db, config=concept_db.config, vocab=vocabulary)
        elif model_pack_path is not None:
            self.cat = CAT.load_model_pack(model_pack_path)

    def update_cat(self, vocabulary: Vocab = None, concept_db: CDB = None):
        """Updates the current MedCAT instance with new Vocabularies and Concept Databases.

        Args:
            vocabulary: Vocabulary to update to.
            concept_db: Concept Database to update to.
        """
        self.cat = CAT(cdb=concept_db, config=concept_db.config, vocab=vocabulary)

    def update_cat_config(self, concept_db_config: Config) -> None:
        """Updates the MedCAT configuration.

        Args:
            concept_db_config: Concept to update to.
        """
        self.concept_db.config = concept_db_config

    def set_filter_by_tui(self, tuis: list[str] | None = None):
        """Restrict results of annotation step to certain tui's (type unique identifiers).
        Note that this will change the MedCat object by updating the concept database config. In every annotation
        process that will be run afterwards, entities are shown, only if they fall into the tui's type.
        A full list of tui's can be found at: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt

        As an exmaple:
        Setting tuis=["T047", "T048"] will only annotate concepts (identified by a CUI (concept unique identifier)) in UMLS that are either diseases or
        syndroms (T047) or mental/behavioural dysfunctions (T048). This is the default value.

        Args:
            tuis: list of TUI's
        Returns:

        """
        if tuis is None:
            tuis = ['T047', 'T048']
        # the filtered cui's that fall into the type of the filter tui's
        cui_filters = set()
        for type_id in tuis:
            cui_filters.update(self.cat.cdb.addl_info['type_id2cuis'][type_id])
        self.cat.cdb.config.linking['filters']['cuis'] = cui_filters


def create_vocabulary(vocabulary_data: str, replace: bool = True) -> Vocab:
    """Creates a MedCAT Vocab and sets it for the MedCAT object.

    Args:
        vocabulary_data: Path to the vocabulary data.
                         It is a tsv file and must look like:

                         <token>\t<word_count>\t<vector_embedding_separated_by_spaces>
                         house    34444     0.3232 0.123213 1.231231
        replace: Whether to replace existing words in the vocabulary.

    Returns:
        Instance of a MedCAT Vocab
    """
    vocabulary = Vocab()
    vocabulary.add_words(vocabulary_data, replace=replace)

    return vocabulary


def create_concept_db(csv_path: list[str], config: Config = None) -> CDB:
    """Creates a MedCAT concept database and sets it for the MedCAT object.

    Args:
        csv_path: List of paths to one or more csv files containing all concepts.
                  The concept csvs must look like:

                  cui,name
                  1,kidney failure
                  7,coronavirus
        config: Optional MedCAT concept database configuration.
                If not provided a default configuration with config.general['spacy_model'] = 'en_core_sci_md' is created.
    Returns:
        Instance of a MedCAT CDB concept database
    """
    if config is None:
        config = Config()
        config.general["spacy_model"] = "en_core_sci_md"
    maker = CDBMaker(config)
    concept_db = maker.prepare_csvs(csv_path, full_build=True)

    return concept_db


def save_vocabulary(vocab: Vocab, output_path: str) -> None:
    """Saves a vocabulary.

    Args:
        output_path: Path to write the vocabulary to.
    """
    vocab.save(output_path)


def load_vocabulary(vocabulary_path) -> Vocab:
    """Loads a vocabulary.

    Args:
        vocabulary_path: Path to load the vocabulary from.
    """

    return Vocab.load(vocabulary_path)


def save_concept_db(cdb, output_path: str) -> None:
    """Saves a concept database.

    Args:
        output_path: Path to save the concept database to.
    """
    cdb.save(output_path)


def load_concept_db(concept_db_path) -> CDB:
    """Loads the concept database.

    Args:
        concept_db_path: Path to load the concept database from.
    """

    return CDB.load(concept_db_path)


def save_model_pack(ep_cat: MedCAT, model_pack_dir: str = ".", name: str = "ehrapy_medcat_model_pack") -> None:
    """Saves a MedCAT model pack.

    Args:
        ep_cat: ehrapy's custom MedCAT object whose model should be saved
        model_pack_dir: Path to save the model to (defaults to current working directory).
        name: Name of the new model pack
    """
    # TODO Pathing is weird here (home/myname/...) will fo example create dir myname inside home inside the cwd instead of using the path
    _ = ep_cat.cat.create_model_pack(model_pack_dir + name)


def run_unsupervised_training(ep_cat: MedCAT, text: pd.Series, progress_print: int = 100, print_statistics: bool = False) -> None:
    """Performs MedCAT unsupervised training on a provided text column.

    Args:
        ep_cat: ehrapy's custom MedCAT object, that keeps track of the vocab, concept database and the (annotated) results
        text: Pandas Series of (free) text to annotate.
        progress_print: print progress after that many training documents
        print_statistics: Whether to print training statistics after training.
    """
    print(f"[bold blue]Running unsupervised training using {len(text)} documents.")
    ep_cat.cat.train(text.values, progress_print=progress_print)

    if print_statistics:
        ep_cat.cat.cdb.print_stats()


def annotate_text(ep_cat: MedCAT, obs: pd.DataFrame, text_column: str, n_proc: int = 2, batch_size_chars: int = 500000) -> None:
    """Annotate the original free text data. Note this will only annotate non null rows. The result
    will be a (large) dict containing all the entities extracted from the free text (and in case filtered before via set_filter_by_tui function).
    This dict will be the base for all further analyses, for example coloring umaps by specific diseases.

    Args:
        ep_cat: Ehrapy's custom MedCAT object
        obs: AnnData obs containing the free text column
        text_column: Name of the column that should be annotated
        n_proc: Number of processors to use
        batch_size_chars: batch size to control for the variablity between document sizes

    """
    non_null_text = _filter_null_values(obs, text_column)
    formatted_text_column = _format_df_column(non_null_text, text_column)
    results = ep_cat.cat.multiprocessing(formatted_text_column, batch_size_chars=batch_size_chars, nproc=n_proc)
    # TODO: Discuss: Should we return a brand new custom "result" object here (as this is basically the base for downstream analysis) or
    # TODO: just add it to the existing ehrapy MedCAT object as the "result" attribute.
    # for testing and debugging, going with simply returning it
    return results


def _format_df_column(df: pd.DataFrame, column_name: str) -> list[tuple[int, str]]:
    """Format the df to match: formatted_data = [(row_id, row_text), (row_id, row_text), ...]
    as this is required by medcat's multiprocessing annotation step
    """
    # TODO This can be very memory consuming -> possible to use generators here instead (medcat compatible?)?
    formatted_data = []
    for id, row in df.iterrows():
        text = row[column_name]
        formatted_data.append((id, text))
    return formatted_data


def _filter_null_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filter null values of a given column and return that column without the null values
    """
    return pd.DataFrame(df[column][~df[column].isnull()])

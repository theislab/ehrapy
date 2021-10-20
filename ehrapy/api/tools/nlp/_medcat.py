from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
from medcat.vocab import Vocab
from rich import print
from rich.progress import track
from spacy import displacy
from spacy.tokens.doc import Doc

from ehrapy.api import settings
from ehrapy.api._util import check_module_importable

spacy_models_modules: List[str] = list(
    map(lambda model: model.replace("-", "_"), ["en-core-sci-sm", "en-core-sci-md", "en-core-sci-lg"])
)
for model in spacy_models_modules:
    if not check_module_importable(model):
        print(
            f"[bold yellow]Model {model} is not installed. Refer to the ehrapy installation instructions if required."
        )


class MedCAT:
    def __init__(self, vocabulary: Vocab = None, concept_db: CDB = None):
        self.vocabulary = vocabulary
        self.concept_db = concept_db
        if self.vocabulary is not None and self.concept_db is not None:
            self.cat = CAT(cdb=concept_db, config=concept_db.config, vocab=vocabulary)

    @staticmethod
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

    @staticmethod
    def create_concept_db(csv_path: List[str], config: Config = None) -> CDB:
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
            # TODO verify that the spacey model actually exists
            config.general["spacy_model"] = "en_core_sci_md"
        maker = CDBMaker(config)
        concept_db = maker.prepare_csvs(csv_path, full_build=True)

        return concept_db

    def save_vocabulary(self, output_path: str):
        self.vocabulary.save(output_path)

    def load_vocabulary(self, vocabulary_path):
        self.vocabulary = Vocab.load(vocabulary_path)

        return self.vocabulary

    def save_concept_db(self, output_path: str):
        self.concept_db.save(output_path)

    def load_concept_db(self, concept_db_path):
        self.concept_db = CDB.load(concept_db_path)

        return self.concept_db

    def update_cat(self, vocabulary: Vocab = None, concept_db: CDB = None):
        self.cat = CAT(cdb=concept_db, config=concept_db.config, vocab=vocabulary)

    def extract_entities_text(self, text: str) -> Doc:
        """Extracts entities for a provided text.

        Args:
            text: The text to extract entities from

        Returns:
            A spacy Doc instance. Extract the entities using. doc.ents
        """
        return self.cat(text)

    def print_cui(self, doc: Doc, table: bool = False) -> None:
        """Prints the concept unique identifier for all entities.

        Args:
            doc: A spacy tokens Doc.
            table: Whether to print a Rich table.
        """
        if table:
            # TODO IMPLEMENT ME
            pass
        else:
            for entity in doc.ents:
                # TODO make me pretty by ensuring that the lengths before the - token are aligned
                print(f"[bold blue]{entity} - {entity._.cui}")

    def print_semantic_type(self, doc: Doc, table: bool = False) -> None:
        """Prints the semantic types for all entities.

        Args:
            doc: A spacy tokens Doc.
            table: Whether to print a Rich table.
        """
        for ent in doc.ents:
            print(ent, " - ", self.concept_db.cui2type_ids.get(ent._.cui))

    def print_displacy(self, doc: Doc, style: Literal["deb", "ent"] = "ent") -> None:
        """Prints a Doc with displacy

        Args:
            doc: A spacy tokens Doc.
            style: The Displacy style to render
        """
        displacy.render(doc, style=style, jupyter=True)

    def run_unsupervised_training(self, text: pd.Series, print_statistics: bool = False) -> None:
        """Performs MedCAT unsupervised training on a provided text column.

        Args:
            text: Pandas Series of text to annotate.
            print_statistics: Whether to print training statistics after training.
        """
        print(f"[bold blue]Training using {len(text)} documents")
        self.cat.train(text.values, progress_print=100)

        if print_statistics:
            self.concept_db.print_stats()

    def filter_tui(self, concept_db: CDB, tui_filters: List[str]) -> None:
        """Filters a concept database by semantic types (TUI)

        Args:
            concept_db: MedCAT concept database.
            tui_filters: A list of semantic type filters. Example: |T047|Disease or Syndrome -> "T047"
        """
        # TODO Figure out a way not to do this inplace and add an inplace parameter
        cui_filters = set()
        for tui in tui_filters:
            cui_filters.update(concept_db.addl_info["type_id2cuis"][tui])
        concept_db.config.linking["filters"]["cuis"] = cui_filters
        print(f"[bold blue]The size of the concept database is now: {len(cui_filters)}")

    def annotate(
        self,
        data: Union[np.ndarray, pd.Series],
        batch_size: int = 100,
        min_text_length: int = None,
        only_cui: bool = False,
        n_jobs: int = settings.n_jobs,
    ) -> Optional[Dict]:
        """

        Args:
            data: Text data to annotate
            batch_size:
            min_text_length:
            only_cui: Returned entities will only have a CUI
            n_jobs:

        Returns:
            A dictionary: {id: doc_json, id2: doc_json2, ...}, in case out_split_size is used
            the last batch will be returned while that and all previous batches will be written to disk (out_save_dir).
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        cui_location: Dict = {}  # CUI to a list of documents where it appears
        tui_location: Dict = {}  # TUI to a list of documents where it appears
        results = None

        batch: List = []
        for text_id, text in track(data.iterrows()):
            if len(text) > min_text_length:
                batch.append((text_id, text))

            if len(batch) > batch_size or text_id == len(data) - 1:
                results = self.cat.multiprocessing(batch, nproc=n_jobs, only_cui=only_cui)

                for pair in results:
                    row_id = pair[0]
                    cui_list = set(pair[1]["entities"].values())  # Convert to set to get unique CUIs

                    for cui in cui_list:
                        if cui in cui_location:
                            cui_location[cui].append(row_id)
                        else:
                            cui_location[cui] = [row_id]

                        # This is not necessary as it can be done later, we have the cdb.cui2type_id map.
                        tuis = self.concept_db.cui2type_ids[cui]
                        for tui in tuis:  # this step is necessary as one cui may map to several tuis
                            if tui in tui_location and row_id not in tui_location[tui]:
                                tui_location[tui].append(row_id)
                            elif tui not in tui_location:
                                tui_location[tui] = [row_id]

                # Reset the batch
                batch = []

        return results

    def plot_subjects_cui(self):
        pass

    def plot_top_diseases(self):
        pass

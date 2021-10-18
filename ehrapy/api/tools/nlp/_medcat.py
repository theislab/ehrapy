from typing import List, Literal

import pandas as pd
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
from medcat.vocab import Vocab
from spacy import displacy
from spacy.tokens.doc import Doc


class MedCAT:
    def __init__(self, vocabulary: Vocab, concept_db: CDB):
        self.vocabulary = vocabulary
        self.concept_db = concept_db
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
            config.general["spacy_model"] = "en_core_sci_md"
        maker = CDBMaker(config)
        concept_db = maker.prepare_csvs(csv_path, full_build=True)

        return concept_db

    def save_vocabulary(self, output_path: str):
        self.vocabulary.save(output_path)

    def load_vocabulary(self, vocabulary_path):
        self.vocabulary = Vocab.load(vocabulary_path)

    def save_concept_db(self, output_path: str):
        self.concept_db.save(output_path)

    def load_concept_db(self, concept_db_path):
        self.concept_db = CDB.load(concept_db_path)

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

    def tui_filter(self):
        pass

    def annotate(self, multiprocessing: bool = True, text_length: int = None):
        pass

    def plot_subjects_cui(self):
        pass

    def plot_top_diseases(self):
        pass

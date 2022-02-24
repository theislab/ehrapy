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
from ehrapy._util import check_module_importable

spacy_models_modules: list[str] = list(map(lambda model: model.replace("-", "_"), ["en-core-web-md"]))
for model in spacy_models_modules:
    if not check_module_importable(model):
        print(
            f"[bold yellow]Model {model} is not installed. Refer to the ehrapy installation instructions if required."
        )


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

    def save_vocabulary(self, output_path: str) -> None:
        """Saves a MedCAT vocabulary.

        Args:
            output_path: Path to write the vocabulary to.
        """
        self.vocabulary.save(output_path)

    def load_vocabulary(self, vocabulary_path) -> Vocab:
        """Loads a MedCAT vocabulary.

        Args:
            vocabulary_path: Path to load the vocabulary from.
        """
        self.vocabulary = Vocab.load(vocabulary_path)

        return self.vocabulary

    def save_concept_db(self, output_path: str) -> None:
        """Saves a MedCAT concept database.

        Args:
            output_path: Path to save the concept database to.
        """
        self.concept_db.save(output_path)

    def load_concept_db(self, concept_db_path) -> CDB:
        """Loads the concept database.

        Args:
            concept_db_path: Path to load the concept database from.
        """
        self.concept_db = CDB.load(concept_db_path)

        return self.concept_db

    def load_model_pack(self, model_pack_path) -> None:
        """Loads a MedCAt model pack.

        Updates the MedCAT object.

        Args:
            model_pack_path: Path to save the model from.
        """
        self.cat.load_model_pack(model_pack_path)

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
        # TODO implement Rich table
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

    def filter_tui(self, concept_db: CDB, tui_filters: list[str]) -> CDB:
        """Filters a concept database by semantic types (TUI).

        Args:
            concept_db: MedCAT concept database.
            tui_filters: A list of semantic type filters. Example: T047 Disease or Syndrome -> "T047"

        Returns:
            A filtered MedCAT concept database.
        """
        # TODO Figure out a way not to do this inplace and add an inplace parameter
        cui_filters = set()
        for tui in tui_filters:
            cui_filters.update(concept_db.addl_info["type_id2cuis"][tui])
        concept_db.config.linking["filters"]["cuis"] = cui_filters
        print(f"[bold blue]The size of the concept database is now: {len(cui_filters)}")

        return concept_db

    def annotate(
        self,
        data: np.ndarray | pd.Series,
        batch_size_chars=500000,
        min_text_length=5,
        n_jobs: int = settings.n_jobs,
    ) -> AnnotationResult:
        """Annotates a set of texts.

        Args:
            data: Text data to annotate.
            batch_size_chars: Batch size in number of characters
            min_text_length: Minimum text length
            n_jobs: Number of parallel processes

        Returns:
            A dictionary: {id: doc_json, id2: doc_json2, ...}, in case out_split_size is used
            the last batch will be returned while that and all previous batches will be written to disk (out_save_dir).
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data = data[data.apply(lambda word: len(str(word)) > min_text_length)]

        cui_location: dict = {}  # CUI to a list of documents where it appears
        type_ids_location = {}  # TUI to a list of documents where it appears

        batch: list = []
        with Progress(
            "[progress.description]{task.description}",
            SpinnerColumn(),
        ) as progress:
            progress.add_task("[red]Annotating...")
            for text_id, text in data.iteritems():  # type: ignore
                batch.append((text_id, text))

            results = self.cat.multiprocessing(batch, batch_size_chars=batch_size_chars, nproc=n_jobs)

            for doc in list(results.keys()):
                for annotation in list(results[doc]["entities"].values()):
                    if annotation["cui"] in cui_location:
                        cui_location[annotation["cui"]].append(doc)
                    else:
                        cui_location[annotation["cui"]] = [doc]

            for cui in cui_location.keys():
                type_ids_location[list(self.cat.cdb.cui2type_ids[cui])[0]] = cui_location[cui]

            patient_to_entities: dict[str, list[str]] = {}
            for patient_id, findings in results.items():
                entities = []
                for _, result in findings["entities"].items():
                    entities.append(result["pretty_name"])

                patient_to_entities[patient_id] = entities

        return AnnotationResult(results, patient_to_entities, cui_location, type_ids_location)

    def calculate_disease_proportions(
        self, adata: AnnData, cui_locations: dict, subject_id_col="subject_id"
    ) -> pd.DataFrame:
        """Calculates the relative proportion of found diseases as percentages.

        Args:
            adata: AnnData object. obs of this object must contain the results.
            cui_locations: A dictionary containing the found CUIs and their location.
            subject_id_col: The column header in the data containing the patient/subject IDs.

        Returns:
            A Pandas Dataframe containing the disease percentages.

            cui 	nsubjects 	tui 	name 	perc_subjects
        """
        cui_subjects: dict[int, list[int]] = {}
        cui_subjects_unique: dict[int, set[int]] = {}
        for cui in cui_locations:
            for location in cui_locations[cui]:
                # TODO: int casting is required as AnnData requires indices to be str (maybe we can change this) so we dont need type casting here
                subject_id = adata.obs.iat[int(location), list(adata.obs.columns).index(subject_id_col)]
                if cui in cui_subjects:
                    cui_subjects[cui].append(subject_id)
                    cui_subjects_unique[cui].add(subject_id)
                else:
                    cui_subjects[cui] = [subject_id]
                    cui_subjects_unique[cui] = {subject_id}

        cui_nsubjects = [("cui", "nsubjects")]
        for cui in cui_subjects_unique.keys():
            cui_nsubjects.append((cui, len(cui_subjects_unique[cui])))  # type: ignore
        df_cui_nsubjects = pd.DataFrame(cui_nsubjects[1:], columns=cui_nsubjects[0])

        df_cui_nsubjects = df_cui_nsubjects.sort_values("nsubjects", ascending=False)
        # Add type_ids for each CUI
        df_cui_nsubjects["type_ids"] = ["unk"] * len(df_cui_nsubjects)
        cols = list(df_cui_nsubjects.columns)
        for i in range(len(df_cui_nsubjects)):
            cui = df_cui_nsubjects.iat[i, cols.index("cui")]
            type_ids = self.cat.cdb.cui2type_ids.get(cui, "unk")
            df_cui_nsubjects.iat[i, cols.index("type_ids")] = type_ids

        # Add name for each CUI
        df_cui_nsubjects["name"] = ["unk"] * len(df_cui_nsubjects)
        cols = list(df_cui_nsubjects.columns)
        for i in range(len(df_cui_nsubjects)):
            cui = df_cui_nsubjects.iat[i, cols.index("cui")]
            name = self.cat.cdb.cui2preferred_name.get(cui, "unk")
            df_cui_nsubjects.iat[i, cols.index("name")] = name

        # Add the percentage column
        total_subjects = len(adata.obs[subject_id_col].unique())
        df_cui_nsubjects["perc_subjects"] = (df_cui_nsubjects["nsubjects"] / total_subjects) * 100

        df_cui_nsubjects.reset_index(drop=True, inplace=True)

        return df_cui_nsubjects

    def plot_top_diseases(self, df_cui_nsubjects: pd.DataFrame, top_diseases: int = 30) -> None:
        """Plots the top n (default: 30) found diseases.

        Args:
            df_cui_nsubjects: Pandas DataFrame containing the determined annotations.
            top_diseases: Number of top diseases to plot
        """
        warnings.warn("This function will be moved and likely removed in a future version!", FutureWarning)
        # TODO this should ideally be drawn with a Scanpy plot or something
        # TODO Needs more options such as saving etc
        sns.set(rc={"figure.figsize": (5, 12)}, style="whitegrid", palette="pastel")
        f, ax = plt.subplots()
        _data = df_cui_nsubjects.iloc[0:top_diseases]
        sns.barplot(x="perc_subjects", y="name", data=_data, label="Disorder Name", color="b")
        _ = ax.set(xlim=(0, 70), ylabel="Disease Name", xlabel="Percentage of patients with disease")
        plt.show()

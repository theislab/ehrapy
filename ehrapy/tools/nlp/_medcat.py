from __future__ import annotations

import pandas as pd
from anndata import AnnData
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
from medcat.vocab import Vocab
from rich import box, print
from rich.console import Console
from rich.table import Table

from ehrapy.core.tool_available import check_module_importable

# TODO: State in docs those models are only needed when not using a model pack (cdb and vocab separatly)


class MedCAT:
    """Wrapper class for Medcat. This class will hold references to the current model (with vocab and concept database) and should be
    passed to all functions exposed to the ehrapy nlp API when required.
    """
    def __init__(self, vocabulary: Vocab = None, concept_db: CDB = None, model_pack_path=None):
        if not check_module_importable("medcat"):
            raise RuntimeError("medcat is not importable. Please install via pip install medcat")
        self.vocabulary = vocabulary
        self.concept_db = concept_db
        if self.vocabulary is not None and self.concept_db is not None:
            self.cat = CAT(cdb=concept_db, config=concept_db.config, vocab=vocabulary)
        elif model_pack_path is not None:
            self.cat = CAT.load_model_pack(model_pack_path)
        # will be initialized as None, but will get updated when running annotate_text
        self.annotated_results = None

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

    def set_filter_by_tui(self, tuis: list[str] | None = None) -> None:
        """Restrict results of annotation step to certain tui's (type unique identifiers).

        Note that this will change the MedCat object by updating the concept database config. In every annotation
        process that will be run afterwards, entities are shown, only if they fall into the tui's type.
        A full list of tui's can be found at: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt

        As an example:
        Setting tuis=["T047", "T048"] will only annotate concepts (identified by a CUI (concept unique identifier)) in UMLS that are either diseases or
        syndroms (T047) or mental/behavioural dysfunctions (T048). This is the default value.

        Args:
            tuis: list of TUI's (default is

        """
        if tuis is None:
            tuis = ["T047", "T048"]
        # the filtered cui's that fall into the type of the filter tui's
        cui_filters = set()
        for type_id in tuis:
            cui_filters.update(self.cat.cdb.addl_info["type_id2cuis"][type_id])
        self.cat.cdb.config.linking["filters"]["cuis"] = cui_filters

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

    @staticmethod
    def save_vocabulary(vocab: Vocab, output_path: str) -> None:
        """Saves a vocabulary.

        Args:
            vocab: The vocabulary object
            output_path: Path to write the vocabulary to.
        """
        vocab.save(output_path)

    @staticmethod
    def load_vocabulary(vocabulary_path) -> Vocab:
        """Loads a vocabulary.

        Args:
            vocabulary_path: Path to load the vocabulary from.
        """

        return Vocab.load(vocabulary_path)

    @staticmethod
    def save_concept_db(cdb, output_path: str) -> None:
        """Saves a concept database.

        Args:
            cdb: the concept database object
            output_path: Path to save the concept database to.
        """
        cdb.save(output_path)

    @staticmethod
    def load_concept_db(concept_db_path) -> CDB:
        """Loads the concept database.

        Args:
            concept_db_path: Path to load the concept database from.
        """
        return CDB.load(concept_db_path)

    def save_model_pack(self, model_pack_dir: str = ".", name: str = "ehrapy_medcat_model_pack") -> None:
        """Saves a MedCAT model pack.

        Args:
            model_pack_dir: Path to save the model to (defaults to current working directory).
            name: Name of the new model pack
        """
        # TODO Pathing is weird here (home/myname/...) will fo example create dir myname inside home inside the cwd instead of using the path
        _ = self.cat.create_model_pack(name)


class EhrapyMedcat:
    """Wrapper class to perform medcat analysis with ehrapy for free text data. This can be simply called by `ep.tl.mc`. This class is not supposed to be
    instantiated at any time, it just serves as a wrapper for import.
    """
    @staticmethod
    def run_unsupervised_training(
        ep_cat: MedCAT, text: pd.Series, progress_print: int = 100, print_statistics: bool = False
    ) -> None:
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

    @staticmethod
    def annotate_text(
        ep_cat: MedCAT, obs: pd.DataFrame, text_column: str, n_proc: int = 2, batch_size_chars: int = 500000
    ) -> None:
        """Annotate the original free text data. Note this will only annotate non null rows.
        The result will be a DataFrame (see example below). It will be set as the annotated_results attribute for the passed MedCat object.
        This dataframe will be the base for all further analyses, for example coloring umaps by specific diseases.

            .. code-block:: python
                       # some setup code here
                       ...
                       res = ep.tl.annotate_text(ep_cat, adata.obs, "text")
                       print(res)
                       # res looks like this:
                       #                                                pretty_name meta ...
                       #
                       # 1 0 (first entitiy extracted from first row)   diabetes11  fb11 ...
                       # 2 0 (second entitiy extracted from first row)  diabetes12  fb12 ...
                       # 3 1 (first entitiy extracted from second row)  diabetes21  fb21 ...
                       # 4 1 (second entitiy extracted from second row) diabetes22  fb22 ...

        Args:
            ep_cat: Ehrapy's custom MedCAT object. The annotated_results attribute will be set here.
            obs: AnnData obs containing the free text column
            text_column: Name of the column that should be annotated
            n_proc: Number of processors to use
            batch_size_chars: batch size to control for the variability between document sizes

        """
        non_null_text = EhrapyMedcat._filter_null_values(obs, text_column)
        formatted_text_column = EhrapyMedcat._format_df_column(non_null_text, text_column)
        results = ep_cat.cat.multiprocessing(formatted_text_column, batch_size_chars=batch_size_chars, nproc=n_proc)
        flattened_res = EhrapyMedcat._flatten_annotated_results(results)
        # sort for row number in ascending order and reset index to keep index updated
        ep_cat.annotated_results = EhrapyMedcat._annotated_results_to_df(flattened_res).sort_values(by=["row_nr"]).reset_index(drop=True)

    @staticmethod
    def get_annotation_overview(ep_cat: MedCAT, n: int = 10, status: str = "Affirmed", save_to_csv: bool = False) -> None:
        """Provide an overview for the annotation results. An overview will look like the following:
           cui (the CUI), nsubjects (from how many rows this one got extracted), type_ids (TUIs), name(name of the entitiy), perc_subjects (how many rows relative
           to absolute number of rows)

        Args:
            ep_cat: The current MedCAT object which holds all infos on medcat analysis with ehrapy.
            n: Basically the parameter for head() of pandas Dataframe. How many of the most common entities should be shown?
            status: One of "Affirmed" (default), "Other" or "Both". Displays stats for either only affirmed entities, negated ones or both.
            save_to_csv: Whether to save the overview dataframe to a local .csv file in the current working directory or not.

        Returns:
            A pandas DataFrame with the overview stats.
        """
        df = EhrapyMedcat._filter_df_by_status(ep_cat.annotated_results, status)
        # group by CUI as this is a unique identifier per entity
        grouped = df.groupby("cui")
        # get absolute number of rows with this entity
        # note for overview, only one TUI and type is shown (there shouldn't be much situations were multiple are even possible or useful)
        res = grouped.agg(
            {
                "pretty_name": (lambda x: next(iter(set(x)))),
                "type_ids": (lambda x: next(iter(x))[0]),
                "types": (lambda x: next(iter(x))[0]),
                "row_nr": "nunique",
            }
        )
        res = res.rename(columns={"row_nr": "n_patient_visit"})
        # relative amount of patient visits with the specific entity to all patient visits (or rows in the original data)
        res["n_patient_visit_percent"] = (res["n_patient_visit"] / df["row_nr"].nunique()) * 100
        res.round({"n_patient_visit_percent": 1})
        # save to csv if wanted
        if save_to_csv:
            res.to_csv(".")

        overview_table = EhrapyMedcat._df_to_rich_table(res.nlargest(n, "n_patient_visit"))
        console = Console()
        console.print(overview_table)

    @staticmethod
    def _add_binary_column_to_obs(ep_cat: MedCAT, adata: AnnData, name: str) -> None:
        """Adds a binary column to obs (temporarily) for plotting infos extracted from freetext.
        Indicates whether the specific entity to color by has been found in that row or not.

        """
        # only extract affirmed entities
        df = EhrapyMedcat._filter_df_by_status(ep_cat.annotated_results, "Affirmed")
        # currently, only the pretty_name column is supported
        # TODO: add support for CUI/types and hierarchical entities (in case of diseases implement this with ICD mapping feature)
        adata.obs[name] = df.groupby("row_nr").agg({"pretty_name": (lambda x: any(x.isin([name])))})

    @staticmethod
    def _annotated_results_to_df(flattened_results: dict) -> pd.DataFrame:
        """Turn the flattened annotated results into a pandas DataFrame and remove duplicates."""
        df = pd.DataFrame.from_dict(flattened_results, orient="index")
        # remove duplicate entries; for example when a single entity like a disease is mentioned multiple times without any meaningful context changes
        # Example: The patient suffers from Diabetes. Cause of the Diabetes, he receives drug X.
        df.drop_duplicates(subset=["cui", "row_nr", "meta_anns"])
        return df

    @staticmethod
    def _flatten_annotated_results(annotation_results: dict) -> dict:
        """Flattens the nested set (usually 5 level nested) of annotation results.
        annotation_results is just a simple flattened dict with infos on all entities found

        """
        flattened_annotated_dict = {}
        entry_nr = 0

        # row numbers where the text column is located in the original data
        for row_id in annotation_results.keys():
            # all entities extracted from a given row
            entities = annotation_results[row_id]["entities"]
            for entity_id in entities.keys():
                # ignore tokens for now
                if entity_id != "tokens":
                    single_entity = {"row_nr": row_id}
                    entity = entities[entity_id]
                    # iterate over all info attributes of a single entity found in a specific row
                    for entity_key in entity.keys():
                        if entity_key in ["pretty_name", "cui", "type_ids", "types"]:
                            single_entity[entity_key] = entities[entity_id][entity_key]
                        elif entity_key == "meta_anns":
                            single_entity[entity_key] = entities[entity_id][entity_key]["Status"]["value"]
                    flattened_annotated_dict[entry_nr] = single_entity
                    entry_nr += 1
        return flattened_annotated_dict

    @staticmethod
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

    @staticmethod
    def _filter_null_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Filter null values of a given column and return that column without the null values"""
        return pd.DataFrame(df[column][~df[column].isnull()])

    @staticmethod
    def _filter_df_by_status(df: pd.DataFrame, status: str) -> pd.DataFrame:
        """Util function to filter passed dataframe by status."""
        df_res = df
        if status != "Both":
            if status not in {"Affirmed", "Other"}:
                raise StatusNotSupportedError(f"{status} is not available. Please use either Affirmed, Other or Both!")
            mask = df["meta_anns"].values == status
            df_res = df[mask]
        return df_res

    @staticmethod
    def _df_to_rich_table(df: pd.DataFrame) -> Table:
        """Convert a pandas dataframe to a rich Table"""
        table = Table(show_header=True, header_style="bold magenta")

        for column in df.columns:
            table.add_column(str(column))

        for _, value_list in enumerate(df.values.tolist()):
            row = []
            row += [str(x) for x in value_list]
            table.add_row(*row)

        # Update the style of the table
        table.row_styles = ["none", "dim"]
        table.box = box.SIMPLE_HEAD

        return table


class StatusNotSupportedError(Exception):
    pass

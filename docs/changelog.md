# Changelog

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.13.0

### 🚀 Features

* Transitioning from AnnData to EHRData
`EHRData` replaces `AnnData` as ehrapy's core data structure to better support time-series electronic health record data.
The key enhancement is native support for 3D tensors (observations × variables × timesteps) alongside the existing 2D matrices, enabling efficient storage of longitudinal patient data.
A new `.tem` DataFrame provides time-point annotations, complementing the existing `.obs` and `.var` annotations for comprehensive temporal data description.
While `EHRData` maintains full backward compatibility with AnnData's API, users can now seamlessly work with time-series data and leverage specialized methods for temporal analysis.
Existing code using `AnnData` objects will continue to work, but migration to `EHRData` is strongly recommended to access enhanced time-series functionality.
* The preferred central data object is now `EHRData` ([#908](https://github.com/theislab/ehrapy/pull/908)) @eroell
* The `layers` argument is now available for all functions operating on X or layers ([#908](https://github.com/theislab/ehrapy/pull/908)) @eroell
* Update expected behaviour of `io.read_fhir` ([#922](https://github.com/theislab/ehrapy/pull/922)) @eroell
* Move `mimic_2`, `mimic_2_preprocessed`, `diabetes_130_raw`, `diabetes_130_fairlearn` to `ehrdata.dt` ([#908](https://github.com/theislab/ehrapy/pull/908))
* Deprecate all `ep.dt.*`, refer to datasets in `ehrdata` ([#908](https://github.com/theislab/ehrapy/pull/908)) @eroell
* Support Python 3.14 ([#996](https://github.com/theislab/ehrapy/pull/996)) @Zethson
* Move kaplan_meier & cox_ph plots to holoviews ([#995](https://github.com/theislab/ehrapy/pull/995)) @Zethson
* Longitudinal normalization ([#958](https://github.com/theislab/ehrapy/pull/958)) @agerardy
* Add interactive `ols` plot ([#992](https://github.com/theislab/ehrapy/pull/992)) @Zethson
* Longitudinal and new qc_metrics ([#967](https://github.com/theislab/ehrapy/pull/967)) @sueoglu
* Simple Impute for timeseries ([#975](https://github.com/theislab/ehrapy/pull/975)) @eroell
* Simple implementation of balanced sampling ([#937](https://github.com/theislab/ehrapy/pull/937)) @sueoglu
* Add Sankey diagram visualization functions ([#989](https://github.com/theislab/ehrapy/pull/989)) @sueoglu
* Add `ep.pl.timeseries()` to visualize variables over time ([#994](https://github.com/theislab/ehrapy/pull/994)) @sueoglu
* Add GPU CI & skeleton ([#998](https://github.com/theislab/ehrapy/pull/998)) @Zethson
* Add FAMD ([#976](https://github.com/theislab/ehrapy/pull/976)) @Zethson
* 3D enabled implementation of ep.pp.filter_observations, ep.pp.filter_features ([#953](https://github.com/theislab/ehrapy/pull/953)) @sueoglu
* Add time series distances ([#954](https://github.com/theislab/ehrapy/pull/954)) @Zethson

### 🐛 Bug Fixes

* All green if GPU skipped ([#1000](https://github.com/theislab/ehrapy/pull/1000)) @Zethson
* Fix neighbors with timeseries ([#973](https://github.com/theislab/ehrapy/pull/973)) @eroell
* Fix use_rep when X none ([#969](https://github.com/theislab/ehrapy/pull/969)) @eroell
* Fix missing_values_barplot errors ([#963](https://github.com/theislab/ehrapy/pull/963)) @sueoglu
* Fix CR notebook ([#939](https://github.com/theislab/ehrapy/pull/939)) @Zethson

### 🧰 Maintenance

* Update actions ([#977](https://github.com/theislab/ehrapy/pull/977)) @Zethson
* Cleanup simple_impute tests ([#974](https://github.com/theislab/ehrapy/pull/974)) @eroell
* Move to ehrdata 0.0.10 ([#971](https://github.com/theislab/ehrapy/pull/971)) @eroell
* Improved notebook CI ([#959](https://github.com/theislab/ehrapy/pull/959)) @Zethson
* Switch to template ([#960](https://github.com/theislab/ehrapy/pull/960)) @Zethson
* Tests for more plots ([#919](https://github.com/theislab/ehrapy/pull/919)) @sueoglu
* Lowerbound cvxpy ([#935](https://github.com/theislab/ehrapy/pull/935)) @Zethson
* Optimize var_metrics ([#927](https://github.com/theislab/ehrapy/pull/927)) @Zethson
* Refactor Dask usage pattern ([#926](https://github.com/theislab/ehrapy/pull/926)) @Zethson
* Add cover to README & remove some tokens ([#923](https://github.com/theislab/ehrapy/pull/923)) @Zethson
* Update test coverage reporting ([#918](https://github.com/theislab/ehrapy/pull/918)) @eroell
* Fix changelog links ([#915](https://github.com/theislab/ehrapy/pull/915)) @Zethson
* Fixed structure of Returns in _rank_features_groups.py documentation ([#911](https://github.com/theislab/ehrapy/pull/911)) @agerardy
* Add EHRData transition code ([#897](https://github.com/theislab/ehrapy/pull/897)) @Zethson @eroell
* Make test that downloads dermatology dataset more robust ([#906](https://github.com/theislab/ehrapy/pull/906)) @Zethson
* Update image source in README.md ([#986](https://github.com/theislab/ehrapy/pull/986)) @eroell
* Fix plot docs formatting ([#952](https://github.com/theislab/ehrapy/pull/952)) @Zethson
* Typo in the documentation of ehrapy.data.mimic_2_preprocessed ([#917](https://github.com/theislab/ehrapy/pull/917)) @sueoglu

## v0.12.1

### 🚀 Features

* Make dowhy optional & remove medcat ([#903](https://github.com/theislab/ehrapy/pull/903)) @Zethson
* Add about page & improve citations ([#902](https://github.com/theislab/ehrapy/pull/902)) @Zethson
* Overhaul doc structure ([#895](https://github.com/theislab/ehrapy/pull/895)) @Zethson
* Move to biome & improve CI & reenable CR ([#890](https://github.com/theislab/ehrapy/pull/890)) @Zethson
* Clean up Round - cut down anndata extension functionality ([#880](https://github.com/theislab/ehrapy/pull/880)) @eroell

## v0.12.0

### 🚀 Features

* Improved KM plot data depth and functionality ([#853](https://github.com/theislab/ehrapy/pull/853)) @aGuyLearning
* New Feature: Forestplot for CoxPH model ([#838](https://github.com/theislab/ehrapy/pull/838)) @aGuyLearning
* Datatype Support in Quality Control and Impute ([#865](https://github.com/theislab/ehrapy/pull/865)) @aGuyLearning
* Revamp survival analysis interface ([#842](https://github.com/theislab/ehrapy/pull/842)) @aGuyLearning
* Improve submodule documentation ([#859](https://github.com/theislab/ehrapy/pull/859)) @Zethson
* Update Kaplan Meier plots in survival analysis notebook ([#864](https://github.com/theislab/ehrapy/pull/864)) @aGuyLearning

### 🐛 Bug Fixes

* Pass all non-nan features along desired var_names to impute (KNN) ([#867](https://github.com/theislab/ehrapy/pull/867)) @nicolassidoux
* Remove Syntax warnings ([#869](https://github.com/theislab/ehrapy/pull/869)) @Zethson
* Fix test_norm_power_group ([#862](https://github.com/theislab/ehrapy/pull/862)) @Zethson

### 🧰 Maintenance

* Fix a typo in `pl.paga_compare`: `pos` -> `pos,` ([#846](https://github.com/theislab/ehrapy/pull/846)) @VladimirShitov

## v0.11.0

### ✨ Features

* Add array type handling for normalization ([#835](https://github.com/theislab/ehrapy/pull/835)) @eroell @Zethson

### 🐛 Bug Fixes

* Fix scipy array support ([#844](https://github.com/theislab/ehrapy/pull/844)) @Zethson
* Fix casting to float when assigning numeric values; fixes normalization of integer arrays ([#837](https://github.com/theislab/ehrapy/pull/837)) @eroell

## v0.9.0 & 0.10.0

### 🚀 Features

* Make all imputation methods consistent in regard to encoding requirements ([#827](https://github.com/theislab/ehrapy/pull/827)) @nicolassidoux
* Add approximate KNN backend ([#791](https://github.com/theislab/ehrapy/pull/791)) @nicolassidoux
* Improve survival analysis interface ([#825](https://github.com/theislab/ehrapy/pull/825)) @aGuyLearning
* Python 3.12 support ([#794](https://github.com/theislab/ehrapy/pull/794)) @Lilly-May
* Python 3.10+ & use uv for docs & fix RTD & support numpy 2 ([#830](https://github.com/theislab/ehrapy/pull/830)) @Zethson

### 🐛 Bug Fixes

* move_to_x: Fix name of non-implemented argument "copy" to "copy_x", implement & test ([#832](https://github.com/theislab/ehrapy/pull/832)) @eroell
* Contributing typo fix ([#821](https://github.com/theislab/ehrapy/pull/821)) @aGuyLearning
* Fix miceforest ([#800](https://github.com/theislab/ehrapy/pull/800)) @Zethson
* style: == to is for type comparison ([#774](https://github.com/theislab/ehrapy/pull/774)) @eroell

## v0.8.0

### 🚀 Features

* remove pyyaml & explicit scikit-learn ([#729](https://github.com/theislab/ehrapy/pull/729)) @Zethson
* Remove fancyimpute ([#728](https://github.com/theislab/ehrapy/pull/728)) @Zethson
* Unify feature type detection ([#724](https://github.com/theislab/ehrapy/pull/724)) @Lilly-May
* catplot ([#721](https://github.com/theislab/ehrapy/pull/721)) @eroell
* Simplify ehrapy ([#719](https://github.com/theislab/ehrapy/pull/719)) @Zethson
* Use __all__ ([#715](https://github.com/theislab/ehrapy/pull/715)) @Zethson
* Add bias detection to preprocessing ([#690](https://github.com/theislab/ehrapy/pull/690)) @Lilly-May
* Use lamin logger ([#707](https://github.com/theislab/ehrapy/pull/707)) @Zethson
* Add faiss backend for KNN imputation ([#704](https://github.com/theislab/ehrapy/pull/704)) @Zethson
* Build RTD docs with uv ([#700](https://github.com/theislab/ehrapy/pull/700)) @Zethson
* Refactor feature importance ranking ([#698](https://github.com/theislab/ehrapy/pull/698)) @Zethson
* Simplify CI ([#694](https://github.com/theislab/ehrapy/pull/694)) @Zethson
* Refactor outliers and IQR ([#692](https://github.com/theislab/ehrapy/pull/692)) @Zethson
* Calculation of feature importances in a supervised setting ([#677](https://github.com/theislab/ehrapy/pull/677)) @Lilly-May
* Speed up winsorize ([#681](https://github.com/theislab/ehrapy/pull/681)) @Zethson
* Remove notebook prefix in tutorial URLs ([#679](https://github.com/theislab/ehrapy/pull/679)) @Zethson
* Add cohort tracking notebook ([#678](https://github.com/theislab/ehrapy/pull/678)) @Zethson
* Switch to uv ([#674](https://github.com/theislab/ehrapy/pull/674)) @Zethson
* Style: typing of _scale_func_group ([#727](https://github.com/theislab/ehrapy/pull/727)) @eroell
* Improved support of encoded features in detect_bias ([#725](https://github.com/theislab/ehrapy/pull/725)) @Lilly-May
* Enable Synchronous dataloader write ([#722](https://github.com/theislab/ehrapy/pull/722)) @wxicu
* Feature scaling on training set when computing feature importances ([#716](https://github.com/theislab/ehrapy/pull/716)) @Lilly-May
* add batch-wise normalization argument ([#711](https://github.com/theislab/ehrapy/pull/711)) @eroell
* add functools.wraps to type check ([#705](https://github.com/theislab/ehrapy/pull/705)) @eroell
* add bias notebook to list of notebooks ([#696](https://github.com/theislab/ehrapy/pull/696)) @eroell
* basic sampling ([#686](https://github.com/theislab/ehrapy/pull/686)) @eroell
* add options for subitles in legend of cohorttrackers barplot ([#688](https://github.com/theislab/ehrapy/pull/688)) @eroell
* doc fix imputation: 70 instead of 30 ([#683](https://github.com/theislab/ehrapy/pull/683)) @eroell

### 🐛 Bug Fixes

* Encoded dtype to float32 instead of np.number ([#714](https://github.com/theislab/ehrapy/pull/714)) @Zethson
* Fix feature importance warnings ([#708](https://github.com/theislab/ehrapy/pull/708)) @Zethson
* Remove notebook prefix in tutorial URLs ([#679](https://github.com/theislab/ehrapy/pull/679)) @Zethson
* fix name of log_rogistic_aft to log_logistic_aft ([#676](https://github.com/theislab/ehrapy/pull/676)) @eroell

### 🧰 Maintenance

* Remove notebook prefix in tutorial URLs ([#679](https://github.com/theislab/ehrapy/pull/679)) @Zethson
* Add cohort tracking notebook ([#678](https://github.com/theislab/ehrapy/pull/678)) @Zethson
* knni amendments ([#706](https://github.com/theislab/ehrapy/pull/706)) @eroell

## v0.7.0

### 🚀 Features

* Cohort Tracker ([#658](https://github.com/theislab/ehrapy/pull/658)) @eroell
* change diabetes-130 datasets which are provided ([#672](https://github.com/theislab/ehrapy/pull/672)) @eroell
* More sa functions ([#664](https://github.com/theislab/ehrapy/pull/664)) @fatisati
* Coxphfitter ([#643](https://github.com/theislab/ehrapy/pull/643)) @fatisati
* Implement little's test ([#667](https://github.com/theislab/ehrapy/pull/667)) @Zethson
* Improve test design ([#651](https://github.com/theislab/ehrapy/pull/651)) @Zethson
* Improve QC docstring ([#639](https://github.com/theislab/ehrapy/pull/639)) @Zethson
* Refactor _missing_values calculation ([#638](https://github.com/theislab/ehrapy/pull/638)) @Zethson

### 🐛 Bug Fixes

* Fix one-hot encoding tests ([#644](https://github.com/theislab/ehrapy/pull/644)) @Zethson

## v0.6.0

### 🚀 Features

#### Breaking changes

* Move information on numerical/non_numerical/encoded_non_numerical from .uns to .var ([#630](https://github.com/theislab/ehrapy/pull/630)) @eroell

Make older AnnData objects compatible using

```python
def move_type_info_from_uns_to_var(adata, copy=False):
    """Move type information from adata.uns to adata.var['ehrapy_column_type'].

    The latter is the current, updated flavor used by ehrapy.
    """
    if copy:
        adata = adata.copy()

    adata.var['ehrapy_column_type'] = 'unknown'

    if 'numerical_columns' in adata.uns.keys():
        for key in adata.uns['numerical_columns']:
            adata.var.loc[key, 'ehrapy_column_type'] = 'numeric'
    if 'non_numerical_columns' in adata.uns.keys():
        for key in adata.uns['non_numerical_columns']:
            adata.var.loc[key, 'ehrapy_column_type'] = 'non_numeric'
    if 'encoded_non_numerical_columns' in adata.uns.keys():
        for key in adata.uns['encoded_non_numerical_columns']:
            adata.var.loc[key, 'ehrapy_column_type'] = 'non_numeric_encoded'

    if copy:
        return adata
```

#### New features

* Medcat refresh ([#623](https://github.com/theislab/ehrapy/pull/623)) @eroell
* Rank features groups obs ([#622](https://github.com/theislab/ehrapy/pull/622)) @eroell
* Add FHIR tutorial and simplify code ([#626](https://github.com/theislab/ehrapy/pull/626)) @Zethson
* Add input checks for imputers ([#625](https://github.com/theislab/ehrapy/pull/625)) @Zethson
* Removed unused dependencies ([#615](https://github.com/theislab/ehrapy/pull/615)) @Zethson
* Refactor encoding ([#588](https://github.com/theislab/ehrapy/pull/588)) @Zethson

### 🐛 Bug Fixes

* Use fixtures for preprocessing tests ([#577](https://github.com/theislab/ehrapy/pull/577)) @Zethson

### 🧰 Maintenance

* Refactoring ([#627](https://github.com/theislab/ehrapy/pull/627)) @Zethson
* Add FHIR tutorial and simplify code ([#626](https://github.com/theislab/ehrapy/pull/626)) @Zethson
* pre-commit ([#587](https://github.com/theislab/ehrapy/pull/587)) @Zethson
* Small edits ([#599](https://github.com/theislab/ehrapy/pull/599)) @eroell

## v0.5.0

### 🚀 Features

* Add g-tests for rank features group ([#546](https://github.com/theislab/ehrapy/pull/546)) @VladimirShitov
* Causal Inference with dowhy ([#502](https://github.com/theislab/ehrapy/pull/502)) @timtreis
* Remove MuData support ([#545](https://github.com/theislab/ehrapy/pull/545)) @Zethson

### 🐛 Bug Fixes

* Fixed reading format warnings  ([#569](https://github.com/theislab/ehrapy/pull/569)) @namsaraeva
* Fixed inability to normalize AnnData that does not require encoding  ([#568](https://github.com/theislab/ehrapy/pull/568)) @namsaraeva
* Fixed adata.uns["non_numericlal_columns"] being empty in mimic_2 dataset ([#567](https://github.com/theislab/ehrapy/pull/567)) @namsaraeva

## v0.4.0

### 🚀 Features

* Add Synthea dataset ([#510](https://github.com/theislab/ehrapy/pull/510)) @namsaraeva
* Added tiny examples to every function ([#498](https://github.com/theislab/ehrapy/pull/498)) @namsaraeva
* add a title parameter ([#494](https://github.com/theislab/ehrapy/pull/494)) @xinyuejohn
* Changed the hue of grey ([#493](https://github.com/theislab/ehrapy/pull/493)) @namsaraeva
* Logger info message when writing to .h5ad files ([#458](https://github.com/theislab/ehrapy/pull/458)) @namsaraeva
* Modified docstrings ([#533](https://github.com/theislab/ehrapy/pull/533)) @namsaraeva
* Added examples to missing modules ([#531](https://github.com/theislab/ehrapy/pull/531)) @namsaraeva
* Allow Python 3.11 ([#523](https://github.com/theislab/ehrapy/pull/523)) @Zethson
* Add test_kmf_logrank ([#516](https://github.com/theislab/ehrapy/pull/516)) @Zethson
* Add scget functions ([#484](https://github.com/theislab/ehrapy/pull/484)) @Zethson
* Add FHIR parsing support ([#463](https://github.com/theislab/ehrapy/pull/463)) @Zethson
* Add new tutorial & switch to python 3.10 ([#454](https://github.com/theislab/ehrapy/pull/454)) @Zethson
* Add docs group ([#437](https://github.com/theislab/ehrapy/pull/437)) @Zethson
* Add thefuzz ([#434](https://github.com/theislab/ehrapy/pull/434)) @Zethson

### 🐛 Bug Fixes

* Fix CI ([#524](https://github.com/theislab/ehrapy/pull/524)) @Zethson
* Error message and minor fixes, issue #447 ([#504](https://github.com/theislab/ehrapy/pull/504)) @namsaraeva
* fix quality control ([#495](https://github.com/theislab/ehrapy/pull/495)) @xinyuejohn
* Fix MacOS CI ([#435](https://github.com/theislab/ehrapy/pull/435)) @Zethson

### 🧰 Maintenance

* Add test_kmf_logrank ([#516](https://github.com/theislab/ehrapy/pull/516)) @Zethson
* Add scget functions ([#484](https://github.com/theislab/ehrapy/pull/484)) @Zethson
* Add new tutorial & switch to python 3.10 ([#454](https://github.com/theislab/ehrapy/pull/454)) @Zethson

## v0.3.0

### 🚀 Features

* Add winsorize, clip quantiles and filter quantiles ([#418](https://github.com/theislab/ehrapy/pull/418)) @Zethson
* Remove PDF support ([#430](https://github.com/theislab/ehrapy/pull/430)) @Zethson
* Logging instance, issue #246 ([#426](https://github.com/theislab/ehrapy/pull/426)) @namsaraeva
* Negative values offset ([#420](https://github.com/theislab/ehrapy/pull/420)) @Zethson
* Missing values visualization, ref issue #271 ([#419](https://github.com/theislab/ehrapy/pull/419)) @namsaraeva
* Add copy_obs parameter to move_to_obs ([#404](https://github.com/theislab/ehrapy/pull/404)) @namsaraeva
* add anova_glm function ([#400](https://github.com/theislab/ehrapy/pull/400)) @xinyuejohn
* issue #397 "check for neighbors run before UMAP" fixed ([#401](https://github.com/theislab/ehrapy/pull/401)) @namsaraeva
* add more tutorials to CI ([#382](https://github.com/theislab/ehrapy/pull/382)) @Zethson
* add support for reading multiple files into Pandas DFs & adapted MIMIC-III Demo ([#386](https://github.com/theislab/ehrapy/pull/386)) @Zethson
* #321: Add X_only option for reading ([#380](https://github.com/theislab/ehrapy/pull/380)) @Imipenem

### 🐛 Bug Fixes

* KeyError fix issue #423 ([#428](https://github.com/theislab/ehrapy/pull/428)) @namsaraeva
* fix qc_metrics bug ([#425](https://github.com/theislab/ehrapy/pull/425)) @xinyuejohn
* df_to_anndata logical XOR to OR, issue #422 ([#429](https://github.com/theislab/ehrapy/pull/429)) @namsaraeva
* Fix docs CI ([#392](https://github.com/theislab/ehrapy/pull/392)) @Zethson
* small fix in the qc_metrics() example ([#407](https://github.com/theislab/ehrapy/pull/407)) @namsaraeva

### 🧰 Maintenance

* Add winsorize, clip quantiles and filter quantiles ([#418](https://github.com/theislab/ehrapy/pull/418)) @Zethson
* Remove PDF support ([#430](https://github.com/theislab/ehrapy/pull/430)) @Zethson
* Negative values offset ([#420](https://github.com/theislab/ehrapy/pull/420)) @Zethson
* Missing values visualization, ref issue #271 ([#419](https://github.com/theislab/ehrapy/pull/419)) @namsaraeva
* Fix docs CI ([#392](https://github.com/theislab/ehrapy/pull/392)) @Zethson

## v0.2.0

### 🚀 Features

* Important cookietemple template update 2.1.0 released! ([#343](https://github.com/theislab/ehrapy/pull/343)) @Zethson
* add chronic kidney disease dataloader ([#301](https://github.com/theislab/ehrapy/pull/301)) @xinyuejohn
* dataloader for diabetes dataset ([#292](https://github.com/theislab/ehrapy/pull/292)) @HorlavaNastassya
* Add X_only option for reading ([#380](https://github.com/theislab/ehrapy/pull/380)) @Imipenem
* MedCAT API improvements & function renaming ([#381](https://github.com/theislab/ehrapy/pull/381)) @Zethson
* minor changes ([#379](https://github.com/theislab/ehrapy/pull/379)) @xinyuejohn
* add functions related to survival analysis ([#371](https://github.com/theislab/ehrapy/pull/371)) @xinyuejohn
* MedCat [#101]: extract biomedical concepts/entities from (free) text ([#367](https://github.com/theislab/ehrapy/pull/367)) @Imipenem
* Add heart dataset to docs ([#377](https://github.com/theislab/ehrapy/pull/377)) @xinyuejohn
* add heart disease data set to ehrapy ([#376](https://github.com/theislab/ehrapy/pull/376)) @xinyuejohn
* add highly_variable_features ([#364](https://github.com/theislab/ehrapy/pull/364)) @xinyuejohn
* add SoftImpute and IterativeSVD to imputation ([#353](https://github.com/theislab/ehrapy/pull/353)) @xinyuejohn
* (#307) Improve KNN with n_neighbours parameter ([#365](https://github.com/theislab/ehrapy/pull/365)) @Imipenem
* add furo theme & switch to markdown ([#359](https://github.com/theislab/ehrapy/pull/359)) @Zethson
* add several datasets and change Docstring examples ([#355](https://github.com/theislab/ehrapy/pull/355)) @xinyuejohn
* Add ability to compare laboratory measurements to reference values ([#352](https://github.com/theislab/ehrapy/pull/352)) @Zethson
* (Feature) New read API #263 ([#351](https://github.com/theislab/ehrapy/pull/351)) @Imipenem
* (Feature) Set index column #305 ([#350](https://github.com/theislab/ehrapy/pull/350)) @Imipenem
* Add encoded parameter to all new datasets amd fix import ([#336](https://github.com/theislab/ehrapy/pull/336)) @xinyuejohn
* (FEATURE) #314: Autodetect binary (0,1) columns ([#327](https://github.com/theislab/ehrapy/pull/327)) @Imipenem
* (FEATURE) Display QC metrics of var #239 ([#323](https://github.com/theislab/ehrapy/pull/323)) @Imipenem
* Add several dataset loaders ([#322](https://github.com/theislab/ehrapy/pull/322)) @xinyuejohn
* (FEATURE) Improve type_overview #306 ([#308](https://github.com/theislab/ehrapy/pull/308)) @Imipenem
* Feature/deep translator integration ([#303](https://github.com/theislab/ehrapy/pull/303)) @MxMstrmn
* remove CLI module ([#298](https://github.com/theislab/ehrapy/pull/298)) @Zethson
* Improve missforest interface ([#284](https://github.com/theislab/ehrapy/pull/284)) @Zethson
* Add example calls and preview images to all plotting functions ([#289](https://github.com/theislab/ehrapy/pull/289)) @xinyuejohn
* add heart failure dataloader ([#291](https://github.com/theislab/ehrapy/pull/291)) @Zethson
* add highly_variable_features ([#364](https://github.com/theislab/ehrapy/pull/364)) @xinyuejohn
* add SoftImpute and IterativeSVD to imputation ([#353](https://github.com/theislab/ehrapy/pull/353)) @xinyuejohn
* add furo theme & switch to markdown ([#359](https://github.com/theislab/ehrapy/pull/359)) @Zethson

### 🐛 Bug Fixes

* (FIX) #255: Encode mutates input adata object ([#348](https://github.com/theislab/ehrapy/pull/348)) @Imipenem
* (FIX) Write .h5ad files ([#347](https://github.com/theislab/ehrapy/pull/347)) @Imipenem
* Fix #331: Improved autodetect docs ([#344](https://github.com/theislab/ehrapy/pull/344)) @Imipenem
* Add encoded parameter to all new datasets amd fix import ([#336](https://github.com/theislab/ehrapy/pull/336)) @xinyuejohn
* (FIX) Autodetect encode + specify encode mode for autodetect ([#310](https://github.com/theislab/ehrapy/pull/310)) @Imipenem

### 🧰 Maintenance

* MedCAT API improvements & function renaming ([#381](https://github.com/theislab/ehrapy/pull/381)) @Zethson
* add functions related to survival analysis ([#371](https://github.com/theislab/ehrapy/pull/371)) @xinyuejohn
* MedCat [#101]: extract biomedical concepts/entities from (free) text ([#367](https://github.com/theislab/ehrapy/pull/367)) @Imipenem
* Add heart dataset to docs ([#377](https://github.com/theislab/ehrapy/pull/377)) @xinyuejohn
* remove CLI module ([#298](https://github.com/theislab/ehrapy/pull/298)) @Zethson

## v0.1.0

### 🚀 Features

* Input and output of CSVs, PDFs, h5ad files
* Several encoding modes (one-hot, label, ...)
* Several imputation methods (simple, KNN, MissForest, ...)
* Several normalization methods (log, scale, ...)
* Full Scanpy API support
* Initial MedCAT integration
* DeepL & Google Translator support

# Ecosystem

ehrapy focuses on reusable building blocks for exploratory analysis of heterogeneous epidemiology and electronic health record data. More specialized workflows can be developed as separate ecosystem packages that build on ehrapy without expanding the core API beyond its general-purpose scope.

## Drug Screening

[`ehrapy-drug-screening`](https://github.com/xushenbo/ehrapy-drug-screening) provides a dedicated home for high-throughput drug safety screening workflows, including exposure episode construction, self-controlled cohort analyses, grouped screening across drug hierarchy levels, indication mapping, post-processing, and optional LLM-assisted review.

Keeping this workflow in a separate package makes the functions easier to discover as a coherent workflow, allows ehrapy to remain focused on foundational EHR analysis, and gives the workflow its own documentation and step-by-step tutorials.

The ecosystem package documentation hosts the detailed usage guide and step-by-step notebooks for the workflow.
